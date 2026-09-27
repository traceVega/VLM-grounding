"""Stage `check --checker gemini`: the same per-instance, per-detail verification as
`checker.py`, answered by Gemini through the API instead of a local model.

Same protocol as the local checker so the two are comparable cell by cell:
full image with one instance outlined in red + a close-up crop of it, one detail
per call, first word parsed as yes / no / unclear.  Differences: the close-up is
cut from the full-resolution source (the API is not bound by the 1024 px cap of
the local processors), calls within a scene run in parallel, and token usage is
recorded per scene so the cost is measurable.

Key: GEMINI_API_KEY in the environment (WSL: ~/.vlmg-secrets, sourced from ~/.profile).
Images sent here are OpenImages (CC-BY); never route GroundingME images through this.
"""

from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from datagen import common as C
from shared.harness import prompts

MODELS = {
    # key: (model id, output file, batched)
    "gemini": ("gemini-3.1-pro-preview", "checker_gemini.jsonl", False),          # one call per detail (test50c baseline)
    "gemini-flash": ("gemini-3.8-flash", "checker_gemini_flash.jsonl", False),
    "gemini-b": ("gemini-3.1-pro-preview", "checker_gemini_b.jsonl", True),        # one call per candidate, all details
    "gemini-flash-b": ("gemini-3.8-flash", "checker_gemini_flash_b.jsonl", True),
}
_LINE = None


def parse_batched(text: str, n: int) -> list[str]:
    """'k: yes|no|unclear' lines -> verdict per statement 1..n ('unparsed' when missing)."""
    import re

    global _LINE
    if _LINE is None:
        _LINE = re.compile(r"^\s*\**\s*(\d+)\s*[.:)\-]+\s*\**\s*(yes|no|unclear)\b", re.IGNORECASE | re.MULTILINE)
    out = ["unparsed"] * n
    for m in _LINE.finditer(text or ""):
        k = int(m.group(1)) - 1
        if 0 <= k < n and out[k] == "unparsed":
            out[k] = m.group(2).lower()
    return out
MAX_SIDE = 1024          # full-scene view, same scale the local checker saw
WORKERS = 6
RETRIES = 6


def _client():
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set (WSL: ~/.vlmg-secrets, sourced from ~/.profile)")
    from google import genai

    return genai.Client()


def _shrink(image: Image.Image, max_side: int = MAX_SIDE) -> tuple[Image.Image, float]:
    w, h = image.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        return image.resize((round(w * s), round(h * s)), Image.LANCZOS), s
    return image, 1.0


def run(run: str, limit: int | None, checker: str = "gemini", target_only: bool = False) -> None:
    from google.genai import types

    model, fname, batched = MODELS[checker]
    client = _client()
    root = C.run_root(run)
    scenes = C.by_image(C.read_jsonl(root / "scenes.jsonl"))
    written = C.by_image(C.read_jsonl(root / "write.jsonl"))
    out = root / fname
    done = C.done_keys(out)
    todo = [i for i in written if (i,) not in done]
    if limit:
        todo = todo[:limit]
    print(f"check[{model}]: {len(written)} written, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    tmpl = prompts.load("verifier_clauses_batched" if batched else "verifier_clause_instance_v2", non_kill=True)
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=512,  # thinking tokens count against this budget: 64 left 16% of answers empty (test50c)
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    usage_lock = threading.Lock()

    def ask(views, category: str, clause, usage: dict) -> str:
        items = None
        if batched:  # clause is the list of statements for this candidate
            items = "\n".join(f"{k + 1}. The {category} {c}." for k, c in enumerate(clause))
            text = tmpl.render(category=category, items=items)
        else:
            text = tmpl.render(category=category, clause=clause)
        last = None
        best = ""
        for attempt in range(RETRIES):
            try:
                resp = client.models.generate_content(model=model, contents=[*views, text], config=cfg)
                u = resp.usage_metadata
                with usage_lock:
                    usage["calls"] += 1
                    usage["prompt_tokens"] += (u.prompt_token_count or 0) if u else 0
                    usage["output_tokens"] += (u.candidates_token_count or 0) if u else 0
                    usage["thought_tokens"] += (getattr(u, "thoughts_token_count", 0) or 0) if u else 0
                out = (resp.text or "").strip()
                if not batched:
                    return out
                missing = parse_batched(out, len(clause)).count("unparsed")
                if missing == 0 or attempt >= 2:
                    return out
                best = best or out  # incomplete answer: ask again, at most twice, insisting on every line
                text = tmpl.render(category=category, items=items) + f"\nYou must answer all {len(clause)} statements, one line each."
            except Exception as e:  # rate limit / transient
                last = e
                time.sleep(min(60, (2 ** attempt) + random.random()))
        return best or f"ERROR: {last}"

    t0 = time.time()
    total = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    for n, image_id in enumerate(todo, 1):
        sc, w = scenes[image_id], written[image_id]
        full = Image.open(C.image_path(image_id)).convert("RGB")
        small, s = _shrink(full)
        cat = sc["category"]
        usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
        insts = [i for i in sc["instances"] if i["is_target"] or not target_only]
        jobs = []  # (iid, kind, k, views, clause-or-statements)
        for inst in insts:
            box_small = [v * s for v in inst["box"]]
            views = [C.outline(small, box_small, "red"), C.closeup(full, inst["box"])]
            if batched:
                stmts = list(w["clauses"]) + ([w["clause_neg"]] if w.get("clause_neg") else [])
                jobs.append((inst["iid"], "batch", 0, views, stmts))
                continue
            for k, clause in enumerate(w["clauses"]):
                jobs.append((inst["iid"], "clause", k, views, clause))
            if w.get("clause_neg"):
                jobs.append((inst["iid"], "neg", 0, views, w["clause_neg"]))
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            answers = list(ex.map(lambda j: ask(j[3], cat, j[4], usage), jobs))
        per_inst = []
        for inst in insts:
            row = {"iid": inst["iid"], "is_target": inst["is_target"], "verdicts": [], "raw": [],
                   "neg_verdict": None, "neg_raw": None}
            for j, a in zip(jobs, answers):
                if j[0] != inst["iid"]:
                    continue
                if j[1] == "batch":
                    n_cl = len(w["clauses"])
                    vs = parse_batched(a, len(j[4]))
                    row["verdicts"], row["raw"] = vs[:n_cl], [a] * n_cl
                    if w.get("clause_neg"):
                        row["neg_verdict"], row["neg_raw"] = vs[n_cl], a
                elif j[1] == "clause":
                    row["verdicts"].append(C.yes_no_unclear(a))
                    row["raw"].append(a)
                else:
                    row["neg_verdict"], row["neg_raw"] = C.yes_no_unclear(a), a
            per_inst.append(row)
        target_row = next(r for r in per_inst if r["is_target"])
        rec = {
            "image_id": image_id,
            "instances": per_inst,
            "target_all_yes": all(v == "yes" for v in target_row["verdicts"]) if target_row["verdicts"] else None,
            "zero_satisfier": all(r["neg_verdict"] == "no" for r in per_inst) if w.get("clause_neg") else None,
            "siblings_distinct": (all(any(v == "no" for v in r["verdicts"]) for r in per_inst if not r["is_target"])
                                  if not target_only else None),
            "checker": model, "prompt": tmpl.name, "batched": batched, "usage": usage,
        }
        C.append_jsonl(out, rec)
        for k in total:
            total[k] += usage[k]
        errors = sum(1 for a in answers if a.startswith("ERROR"))
        print(f"  [{n}/{len(todo)}] {image_id} {usage['calls']} calls{' (' + str(errors) + ' errors)' if errors else ''}, "
              f"{(time.time() - t0) / 60:.1f} min, target_ok={rec['target_all_yes']} zero={rec['zero_satisfier']} "
              f"distinct={rec['siblings_distinct']}", flush=True)
    print(f"done: {total['calls']} calls, prompt {total['prompt_tokens']:,} tok, output {total['output_tokens']:,} tok, "
          f"thoughts {total['thought_tokens']:,} tok, {(time.time() - t0) / 60:.1f} min", flush=True)
