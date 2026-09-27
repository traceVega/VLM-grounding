"""Observation rationales for every verdict cell, written by Gemini through the API
(the strong-teacher counterpart of train.rationales, same protocol and output schema).

    python -m train.rationales_gemini [--limit N] [--workers 6] [--model gemini-3.1-pro-preview]

Per (scene, candidate instance) of the training label matrices: two batched calls with the
instance outlined in the full image plus a close-up crop,
  1. blind verdicts (configs/prompts/verifier_clauses_batched.txt) without the labels - a
     rationale is kept by train.traces only where this agrees with the label;
  2. rationales (configs/prompts/rationale_batched.txt) given the label verdicts.
Output: $VLMG_DATA_ROOT/train/rationales_gemini.jsonl (same rows as rationales.jsonl) and
rationales_gemini_usage.json (calls and tokens).  Resumable.  Select it for training with
VLMG_RATIONALES=~/vlmg-data/train/rationales_gemini.jsonl.
Images are our own OpenImages scenes (CC-BY); GroundingME never passes through here.
Key: GEMINI_API_KEY (WSL ~/.vlmg-secrets, sourced from ~/.profile); never printed.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from datagen import common as C
from datagen.checker_api import RETRIES, _client, _shrink, parse_batched
from shared.harness import prompts
from train import data as D
from train import traces as TR
from train.rationales import cells_by_instance, parse_lines

OUT = D.TRAIN_ROOT / "rationales_gemini.jsonl"
USAGE = D.TRAIN_ROOT / "rationales_gemini_usage.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    from google.genai import types

    cells = cells_by_instance()
    done = set()
    if OUT.is_file():
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add((r["run"], r["image_id"], r["iid"]))
    todo = [k for k in cells if k not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"rationales[{args.model}]: {len(cells)} instances, {sum(len(v) for v in cells.values())} cells, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    recs = TR.records()
    t_blind = prompts.load("verifier_clauses_batched", non_kill=True)
    t_rat = prompts.load("rationale_batched", non_kill=True)
    client = _client()
    cfg_blind = types.GenerateContentConfig(temperature=0.0, max_output_tokens=512, thinking_config=types.ThinkingConfig(thinking_level="low"))
    cfg_rat = types.GenerateContentConfig(temperature=0.0, max_output_tokens=1024, thinking_config=types.ThinkingConfig(thinking_level="low"))
    usage = json.load(open(USAGE, encoding="utf-8")) if USAGE.is_file() else {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    lock = threading.Lock()

    def ask(views, text: str, cfg, n: int, complete) -> str:
        """One call with retries; re-asks up to twice when fewer than n lines were parsed."""
        last, best = None, ""
        for attempt in range(RETRIES):
            try:
                resp = client.models.generate_content(model=args.model, contents=[*views, text], config=cfg)
                u = resp.usage_metadata
                with lock:
                    usage["calls"] += 1
                    usage["prompt_tokens"] += (u.prompt_token_count or 0) if u else 0
                    usage["output_tokens"] += (u.candidates_token_count or 0) if u else 0
                    usage["thought_tokens"] += (getattr(u, "thoughts_token_count", 0) or 0) if u else 0
                out = (resp.text or "").strip()
                if complete(out) or attempt >= 2:
                    return out
                best = best or out
                text = text + f"\nYou must answer all {n} statements, one line each."
            except Exception as e:  # rate limit / transient
                last = e
                time.sleep(min(60, (2 ** attempt) + random.random()))
        return best or f"ERROR: {last}"

    def one(key):
        run, image_id, iid = key
        rec = recs.get((run, image_id))
        inst = next((i for i in rec["instances"] if i["iid"] == iid), None) if rec else None
        if inst is None:
            return None
        conds = list(cells[key].keys())
        labels = [cells[key][c] for c in conds]
        full = Image.open(C.image_path(image_id)).convert("RGB")
        small, s = _shrink(full)
        views = [C.outline(small, [v * s for v in inst["box"]], "red"), C.closeup(full, inst["box"])]
        cat = rec["category"]
        n = len(conds)
        items_blind = "\n".join(f"{k + 1}. The {cat} {c}." for k, c in enumerate(conds))
        raw_b = ask(views, t_blind.render(category=cat, items=items_blind), cfg_blind, n, lambda o: parse_batched(o, n).count("unparsed") == 0)
        blind = parse_batched(raw_b, n)
        items_rat = "\n".join(f"{k + 1}. The {cat} {c}. Verdict: {v}." for k, (c, v) in enumerate(zip(conds, labels)))
        raw_r = ask(views, t_rat.render(category=cat, items=items_rat), cfg_rat, n, lambda o: sum(x is not None for x in parse_lines(o, n)) == n)
        rat = parse_lines(raw_r, n)
        return {"run": run, "image_id": image_id, "iid": iid, "conditions": conds, "label": labels, "blind": blind, "rationale": rat}

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row in ex.map(one, todo):
            n_done += 1
            if row is not None:
                with lock:
                    with open(OUT, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    USAGE.write_text(json.dumps(usage), encoding="utf-8")
            if n_done % 25 == 0 or n_done == len(todo):
                print(f"  [{n_done}/{len(todo)}] {(time.time() - t0) / 60:.1f} min  usage {usage}", flush=True)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
