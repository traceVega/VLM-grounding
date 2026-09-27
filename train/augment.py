"""Cross-scene negatives: a reviewed target or sibling description from scene A, asked on a
same-category image B in which no candidate satisfies it -> {"bbox_2d": null}.

    python -m train.augment build  [--per-expr 1] [--seed 0]          -> pairs.jsonl
    python -m train.augment verify [--limit N] [--checker gemini-b]    -> verify.jsonl (resumable)
    python -m train.augment export                                     -> cross_v1.jsonl (item format)

Why: the same words are a positive on A and a negative on B, so no text feature predicts
null, and the null path sees "no candidate fits at all" next to the flipped-detail negatives.
Verification: the batched Gemini checker looks at every candidate of B with the
description's clauses as statements; a candidate with every statement yes/unclear/unparsed
satisfies it and the pair is dropped (conservative).  About 0.03 USD per pair.
Items carry `group` = B (the image) and `src_group` = A; the trainers exclude an item if
either scene is held out.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from datagen import checker_api as A
from datagen import common as C
from shared.harness import prompts
from train import data as D

ROOT = D.TRAIN_ROOT / "augment"
RUNS = ("test50c", "poc155")


def load_scenes() -> dict[tuple[str, str], dict]:
    out = {}
    for run in RUNS:
        root = C.run_root(run)
        if not (root / "scenes.jsonl").is_file():
            continue
        for sc in C.read_jsonl(root / "scenes.jsonl"):
            sc["run"] = run
            out[(run, sc["image_id"])] = sc
    return out


def load_clauses() -> dict[tuple[str, str, str], list[str]]:
    """(run, image_id, 'target'|'sibling') -> clauses."""
    out = {}
    for run in RUNS:
        root = C.run_root(run)
        for r in C.read_jsonl(root / "write.jsonl") if (root / "write.jsonl").is_file() else []:
            out[(run, r["image_id"], "target")] = r.get("clauses") or []
        for r in C.read_jsonl(root / "sibling.jsonl") if (root / "sibling.jsonl").is_file() else []:
            out[(run, r["image_id"], "sibling")] = r.get("clauses_sibling") or []
    return out


def build(args) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    scenes = load_scenes()
    clauses = load_clauses()
    by_cat: dict[str, list[tuple[str, str]]] = {}
    for key, sc in scenes.items():
        by_cat.setdefault(sc["category"], []).append(key)
    rng = random.Random(args.seed)
    pairs = []
    for it in D.load_items():
        if it["kind"] not in ("positive", "sibling_positive"):
            continue
        which = "target" if it["kind"] == "positive" else "sibling"
        cl = clauses.get((it["run"], it["group"], which))
        if not cl:
            continue
        cands = [k for k in by_cat.get(it["category"], []) if k[1] != it["group"]]
        rng.shuffle(cands)
        for run_b, img_b in cands[: args.per_expr]:
            pairs.append({"pair_id": f"{it['id']}->{run_b}:{img_b}", "src_id": it["id"], "src_kind": it["kind"], "src_run": it["run"],
                          "src_group": it["group"], "run": run_b, "group": img_b, "category": it["category"],
                          "expression": it["expression"], "clauses": cl})
    with open(ROOT / "pairs.jsonl", "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    cats = {}
    for p in pairs:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print(f"pairs {len(pairs)} over {len(cats)} categories: {dict(sorted(cats.items(), key=lambda kv: -kv[1])[:8])}")


def verify(args) -> None:
    from google.genai import types

    model, _, _ = A.MODELS[args.checker]
    client = A._client()
    tmpl = prompts.load("verifier_clauses_batched", non_kill=True)
    cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=512,
                                      thinking_config=types.ThinkingConfig(thinking_level="low"))
    scenes = load_scenes()
    pairs = [json.loads(l) for l in open(ROOT / "pairs.jsonl", encoding="utf-8")]
    out = ROOT / "verify.jsonl"
    done = {json.loads(l)["pair_id"] for l in open(out, encoding="utf-8")} if out.is_file() else set()
    todo = [p for p in pairs if p["pair_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"verify[{model}]: {len(pairs)} pairs, {len(done)} done, {len(todo)} to run", flush=True)
    usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0}

    def ask(views, text):
        for attempt in range(A.RETRIES):
            try:
                resp = client.models.generate_content(model=model, contents=[*views, text], config=cfg)
                u = resp.usage_metadata
                usage["calls"] += 1
                if u:
                    usage["prompt_tokens"] += u.prompt_token_count or 0
                    usage["output_tokens"] += u.candidates_token_count or 0
                    usage["thought_tokens"] += getattr(u, "thoughts_token_count", 0) or 0
                return resp.text or ""
            except Exception:
                time.sleep(min(60, 2 ** attempt))
        return ""

    t0 = time.time()
    for n, p in enumerate(todo, 1):
        sc = scenes[(p["run"], p["group"])]
        full = Image.open(C.image_path(p["group"])).convert("RGB")
        small, s = A._shrink(full)
        items = "\n".join(f"{k + 1}. The {sc['category']} {c}." for k, c in enumerate(p["clauses"]))
        text = tmpl.render(category=sc["category"], items=items)

        def one(inst):
            views = [C.outline(small, [v * s for v in inst["box"]], "red"), C.closeup(full, inst["box"])]
            return A.parse_batched(ask(views, text), len(p["clauses"]))

        with ThreadPoolExecutor(max_workers=A.WORKERS) as ex:
            verdicts = list(ex.map(one, sc["instances"]))
        per = {str(i["iid"]): v for i, v in zip(sc["instances"], verdicts)}
        satisfiers = [int(k) for k, v in per.items() if all(x != "no" for x in v)]
        rec = {**p, "verdicts": per, "satisfiers": satisfiers, "usable": not satisfiers}
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n % 10 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] usable so far (this session) see file; {usage['calls']} calls, "
                  f"{usage['prompt_tokens']} in / {usage['output_tokens']}+{usage['thought_tokens']} out tokens, {(time.time() - t0) / 60:.1f} min", flush=True)
    (ROOT / "verify_usage.json").write_text(json.dumps(usage), encoding="utf-8")


def export(args) -> None:
    scenes = load_scenes()
    rows = [json.loads(l) for l in open(ROOT / "verify.jsonl", encoding="utf-8")]
    items = []
    for r in rows:
        if not r["usable"]:
            continue
        sc = scenes[(r["run"], r["group"])]
        items.append({"image": str(C.image_path(r["group"])), "image_wh": sc["image_wh"], "group": r["group"], "src_group": r["src_group"],
                      "category": sc["category"], "label": sc["label"], "run": r["run"], "source": "cross",
                      "n_candidates": len(sc["instances"]), "id": "cross:" + r["pair_id"], "kind": "negative",
                      "expression": r["expression"], "answer": {"bbox_2d": None}, "target_iid": None,
                      "cross": {"src_id": r["src_id"], "src_kind": r["src_kind"]}})
    out = D.TRAIN_ROOT / "augment" / args.out
    with open(out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} usable of {len(rows)} verified -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "verify", "export"])
    ap.add_argument("--per-expr", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checker", default="gemini-b")
    ap.add_argument("--out", default="cross_v1.jsonl")
    args = ap.parse_args()
    {"build": build, "verify": verify, "export": export}[args.stage](args)


if __name__ == "__main__":
    main()
