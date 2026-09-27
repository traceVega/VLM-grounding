"""Observation rationales for every verdict cell, written locally by Qwen3-VL-8B (no API).

    python -m train.rationales [--limit N] [--model 8b]

For every (scene, candidate instance) that appears in a training item's label matrix, the
instance is shown outlined in the full image plus as a close-up, with the union of all
condition texts that any item asks about it.  Two batched calls:
  1. blind verdicts (configs/prompts/verifier_clauses_batched.txt) - the 8B's own yes/no/unclear
     without seeing the labels; a rationale is kept only where this agrees with the label
     (consistency filter, so no rationale is written "knowing the answer" it cannot see);
  2. rationales (configs/prompts/rationale_batched.txt) - given the label verdicts, one clause
     per statement naming what is seen (for "no", the actual value).
Output: $VLMG_DATA_ROOT/train/rationales.jsonl, one row per (run, image_id, iid):
  {"run", "image_id", "iid", "conditions": [...], "label": [...], "blind": [...], "rationale": [...]}
Resumable.  train.traces reads it for the rationale ("rat") trace style.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image

from datagen import common as C
from datagen.checker_api import _shrink, parse_batched
from shared.harness import prompts
from shared.harness import tokens as T
from train import data as D
from train import traces as TR

OUT = D.TRAIN_ROOT / "rationales.jsonl"
_LINE = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(.+?)\s*$")


def parse_lines(text: str, n: int) -> list[str | None]:
    out: list[str | None] = [None] * n
    for line in (text or "").splitlines():
        m = _LINE.match(line)
        if m:
            k = int(m.group(1)) - 1
            if 0 <= k < n and out[k] is None:
                out[k] = m.group(2).strip().strip('"')
    return out


def cells_by_instance() -> dict[tuple[str, str, int], dict[str, str]]:
    """(run, image_id, iid) -> {condition text: label verdict} over all training items."""
    files = [D.EXPORT, D.TRAIN_ROOT / "augment" / "cross_v1.jsonl", D.TRAIN_ROOT / "alt_negatives.jsonl", D.TRAIN_ROOT / "short_items.jsonl"]
    items = []
    for f in files:
        if Path(f).is_file():
            items.extend(D.load_items(Path(f)))
    out: dict[tuple[str, str, int], dict[str, str]] = {}
    for it in items:
        m, _ = TR.label_matrix(it)
        if m is None:
            continue
        for row in m["rows"]:
            d = out.setdefault((it["run"], it["group"], row["iid"]), {})
            for cond, v in zip(m["conditions"], row["verdicts"]):
                d.setdefault(cond, v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="8b", choices=list(D.MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText

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
    print(f"rationales: {len(cells)} instances, {sum(len(v) for v in cells.values())} cells, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    recs = TR.records()
    t_blind = prompts.load("verifier_clauses_batched", non_kill=True)
    t_rat = prompts.load("rationale_batched", non_kill=True)
    hf, rev = D.MODELS[args.model]
    processor = T.load_capped_processor(hf, rev)
    model = AutoModelForImageTextToText.from_pretrained(hf, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def ask(views, text, max_new):
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in views] + [{"type": "text", "text": text}]}]
        chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=views, text=chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    cache: dict[str, tuple] = {}
    for n, key in enumerate(todo, 1):
        run, image_id, iid = key
        rec = recs.get((run, image_id))
        if rec is None:
            continue
        inst = next((i for i in rec["instances"] if i["iid"] == iid), None)
        if inst is None:
            continue
        conds = list(cells[key].keys())
        labels = [cells[key][c] for c in conds]
        if image_id not in cache:
            full = Image.open(C.image_path(image_id)).convert("RGB")
            cache = {image_id: (full, *_shrink(full))}
        full, small, s = cache[image_id]
        views = [C.outline(small, [v * s for v in inst["box"]], "red"), C.closeup(full, inst["box"])]
        cat = rec["category"]
        items_blind = "\n".join(f"{k + 1}. The {cat} {c}." for k, c in enumerate(conds))
        blind = parse_batched(ask(views, t_blind.render(category=cat, items=items_blind), 96), len(conds))
        items_rat = "\n".join(f"{k + 1}. The {cat} {c}. Verdict: {v}." for k, (c, v) in enumerate(zip(conds, labels)))
        rat = parse_lines(ask(views, t_rat.render(category=cat, items=items_rat), args.max_new), len(conds))
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run": run, "image_id": image_id, "iid": iid, "conditions": conds, "label": labels, "blind": blind, "rationale": rat}, ensure_ascii=False) + "\n")
        if n % 25 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
