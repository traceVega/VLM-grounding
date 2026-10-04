"""General-capability check outside grounding: MMStar (1,500 multiple-choice questions, evaluation only).

    python -m train.mmstar_eval --tag mmstar_base
    python -m train.mmstar_eval --tag mmstar_grpo_coa11 --adapter ~/vlmg-data/train/grpo_coa11/adapter
    python -m train.mmstar_eval --compare mmstar_base mmstar_grpo_coa11

The question (options included) is asked with the standard direct-answer instruction; the first letter A-D of the reply is
the answer.  Records go to eval/<tag>/mmstar.jsonl (resumable); the summary gives accuracy overall and per category, and
--compare an exact McNemar test between two tags on the same items.  The adapter is on for every question: this measures
whether training for grounding / rejection changed the model's general visual question answering.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

RAW = Path.home() / "vlmg-data/raw/mmstar/hf/mmstar.parquet"
INSTRUCTION = "\nAnswer with the option's letter from the given choices directly."


def items() -> list[dict]:
    import pyarrow.parquet as pq

    d = pq.read_table(RAW).to_pydict()
    out = []
    for i in range(len(d["index"])):
        img = d["image"][i]
        out.append({"id": f"mmstar:{d['index'][i]}", "question": d["question"][i], "answer": d["answer"][i].strip().upper(),
                    "category": d["category"][i], "l2_category": d["l2_category"][i], "image_bytes": img["bytes"] if isinstance(img, dict) else img})
    return out


def letter(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", (text or "").strip().upper())
    return m.group(1) if m else None


def run(args) -> None:
    import torch
    from PIL import Image

    from train import data as D
    from train import eval_suite as E

    out = D.TRAIN_ROOT / "eval" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mmstar.jsonl"
    done = E.done_ids(path)
    todo = [it for it in items() if it["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"  mmstar: {len(done)} done, {len(todo)} to run", flush=True)
    if todo:
        model, processor = E.load_model(args.model, args.adapter)
        tok = processor.tokenizer
        eos = {t for t in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None}
        processor.tokenizer.padding_side = "left"
        t0 = time.time()
        for b0 in range(0, len(todo), args.batch):
            chunk = todo[b0: b0 + args.batch]
            images = [Image.open(io.BytesIO(it["image_bytes"])).convert("RGB") for it in chunk]
            texts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["question"] + INSTRUCTION}]}],
                                                   tokenize=False, add_generation_prompt=True) for it in chunk]
            inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
            n_prompt = inputs["input_ids"].shape[1]
            with open(path, "a", encoding="utf-8") as fh:
                for j, it in enumerate(chunk):
                    raw = processor.decode(E._trim(gen[j][n_prompt:].tolist(), eos), skip_special_tokens=True)
                    pred = letter(raw)
                    fh.write(json.dumps({"id": it["id"], "category": it["category"], "l2_category": it["l2_category"], "answer": it["answer"],
                                         "pred": pred, "correct": pred == it["answer"], "raw": raw[:200]}) + "\n")
            if (b0 // args.batch) % 20 == 0:
                n = b0 + len(chunk)
                print(f"    [{n}/{len(todo)}] {(time.time() - t0) / n:.2f}s/item", flush=True)
    summarize(args.tag)


def load(tag: str) -> dict[str, dict]:
    from train import data as D

    f = D.TRAIN_ROOT / "eval" / tag / "mmstar.jsonl"
    return {r["id"]: r for r in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip())}


def summarize(tag: str) -> dict:
    rows = list(load(tag).values())
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r["correct"])
    s = {"n": len(rows), "acc": round(sum(r["correct"] for r in rows) / max(1, len(rows)), 4), "unparsed": sum(r["pred"] is None for r in rows),
         "by_category": {c: round(sum(v) / len(v), 4) for c, v in sorted(by.items())}}
    print(f"[{tag}] " + json.dumps(s))
    return s


def compare(a: str, b: str) -> None:
    ra, rb = load(a), load(b)
    ids = [i for i in ra if i in rb]
    x = sum(ra[i]["correct"] and not rb[i]["correct"] for i in ids)
    y = sum(rb[i]["correct"] and not ra[i]["correct"] for i in ids)
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)
    acc = lambda r: 100 * sum(r[i]["correct"] for i in ids) / len(ids)
    print(f"{a} {acc(ra):.1f} vs {b} {acc(rb):.1f} on {len(ids)} items | only {a} correct {x}, only {b} correct {y}, McNemar p = {p:.3g}")
    cats = sorted({ra[i]["category"] for i in ids})
    for c in cats:
        sub = [i for i in ids if ra[i]["category"] == c]
        print(f"  {c:28s} {100 * sum(ra[i]['correct'] for i in sub) / len(sub):5.1f} vs {100 * sum(rb[i]['correct'] for i in sub) / len(sub):5.1f}  (n = {len(sub)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag")
    ap.add_argument("--model", default="4b")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("TAG_A", "TAG_B"))
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        assert a.tag, "--tag is required"
        run(a)
