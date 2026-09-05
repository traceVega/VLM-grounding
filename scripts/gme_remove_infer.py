"""Ask the same question again, on the image the referent has left.

    python scripts/gme_remove_infer.py

The ORIGINAL condition is already run (`tables/gme_original.jsonl`); this is the
other half of the pair. Identical prompt, identical image but for the referent's
own pixels, and the same two-way decision recorded at the token after
`{"bbox_2d":`.

Run before the human review rather than after, so that when the review finishes
the numbers are already there. Items the review later marks `not_clean` or
`other_match` drop out then; running them now costs eight minutes and buys the
artifact control for free, exactly as `not_clean` did on the class-label set.

Exploratory, `--non-kill`.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.pilot_abstain_signal import _decision, _first_ids, iou

Image.MAX_IMAGE_PIXELS = None

HF = "Qwen/Qwen3-VL-8B-Instruct"
REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
OUT = Path("tables/gme_remove.jsonl")
ATTEMPTS = Path("tables/gme_remove.attempts")
MAX_ATTEMPTS = 2


def done_ids() -> set[str]:
    if not OUT.is_file():
        return set()
    return {json.loads(x)["item_id"]
            for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()}


def poisoned() -> set[str]:
    if not ATTEMPTS.is_file():
        return set()
    counts = collections.Counter(
        x.strip() for x in ATTEMPTS.read_text(encoding="utf-8").splitlines() if x.strip())
    finished = done_ids()
    return {k for k, v in counts.items() if v >= MAX_ATTEMPTS and k not in finished}


def run(limit: int | None) -> None:
    from transformers import AutoModelForImageTextToText

    from idea91.edits.build import load_edited
    from scripts.build_gme_removals import edits_root
    from shared import paths
    from shared.harness import parsers as P
    from shared.harness import prompts
    from shared.harness import tokens as T

    rows = json.loads(Path("tables/gme_removals_index.json").read_text(encoding="utf-8"))
    if limit:
        rows = rows[:limit]
    total, done, skip = len(rows), done_ids(), poisoned()
    rows = [r for r in rows if r["item_id"] not in done and r["item_id"] not in skip]
    print(f"{total} edits; {len(done)} done, {len(skip)} skipped, {len(rows)} to run\n")
    if not rows:
        return

    template = prompts.load("grounding_qwen3vl_primary")
    processor = T.load_capped_processor(HF, REV)
    T.assert_cap_is_in_force(processor)
    tokenizer = processor.tokenizer
    null_ids = _first_ids(tokenizer, [" null", "null", " Null"])
    box_ids = _first_ids(tokenizer, [" [", "["])
    model = AutoModelForImageTextToText.from_pretrained(
        HF, revision=REV, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for n, item in enumerate(rows, 1):
        with open(ATTEMPTS, "a", encoding="utf-8") as fh:
            fh.write(item["item_id"] + "\n")
        try:
            original = np.asarray(
                Image.open(paths.DATA_ROOT / item["image_path"]).convert("RGB"))
            edited = Image.fromarray(load_edited(original, item, edits_root()))
        except Exception as exc:
            print(f"  {item['item_id']}: {type(exc).__name__}: {exc}", flush=True)
            continue

        messages = [{"role": "user", "content": [
            {"type": "image"},
            # The English original, always. The Chinese in the review UI is for
            # the reviewer's eyes and never reaches a model.
            {"type": "text", "text": template.render(expr=item["expr"])}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[edited], text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0][inputs["input_ids"].shape[1]:].tolist()
        raw = processor.decode(seq, skip_special_tokens=True)
        parsed = P.parse_qwen3vl(raw, convention=P.RELATIVE_1000,
                                 original_wh=edited.size, sent_wh=edited.size)
        box = list(parsed.box_xyxy_px) if parsed.box_xyxy_px else None

        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "item_id": item["item_id"],
                "dimension": item["dimension"],
                "gt_box": item["gt_box"],
                "original_box": item["model_box"],
                "removed_type": parsed.output_type,
                "removed_box": box,
                "same_box": iou(box, item["model_box"]) if box else None,
                **_decision(seq, out.scores, tokenizer, null_ids, box_ids),
            }) + "\n")
            fh.flush()

        if n % 25 == 0:
            torch.cuda.empty_cache()
            rate = (time.time() - started) / n
            print(f"  [{n}/{len(rows)}]  {rate:.1f}s/item, "
                  f"{(len(rows) - n) * rate / 60:.0f} min left", flush=True)

    print(f"\n-> {OUT}  ({len(done_ids())} records)")


def report() -> None:
    records = [json.loads(x) for x in
               OUT.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"\n{len(records)} REMOVE records (before the human review filters them)\n")
    by_dim = collections.defaultdict(list)
    for r in records:
        by_dim[r["dimension"]].append(r)
    print(f"  {'dimension':<16} {'n':>5} {'declined':>10} {'same box':>10} "
          f"{'median p(null)':>16}")
    for dim, rows in sorted(by_dim.items()):
        none = sum(1 for r in rows if r["removed_type"] == "none")
        same = sum(1 for r in rows if (r["same_box"] or 0) >= 0.5)
        ps = [r["p_null_norm"] for r in rows if r["p_null_norm"] is not None]
        med = float(np.median(ps)) if ps else float("nan")
        print(f"  {dim:<16} {len(rows):5d} {none / len(rows):9.0%} "
              f"{same / len(rows):9.0%} {med:16.3e}")
    print("\n  provisional: the review has not yet removed failed removals or"
          "\n  items where something else satisfies the expression.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not args.report:
        run(args.limit)
    if not args.no_report:
        report()


if __name__ == "__main__":
    main()
