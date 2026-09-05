"""Before spending anyone's afternoon: can the removal experiment even be run here?

    python scripts/gme_original_precheck.py            # run, then report
    python scripts/gme_original_precheck.py --report   # re-read the saved run

No editing, no annotation, no judge -- every GroundingME item on its own
unmodified image, once. Two things come out of the same pass.

**The sample size.** The removal experiment only counts items whose ORIGINAL box
is correct, since a model that never found the referent cannot be shown to have
stopped looking at it. GroundingME is hard -- the paper puts the best of 25
MLLMs at 45.1% -- so that filter may leave too few items per dimension to
compare them, and the comparison is the experiment. Measuring it costs nothing
and settles whether `Limited` is worth reviewing at all.

**A rejection baseline that needs no editor.** 201 items have no answer in the
picture: the expression describes something that is not there, on a real
photograph with no inpainting anywhere. The paper reports 20 of 25 models
scoring exactly 0% on this. Our removals produced 50% abstention, and the
difference between those two numbers is either the prompt or the artifact. This
row is the only one that can tell them apart without an edit.

Exploratory, `--non-kill`. The K2 freeze forbids scoring a K2 item before B0b,
so nothing here may enter a kill table or change P8 to P21; ORIGINAL accuracy
and the rejection row are read as feasibility, not as results.
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
OUT = Path("tables/gme_original.jsonl")
ATTEMPTS = Path("tables/gme_original.attempts")
MAX_ATTEMPTS = 2

#: The keyword screen for expressions whose answer moves when the referent
#: goes: remove the third leaf from the left and the fourth becomes the third,
#: so a box on it is arguably right and the item cannot test necessity. Rough
#: on purpose -- it is a budgeting number here, and the review decides per item.
ORDINAL = ("first", "second", "third", "fourth", "fifth", "leftmost", "rightmost",
           "topmost", "bottommost", "from the left", "from the right",
           "from the top", "from the bottom", "closest", "nearest", "farthest")


def items() -> list[dict]:
    import pyarrow.parquet as pq

    from shared import paths

    table = pq.read_table(paths.DATA_ROOT / "prepared/groundingme/items.parquet")
    col = {n: table.column(n).to_pylist() for n in table.column_names}
    return [{
        "item_id": col["item_id"][i],
        "expr": col["expr"][i],
        "dimension": col["dimension"][i],
        "size_bin": col["size_bin"][i],
        "image_path": col["image_path"][i],
        "gt_boxes": col["gt_boxes_xyxy_px"][i],
        "n_gt": col["n_gt"][i],
    } for i in range(table.num_rows)]


def done_ids() -> set[str]:
    if not OUT.is_file():
        return set()
    return {json.loads(x)["item_id"]
            for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()}


def poisoned() -> set[str]:
    if not ATTEMPTS.is_file():
        return set()
    counts: dict[str, int] = collections.Counter(
        x.strip() for x in ATTEMPTS.read_text(encoding="utf-8").splitlines() if x.strip())
    finished = done_ids()
    return {k for k, v in counts.items() if v >= MAX_ATTEMPTS and k not in finished}


def run(limit: int | None) -> None:
    from transformers import AutoModelForImageTextToText

    from shared import paths
    from shared.harness import parsers as P
    from shared.harness import prompts
    from shared.harness import tokens as T

    rows = items()
    if limit:
        rows = rows[:limit]
    total, done, skip = len(rows), done_ids(), poisoned()
    rows = [r for r in rows if r["item_id"] not in done and r["item_id"] not in skip]
    print(f"{total} GroundingME items; {len(done)} done, {len(skip)} skipped, "
          f"{len(rows)} to run\n")
    if not rows:
        return

    template = prompts.load("grounding_qwen3vl_primary")
    processor = T.load_capped_processor(HF, REV)
    T.assert_cap_is_in_force(processor)
    tokenizer = processor.tokenizer
    null_ids, box_ids = _first_ids(tokenizer, [" null", "null", " Null"]), \
        _first_ids(tokenizer, [" [", "["])
    model = AutoModelForImageTextToText.from_pretrained(
        HF, revision=REV, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for n, item in enumerate(rows, 1):
        with open(ATTEMPTS, "a", encoding="utf-8") as fh:
            fh.write(item["item_id"] + "\n")
        try:
            image = Image.open(paths.DATA_ROOT / item["image_path"]).convert("RGB")
        except Exception as exc:
            print(f"  {item['item_id']}: {type(exc).__name__}: {exc}")
            continue

        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": template.render(expr=item["expr"])}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0][inputs["input_ids"].shape[1]:].tolist()
        raw = processor.decode(seq, skip_special_tokens=True)
        parsed = P.parse_qwen3vl(raw, convention=P.RELATIVE_1000,
                                 original_wh=image.size, sent_wh=image.size)
        box = list(parsed.box_xyxy_px) if parsed.box_xyxy_px else None

        rec = {k: item[k] for k in ("item_id", "dimension", "size_bin", "n_gt")}
        rec.update({
            "image_wh": list(image.size),
            "output_type": parsed.output_type,
            "box": box,
            "iou_gt": max((iou(box, g) for g in item["gt_boxes"]), default=None) if box else None,
            "ordinal": any(w in item["expr"].lower() for w in ORDINAL),
            **_decision(seq, out.scores, tokenizer, null_ids, box_ids),
        })
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
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
    print(f"{len(records)} items\n")

    positives = [r for r in records if r["n_gt"] == 1]
    print("ORIGINAL accuracy -- the filter the removal experiment has to survive")
    print(f"  {'dimension':<16} {'n':>5} {'correct':>14} {'declined':>10} {'ordinal':>9}")
    by_dim = collections.defaultdict(list)
    for r in positives:
        by_dim[r["dimension"]].append(r)
    for dim, rows in sorted(by_dim.items()):
        ok = sum(1 for r in rows if (r["iou_gt"] or 0) >= 0.5)
        no = sum(1 for r in rows if r["output_type"] == "none")
        od = sum(1 for r in rows if r["ordinal"])
        print(f"  {dim:<16} {len(rows):5d} {ok:6d} {ok / len(rows):6.0%} "
              f"{no / len(rows):9.0%} {od / len(rows):8.0%}")
    total_ok = sum(1 for r in positives if (r["iou_gt"] or 0) >= 0.5)
    print(f"  {'ALL':<16} {len(positives):5d} {total_ok:6d} "
          f"{total_ok / max(len(positives), 1):6.0%}")

    reject = [r for r in records if r["n_gt"] == 0]
    if reject:
        none = sum(1 for r in reject if r["output_type"] == "none")
        med = float(np.median([r["p_null_norm"] for r in reject
                               if r["p_null_norm"] is not None]))
        print(f"\nREJECTION row -- {len(reject)} real photographs, nothing edited,")
        print("the described object simply is not in them")
        print(f"  abstained          {none:4d}/{len(reject)} = {none / len(reject):.0%}")
        print(f"  median p(null)     {med:.3e}")
        print("  published: 20 of 25 MLLMs score exactly 0% here (non-thinking mode)")
        print("  our edited removals: 50% abstention, median p(null) 6.2e-01")

    print("\nProjected usable items for the removal experiment")
    print(f"  {'dimension':<16} {'n':>5} {'-ordinal':>10} {'x correct':>11} "
          f"{'x removal .6':>13}")
    for dim, rows in sorted(by_dim.items()):
        keep = [r for r in rows if not r["ordinal"]]
        ok = sum(1 for r in keep if (r["iou_gt"] or 0) >= 0.5)
        print(f"  {dim:<16} {len(rows):5d} {len(keep):10d} {ok:11d} {round(ok * 0.6):13d}")
    print("\n  (removal success assumed 0.6; it was 0.46 on OpenImages, and these"
          "\n   referents are far smaller, so that is a floor rather than a guess)")


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
