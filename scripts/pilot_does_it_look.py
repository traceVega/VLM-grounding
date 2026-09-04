"""The fast question: does the model look at the image, or pattern-match?

    python scripts/pilot_does_it_look.py --n 300

This is the whole research idea in one measurement, and it needs no data that is
not already built.  Take an image where the referent's class occurs exactly once
-- so the class label is an unambiguous referring expression -- and ask
Qwen3-VL to ground it twice: on the original, and on the edit where that object
has been removed.

If the model keeps returning the same box on an image the object has left, then
the box was not read off the pixels, and the idea holds.  If it moves the box or
declines, the concern is unfounded and weeks are saved.

**This is exploratory, and deliberately not K2.**

* It runs `--non-kill`, which the design exempts from the K2 freeze, and no
  non-kill run id can enter a kill table (SPEC Section 2).
* The expression is a class label, not GroundingME's full referring expression,
  so the numbers are not P15's and must never be quoted as them.
* Nothing here may change P8 to P21 afterwards. Deciding whether to continue is
  a legitimate use of a pilot; retuning the design around its result is the
  contamination P20 exists to prevent.

What it can establish is whether the phenomenon is there at all, which is the
only question that decides whether the careful version is worth building.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HF = "Qwen/Qwen3-VL-8B-Instruct"
REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SET_NAME = "openimages_pool"


def unambiguous_referents() -> dict[str, dict]:
    """``image_id -> referent`` for images where the referent's class occurs once.

    This filter is what makes a bare class label serve as a referring
    expression: with one instance of the class in the picture, "the cup" can
    only mean one thing.  With two, the model may well point at the other one
    and be marked wrong for it.

    The count is taken over the *image's own instances*, not from
    ``n_head_noun_instances`` -- that column is populated for K2 scenes, and on
    the K1 pool it is 1 everywhere, so filtering on it silently keeps every
    image and lets ambiguous ones through.  Measured: it admitted 9,692 of 9,692
    and ORIGINAL accuracy came out at 17%.
    """
    from shared import paths

    directory = paths.PREPARED / SET_NAME / "instances"
    referents: dict[str, dict] = {}
    labels_per_image: dict[str, list[str]] = {}
    for shard in sorted(Path(directory).glob("shard-*.parquet")):
        table = pq.read_table(
            shard, columns=["image_id", "source", "label", "bbox_xyxy_px"]
        )
        for row in table.to_pylist():
            if row["label"]:
                labels_per_image.setdefault(row["image_id"], []).append(row["label"].lower())
            if row["source"] == "referent" and row["label"]:
                referents.setdefault(row["image_id"], row)

    out: dict[str, dict] = {}
    for image_id, ref in referents.items():
        same = sum(1 for lab in labels_per_image.get(image_id, []) if lab == ref["label"].lower())
        if same == 1:
            out[image_id] = ref
    return out


def remove_edits() -> dict[str, dict]:
    """``image_id -> the mask REMOVE edit`` from the K1 bank."""
    from shared import paths

    shards = sorted((paths.EDITS_ROOT / "index").glob("shard-*.parquet"))
    table = pa.concat_tables([pq.read_table(s) for s in shards])
    cols = {n: table.column(n).to_pylist() for n in table.column_names}
    out: dict[str, dict] = {}
    for i in range(table.num_rows):
        if cols["operator"][i] != "REMOVE":
            continue
        out[cols["image_id"][i]] = {
            "window_xyxy_px": cols["window_xyxy_px"][i],
            "window_path": cols["window_path"][i],
            "mask_rle": cols["mask_rle"][i],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("tables/pilot_does_it_look.json"))
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText

    from idea91.edits.build import load_edited, read_image
    from shared import paths
    from shared.harness import parsers as P
    from shared.harness import prompts
    from shared.harness import tokens as T

    referents = unambiguous_referents()
    edits = remove_edits()
    shared_ids = sorted(set(referents) & set(edits))
    random.Random(0).shuffle(shared_ids)
    shared_ids = shared_ids[: args.n]
    print(f"{len(referents)} unambiguous referents, {len(edits)} REMOVE edits, "
          f"{len(shared_ids)} usable; running {len(shared_ids)}\n")

    template = prompts.load("grounding_qwen3vl_primary")
    processor = T.load_capped_processor(HF, REV)
    T.assert_cap_is_in_force(processor)
    model = AutoModelForImageTextToText.from_pretrained(
        HF, revision=REV, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    def ask(image: np.ndarray, expr: str) -> str:
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text",
                                                     "text": template.render(expr=expr)}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[Image.fromarray(image)], text=text,
                           return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    image_dir = paths.RAW / "openimages" / "images"
    records = []
    for n, image_id in enumerate(shared_ids, 1):
        ref, edit = referents[image_id], edits[image_id]
        try:
            original = read_image(image_dir / f"{image_id}.jpg")
            removed = load_edited(original, edit, paths.EDITS_ROOT)
        except Exception as exc:
            print(f"  {image_id}: {type(exc).__name__}: {exc}")
            continue
        wh = (original.shape[1], original.shape[0])
        expr = f"the {ref['label']}"

        parsed = {}
        for name, image in (("original", original), ("removed", removed)):
            raw = ask(image, expr)
            parsed[name] = P.parse_qwen3vl(
                raw, convention=P.RELATIVE_1000, original_wh=wh, sent_wh=wh
            )

        gt = tuple(float(v) for v in ref["bbox_xyxy_px"])
        o, r = parsed["original"], parsed["removed"]
        records.append({
            "image_id": image_id,
            "expr": expr,
            "gt_box": list(gt),
            "original_type": o.output_type,
            "removed_type": r.output_type,
            "original_box": list(o.box_xyxy_px) if o.box_xyxy_px else None,
            "removed_box": list(r.box_xyxy_px) if r.box_xyxy_px else None,
            "original_iou_gt": P.iou(o.box_xyxy_px, gt) if o.box_xyxy_px else None,
            "same_box": (
                P.iou(r.box_xyxy_px, o.box_xyxy_px)
                if (o.box_xyxy_px and r.box_xyxy_px) else None
            ),
        })
        if n % 25 == 0:
            print(f"  [{n}/{len(shared_ids)}]", flush=True)

    report(records, args.out)


def report(records: list[dict], out: Path) -> None:
    n = len(records)
    correct = [r for r in records if (r["original_iou_gt"] or 0) >= 0.5]
    print(f"\n{'='*66}\n{n} items\n")
    print(f"ORIGINAL: box returned {sum(r['original_type'] == 'box' for r in records)/max(n,1):.0%}, "
          f"declined {sum(r['original_type'] == 'none' for r in records)/max(n,1):.0%}")
    print(f"  of those, IoU >= 0.5 with ground truth: {len(correct)}/{n} "
          f"({len(correct)/max(n,1):.0%})  <- the model can do the task at all\n")

    if not correct:
        print("no ORIGINAL-correct items; the pilot cannot say anything")
        return

    box_on_removed = sum(r["removed_type"] == "box" for r in correct)
    declined = sum(r["removed_type"] == "none" for r in correct)
    same = [r for r in correct if (r["same_box"] or 0) >= 0.5]
    print(f"On the {len(correct)} items the model got right, after the object is REMOVED:")
    print(f"  still returns a box            {box_on_removed}/{len(correct)} "
          f"({box_on_removed/len(correct):.0%})")
    print(f"  declines ('bbox_2d': null)     {declined}/{len(correct)} "
          f"({declined/len(correct):.0%})")
    print(f"  SAME BOX (IoU >= 0.5 vs its own ORIGINAL box)  {len(same)}/{len(correct)} "
          f"({len(same)/len(correct):.0%})   <- the headline")
    ious = [r["same_box"] for r in correct if r["same_box"] is not None]
    if ious:
        print(f"  median IoU(removed, original) = {np.median(ious):.3f}")
    print()
    print("Reading it: a high same-box rate means the box survived the object leaving")
    print("the picture, so it was not read off the pixels. A high decline rate means")
    print("the model noticed, and the concern is unfounded.")
    print(f"{'='*66}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": n, "records": records}, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
