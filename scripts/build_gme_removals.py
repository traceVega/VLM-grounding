"""Build REMOVE edits for the GroundingME items the model already grounds correctly.

    python scripts/build_gme_removals.py

Only the referent is removed. There is no CONTROL_OBJ here and that is
deliberate: the artifact control this run needs comes out of the human review
for free, as the items marked `not_clean` -- the editor ran on the referent and
the object survived it -- which is the cell that closed the same confound on the
class-label run. A CONTROL_BG hole would be cheap but risky on this set, where a
39-word expression names five landmarks and a randomly placed hole lands on one
of them often enough to invalidate the control.

Only items whose ORIGINAL box was correct are built, since an item the model
never grounded cannot show it has stopped looking. That is the frame P11 already
uses -- pre-specified, data-dependent, drawn after the ORIGINAL condition.

SAM 3 is prompted on the **window**, not the whole image, per P8. These
photographs run to 59 Mpx and a referent occupies a median 0.1% to 2.6% of one,
so a full-image prompt would hand the segmenter a few pixels after its own
internal resize.

Writes to its own edits root. The K1 bank is frozen under B0a and nothing here
may touch it.

Exploratory, `--non-kill`: no number from this bank may enter a kill table.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SET_NAME = "groundingme"
#: Separate from `paths.EDITS_ROOT`, which holds the K1 bank frozen at B0a.
OUT_DIRNAME = "edits_gme"
INDEX = Path("tables/gme_removals_index.json")
ATTEMPTS = Path("tables/gme_removals.attempts")
MAX_ATTEMPTS = 2
#: P9: below this the SAM 3 mask is not the box's object, so the box itself is
#: inpainted and the item carries the flag.
BOX_TO_MASK_MIN = 0.5


def edits_root():
    from shared import paths

    return paths.DATA_ROOT / OUT_DIRNAME


def wanted() -> list[dict]:
    """ORIGINAL-correct positives, with their expression, box and dimension."""
    import pyarrow.parquet as pq

    from shared import paths

    correct = {}
    for line in Path("tables/gme_original.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["n_gt"] == 1 and (rec["iou_gt"] or 0) >= 0.5:
            correct[rec["item_id"]] = rec

    table = pq.read_table(paths.DATA_ROOT / "prepared/groundingme/items.parquet")
    col = {n: table.column(n).to_pylist() for n in table.column_names}
    out = []
    for i in range(table.num_rows):
        item_id = col["item_id"][i]
        if item_id not in correct:
            continue
        out.append({
            "item_id": item_id,
            "expr": col["expr"][i],
            "dimension": col["dimension"][i],
            "size_bin": col["size_bin"][i],
            "image_path": col["image_path"][i],
            "gt_box": [float(v) for v in col["gt_boxes_xyxy_px"][i][0]],
            "model_box": correct[item_id]["box"],
            "iou_gt": correct[item_id]["iou_gt"],
        })
    out.sort(key=lambda r: r["item_id"])
    return out


def done_ids() -> set[str]:
    if not INDEX.is_file():
        return set()
    return {r["item_id"] for r in json.loads(INDEX.read_text(encoding="utf-8"))}


def poisoned() -> set[str]:
    if not ATTEMPTS.is_file():
        return set()
    counts = collections.Counter(
        x.strip() for x in ATTEMPTS.read_text(encoding="utf-8").splitlines() if x.strip())
    finished = done_ids()
    return {k for k, v in counts.items() if v >= MAX_ATTEMPTS and k not in finished}


def append_row(row: dict) -> None:
    rows = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.is_file() else []
    rows.append(row)
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    tmp.replace(INDEX)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from idea91 import masks as M
    from idea91.edits.build import materialize
    from idea91.edits.inpaint import get_inpainter
    from idea91.edits.sampler import EditPlan, hole_for
    from idea91.edits.window import window_box
    from idea91.instances.sam import Sam3Segmenter
    from shared import paths

    rows = wanted()
    if args.limit:
        rows = rows[: args.limit]
    total, done, skip = len(rows), done_ids(), poisoned()
    rows = [r for r in rows if r["item_id"] not in done and r["item_id"] not in skip]
    print(f"{total} ORIGINAL-correct items; {len(done)} built, {len(skip)} skipped, "
          f"{len(rows)} to do")
    print(f"edits root: {edits_root()}\n", flush=True)
    if not rows:
        return

    INDEX.parent.mkdir(parents=True, exist_ok=True)
    if not INDEX.is_file():
        INDEX.write_text("[]", encoding="utf-8")

    segmenter = Sam3Segmenter()
    inpainter = get_inpainter("big_lama")
    counts: collections.Counter = collections.Counter()
    started = time.perf_counter()

    for n, item in enumerate(rows, 1):
        with open(ATTEMPTS, "a", encoding="utf-8") as fh:
            fh.write(item["item_id"] + "\n")
        try:
            image = np.asarray(
                Image.open(paths.DATA_ROOT / item["image_path"]).convert("RGB"))
            h, w = image.shape[:2]
            box = item["gt_box"]

            # P8: prompt SAM 3 on the window. A 59 Mpx frame would otherwise be
            # resized to the segmenter's own input and leave the referent a
            # handful of pixels wide.
            window = window_box(box, (w, h))
            crop = np.ascontiguousarray(window.crop(image))
            local_box = (box[0] - window.x0, box[1] - window.y0,
                         box[2] - window.x0, box[3] - window.y0)
            local_mask = segmenter.from_box(crop, local_box)

            mask = np.zeros((h, w), dtype=bool)
            mask[window.y0:window.y1, window.x0:window.x1] = local_mask
            if not mask.any():
                counts["drop:empty_referent_mask"] += 1
                continue

            iou = M.box_to_mask_iou(box, mask) if hasattr(M, "box_to_mask_iou") else None
            if iou is None:
                mb = M.bbox_xyxy(mask)
                ix0, iy0 = max(box[0], mb[0]), max(box[1], mb[1])
                ix1, iy1 = min(box[2], mb[2]), min(box[3], mb[3])
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                union = ((box[2] - box[0]) * (box[3] - box[1])
                         + (mb[2] - mb[0]) * (mb[3] - mb[1]) - inter)
                iou = inter / union if union > 0 else 0.0

            hole_type = "mask" if iou >= BOX_TO_MASK_MIN else "rect"
            box_inpaint = hole_type == "rect"
            if box_inpaint:
                mask = M.mask_from_box(tuple(box), (h, w))
            plan = EditPlan(operator="REMOVE", hole=hole_for(mask, hole_type),
                            mask=mask, hole_type=hole_type)

            built = materialize(
                image, plan, inpainter,
                image_id=item["item_id"], instance_id="ref", set_or_pool=SET_NAME,
                edits_root=edits_root(), box_to_mask_iou=float(iou),
                box_inpaint_flag=box_inpaint, freeze_version="unfrozen",
            )
            append_row({**item, **built.index_row,
                        "mask_rle": built.index_row["mask_rle"],
                        "image_wh": [w, h]})
            counts[f"built:{hole_type}"] += 1
            counts[f"dim:{item['dimension']}"] += 1
        except Exception as exc:
            counts[f"error:{type(exc).__name__}"] += 1
            print(f"  {item['item_id']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        finally:
            image = None

        if n % 20 == 0:
            rate = (time.perf_counter() - started) / n
            print(f"  [{n}/{len(rows)}]  {rate:.1f}s/item, "
                  f"{(len(rows) - n) * rate / 60:.0f} min left  {dict(counts)}", flush=True)

    print(f"\n{len(done_ids())} edits in {INDEX}")
    for key, value in sorted(counts.items()):
        print(f"  {key:<34} {value}")


if __name__ == "__main__":
    main()
