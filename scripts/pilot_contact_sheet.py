"""Render the same-box cases side by side, so a person can judge them in minutes.

    python scripts/pilot_contact_sheet.py

The claim lives on the items where the model returned essentially the same box
after the referent was removed. Those are the ones worth looking at with human
eyes, and there are about a hundred of them rather than several hundred, so a
contact sheet is a ten-minute job instead of an hour.

Each row is one item: the original with the model's box, and the removal with
both boxes -- the model's new box in the same colour, and where its original box
was, dashed, so a drift is visible without measuring it. The verifier's own
answer is printed alongside, so a reader can see where model and verifier
disagree, which is exactly what P11's human check is for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PANEL = 420
ORIGINAL_COLOUR = (60, 200, 60)   # BGR: the model's box on the original
REMOVED_COLOUR = (60, 60, 235)    # its box after removal
GHOST_COLOUR = (150, 150, 150)    # where the original box was
HOLE_COLOUR = (0, 200, 255)       # what the editor actually removed


def edit_rows() -> dict[str, dict]:
    from shared import paths

    shards = sorted((paths.EDITS_ROOT / "index").glob("shard-*.parquet"))
    table = pa.concat_tables([pq.read_table(s) for s in shards])
    cols = {n: table.column(n).to_pylist() for n in table.column_names}
    return {
        cols["image_id"][i]: {
            "window_xyxy_px": cols["window_xyxy_px"][i],
            "window_path": cols["window_path"][i],
            "mask_rle": cols["mask_rle"][i],
        }
        for i in range(table.num_rows)
        if cols["operator"][i] == "REMOVE"
    }


def fit(image: np.ndarray, side: int = PANEL) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = side / max(h, w)
    out = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.full((side, side, 3), 245, np.uint8)
    canvas[: out.shape[0], : out.shape[1]] = out
    return canvas, scale


def draw(canvas: np.ndarray, box, scale: float, colour, dashed: bool = False) -> None:
    if not box:
        return
    x0, y0, x1, y1 = (int(v * scale) for v in box)
    if not dashed:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
        return
    for x in range(x0, x1, 12):
        cv2.line(canvas, (x, y0), (min(x + 6, x1), y0), colour, 1)
        cv2.line(canvas, (x, y1), (min(x + 6, x1), y1), colour, 1)
    for y in range(y0, y1, 12):
        cv2.line(canvas, (x0, y), (x0, min(y + 6, y1)), colour, 1)
        cv2.line(canvas, (x1, y), (x1, min(y + 6, y1)), colour, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verified", type=Path, default=Path("tables/pilot_verified.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("tables/contact_sheets"))
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--category", choices=["same_box", "rejected"], default="same_box")
    args = ap.parse_args()

    from idea91.edits.build import load_edited, read_image
    from shared import paths

    records = json.loads(args.verified.read_text())["records"]
    if args.category == "same_box":
        cases = [r for r in records
                 if r.get("removal_clean") and (r["same_box"] or 0) >= 0.5]
        cases.sort(key=lambda r: -(r["same_box"] or 0))
        print(f"{len(cases)} verified-clean same-box cases")
    else:
        # The other half of the audit: the verifier dropped 42% of removals as
        # "the object is still there". Whether that is the editor failing or the
        # verifier being wrong changes the headline, and only eyes can say.
        cases = [r for r in records if not r.get("removal_clean")]
        cases.sort(key=lambda r: r["image_id"])
        print(f"{len(cases)} removals the verifier rejected")

    edits = edit_rows()
    image_dir = paths.RAW / "openimages" / "images"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, sheet_no = [], 0
    for rec in cases:
        edit = edits.get(rec["image_id"])
        if edit is None:
            continue
        try:
            original = read_image(image_dir / f"{rec['image_id']}.jpg")
            removed = load_edited(original, edit, paths.EDITS_ROOT)
            from idea91.masks import bbox_xyxy, decode_rle

            rec["hole_box"] = list(bbox_xyxy(decode_rle(edit["mask_rle"])))
        except Exception:
            continue

        left, scale = fit(cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        right, _ = fit(cv2.cvtColor(removed, cv2.COLOR_RGB2BGR))
        # What was actually taken out.  Without it a reader is judging the
        # removal through the model's box, which is a different rectangle: one
        # is what the editor did, the other is what the model thinks it sees.
        draw(left, rec.get("hole_box"), scale, HOLE_COLOUR)
        draw(right, rec.get("hole_box"), scale, HOLE_COLOUR)
        draw(left, rec["original_box"], scale, ORIGINAL_COLOUR)
        draw(right, rec["original_box"], scale, GHOST_COLOUR, dashed=True)
        draw(right, rec["removed_box"], scale, REMOVED_COLOUR)

        # same_box is None when the model declined on the removed image -- there
        # is no box to compare, which is a result rather than a missing value.
        iou = (f"IoU {rec['same_box']:.2f}" if rec["same_box"] is not None
               else f"model answered '{rec['removed_type']}' after removal")
        strip = np.full((58, PANEL * 2, 3), 255, np.uint8)
        cv2.putText(strip, f"{rec['expr']}   {iou}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(strip, f"verifier asked if it is still there -> "
                           f"\"{rec.get('verifier_answer', '?')}\"   {rec['image_id']}", (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
        rows.append(np.vstack([np.hstack([left, right]), strip]))

        if len(rows) == args.per_sheet:
            sheet_no += 1
            _write(rows, args.out_dir / f"{args.category}_{sheet_no:02d}.jpg")
            rows = []
    if rows:
        sheet_no += 1
        _write(rows, args.out_dir / f"{args.category}_{sheet_no:02d}.jpg")

    print("\n  amber   = what the editor actually removed (the hole)")
    print("  green   = the model's box on the ORIGINAL")
    print("  dashed  = that same box copied onto the removed image, for reference")
    print("  red     = the model's box AFTER removal (absent if it declined)")
    print(f"-> {sheet_no} sheet(s) in {args.out_dir}")


def _write(rows: list[np.ndarray], path: Path) -> None:
    header = np.full((34, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(header, "verified-clean removals where the box did NOT move", (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([header, *rows]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  {path.name}  ({len(rows)} cases)")


if __name__ == "__main__":
    main()
