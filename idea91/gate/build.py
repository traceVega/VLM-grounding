"""Turn the edit index into the sample sets the K1 gate measures (design P5, P7,
acceptance check 1).

Six kinds of sample set come out of the same bank:

* **gate contrast** REMOVE versus CONTROL_OBJ -- what P7's verdict is taken on.
* **comparison** REMOVE versus CONTROL_BG -- reported on the same rows.
* **1b null** CONTROL_OBJ versus CONTROL_OBJ_2, two edits from the same
  distribution with one labelled each way.  AUROC must land at 0.5 +/- 0.05, and
  if it does not, the gate's AUROC is not measuring the class difference.
* **1a global ladder** one of those two edits re-encoded whole at q75/q90/q92
  against the other at q95 -- can the classifier see a whole-image re-encode at
  all?
* **1a local ladder** the same pairs re-encoded inside the dilated hole only at
  q75/q50/q30 -- can it see damage at *editor* scale?
* **1c null** paired crops outside both holes, which are pixel-identical by
  construction; asserted rather than trained.

Every ladder runs on the 1b pairs, never on REMOVE, so no real class signal can
contaminate a sensitivity measurement.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from idea91.gate.dataset import GateSample
from idea91.gate.ladders import (
    GLOBAL_LADDER_QUALITIES,
    LOCAL_LADDER_QUALITIES,
    REFERENCE_QUALITY,
    hole_area_bin,
)

#: operator -> the pair member it plays in the 1b null and the ladders
FIRST_CONTROL = "CONTROL_OBJ"
SECOND_CONTROL = "CONTROL_OBJ_2"


def _prefix(hole_type: str) -> str:
    return "" if hole_type == "mask" else "RECT_"


def _op(name: str, hole_type: str) -> str:
    return f"{_prefix(hole_type)}{name}"


@dataclass
class EditRow:
    """One row of the edit index, with only what the gate needs."""

    image_id: str
    operator: str
    hole_type: str
    window_path: str
    window_xyxy: tuple[int, int, int, int]
    mask_area_frac: float
    control_source: str | None
    mask_rle: str


def rows_by_image(index: pa.Table) -> dict[str, dict[str, EditRow]]:
    """``image_id -> operator -> row``."""
    columns = {n: index.column(n).to_pylist() for n in index.column_names}
    out: dict[str, dict[str, EditRow]] = collections.defaultdict(dict)
    for i in range(index.num_rows):
        row = EditRow(
            image_id=columns["image_id"][i],
            operator=columns["operator"][i],
            hole_type=columns["hole_type"][i],
            window_path=columns["window_path"][i],
            window_xyxy=tuple(columns["window_xyxy_px"][i]),
            mask_area_frac=columns["mask_area_frac"][i],
            control_source=columns["control_source"][i],
            mask_rle=columns["mask_rle"][i],
        )
        out[row.image_id][row.operator] = row
    return out


def _sample(
    row: EditRow,
    label: int,
    image_dir: Path,
    *,
    suffix: str = "",
    damage: tuple[str, int] | None = None,
) -> GateSample:
    return GateSample(
        sample_id=f"{row.image_id}_{row.operator}{suffix}",
        image_id=row.image_id,
        label=label,
        original_path=image_dir / f"{row.image_id}.jpg",
        window_path=Path(row.window_path),
        window_xyxy=row.window_xyxy,
        control_source=row.control_source,
        hole_area_bin=hole_area_bin(float(row.mask_area_frac)),
        hole_rle=row.mask_rle,
        damage=damage,
    )


def contrast_samples(
    index: pa.Table,
    image_dir: Path,
    *,
    hole_type: str = "mask",
    positive: str = "REMOVE",
    negative: str = "CONTROL_OBJ",
) -> list[GateSample]:
    """The gate contrast: one positive and one negative edit per image.

    Only images carrying *both* operators contribute, so every image supplies
    exactly one sample of each class and the per-image AUROC is balanced.
    """
    pos_op, neg_op = _op(positive, hole_type), _op(negative, hole_type)
    out: list[GateSample] = []
    for _, ops in rows_by_image(index).items():
        if pos_op not in ops or neg_op not in ops:
            continue
        out.append(_sample(ops[pos_op], 1, image_dir))
        out.append(_sample(ops[neg_op], 0, image_dir))
    return out


def null_1b_samples(index: pa.Table, image_dir: Path, *, hole_type: str = "mask") -> list[GateSample]:
    """Check 1b: two independent CONTROL_OBJ edits, one labelled each way."""
    first, second = _op(FIRST_CONTROL, hole_type), _op(SECOND_CONTROL, hole_type)
    out: list[GateSample] = []
    for _, ops in rows_by_image(index).items():
        if first not in ops or second not in ops:
            continue
        out.append(_sample(ops[first], 1, image_dir))
        out.append(_sample(ops[second], 0, image_dir))
    return out


def ladder_samples(
    index: pa.Table,
    image_dir: Path,
    *,
    arm: str,
    quality: int,
    hole_type: str = "mask",
    reference_quality: int = REFERENCE_QUALITY,
) -> list[GateSample]:
    """Check 1a: the 1b pairs, one edit damaged and the other the reference.

    ``arm`` is ``global`` (whole-image re-encode) or ``local`` (in the dilated
    hole only).  The damaged edit is the positive class.
    """
    if arm not in ("global", "local"):
        raise ValueError(f"arm must be global or local, got {arm!r}")
    qualities = GLOBAL_LADDER_QUALITIES if arm == "global" else LOCAL_LADDER_QUALITIES
    if quality not in qualities:
        raise ValueError(f"{arm} ladder quality must be one of {qualities}, got {quality}")

    first, second = _op(FIRST_CONTROL, hole_type), _op(SECOND_CONTROL, hole_type)
    out: list[GateSample] = []
    for _, ops in rows_by_image(index).items():
        if first not in ops or second not in ops:
            continue
        out.append(_sample(ops[first], 1, image_dir, suffix=f"_q{quality}", damage=(arm, quality)))
        out.append(
            _sample(
                ops[second],
                0,
                image_dir,
                suffix=f"_ref{reference_quality}",
                damage=(arm, reference_quality),
            )
        )
    return out


def by_hole_area_bin(samples: list[GateSample]) -> dict[str, list[GateSample]]:
    """The local floor is reported per hole-area bin (check 1a)."""
    out: dict[str, list[GateSample]] = collections.defaultdict(list)
    for s in samples:
        out[s.hole_area_bin or "unknown"].append(s)
    return dict(out)


def assert_null_1c(index: pa.Table, image_dir: Path, edits_root: Path, *, hole_type: str = "mask",
                   limit: int = 200, seed: int = 0) -> dict[str, int]:
    """Check 1c: a crop outside both holes is pixel-identical between the two classes.

    This is an assert, not a trained row: because compositing is in-mask, the
    crops are the same bytes, so the AUROC is 0.5 by construction.  Returns the
    counts so the table can say how many pairs were checked.
    """
    import numpy as np

    from idea91.edits.build import hole_from_row, load_edited, read_image
    from idea91.edits.composite import assert_paired_crop_identical
    from idea91.gate.inputs import sample_paired_crop_box
    from idea91.masks import bbox_xyxy

    remove_op, control_op = _op("REMOVE", hole_type), _op(FIRST_CONTROL, hole_type)
    rng = np.random.default_rng(seed)
    checked = skipped = 0
    for _, ops in rows_by_image(index).items():
        if checked >= limit:
            break
        if remove_op not in ops or control_op not in ops:
            continue
        a, b = ops[remove_op], ops[control_op]
        original = read_image(image_dir / f"{a.image_id}.jpg")
        remove = load_edited(
            original, {"window_xyxy_px": list(a.window_xyxy), "window_path": a.window_path},
            edits_root,
        )
        control = load_edited(
            original, {"window_xyxy_px": list(b.window_xyxy), "window_path": b.window_path},
            edits_root,
        )
        # The *hole*, not the mask. `mask_rle` stores the undilated mask, while
        # what the editor actually replaced is that mask dilated by 5 px (P2), or
        # its bounding box for a rect hole. Taking the crop box off the mask let
        # a crop land inside the dilation band, where the two images legitimately
        # differ -- and check 1c then reported "in-mask compositing was violated"
        # for a compositing that was perfectly correct. Since check 1c gates
        # whether a K1 PASS counts as decisive, that false failure would have
        # turned a good verdict into "NOT DECISIVE".
        ha = bbox_xyxy(hole_from_row({"mask_rle": a.mask_rle, "hole_type": hole_type}))
        hb = bbox_xyxy(hole_from_row({"mask_rle": b.mask_rle, "hole_type": hole_type}))
        height, width = original.shape[:2]
        box = sample_paired_crop_box((width, height), [ha, hb], rng=rng)
        if box is None:
            skipped += 1
            continue
        assert_paired_crop_identical(remove, control, box, ha, hb)
        checked += 1
    return {"checked": checked, "skipped_no_free_crop": skipped}
