"""Build ``items.parquet`` (SPEC Section 3).

Each benchmark reaches this module through one small adapter that turns its own
annotation format into :class:`RawItem`.  The adapters are the seam that has to
wait for the data: nothing here guesses a JSON key it has not seen.  Everything
downstream of :class:`RawItem` -- hashing, perceptual hashing, size bins, dev-slice
assignment and the disjointness assert -- is written and tested.

    python -m shared.data.items --set groundingme --check   # validate an existing file
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa

from idea91.masks import size_bin_of
from shared import paths
from shared.harness import schema as S

Box = tuple[float, float, float, float]


@dataclass
class RawItem:
    """One benchmark item, normalised.  Adapters produce these; nothing else."""

    item_id: str
    image_path: Path  # absolute, or relative to the data root
    expr: str
    gt_boxes_xyxy_px: list[Box] = field(default_factory=list)
    dimension: str | None = None
    split_half: str | None = None
    pair_id: str | None = None
    in_kill_set: bool = True
    dev_slice: str | None = None
    ceiling_iou_d4: float | None = None
    downscale_factor_d4: float | None = None


def image_facts(path: Path) -> tuple[str, str, tuple[int, int]]:
    """``(sha256, 64-bit pHash hex, (width, height))`` for one image file."""
    import imagehash
    from PIL import Image

    sha = S.sha256_file(path)
    with Image.open(path) as im:
        im = im.convert("RGB")
        wh = im.size
        phash = format(int(str(imagehash.phash(im)), 16), "016x")
    return sha, phash, wh


def to_row(item: RawItem, set_name: str, data_root: Path | None = None) -> dict:
    """One ``items.parquet`` row, with the derived columns filled."""
    data_root = data_root or paths.DATA_ROOT
    path = Path(item.image_path)
    absolute = path if path.is_absolute() else data_root / path
    sha, phash, _ = image_facts(absolute)
    try:
        relative = absolute.relative_to(data_root).as_posix()
    except ValueError:
        relative = absolute.as_posix()
    boxes = [list(map(float, b)) for b in item.gt_boxes_xyxy_px]
    return {
        "item_id": item.item_id,
        "set": set_name,
        "split_half": item.split_half,
        "image_path": relative,
        "image_sha256": sha,
        "image_phash": phash,
        "expr": item.expr,
        "gt_boxes_xyxy_px": boxes,
        "n_gt": len(boxes),
        "size_bin": size_bin_of(tuple(boxes[0])) if boxes else None,
        "dimension": item.dimension,
        "in_kill_set": item.in_kill_set,
        "dev_slice": item.dev_slice,
        "pair_id": item.pair_id,
        "ceiling_iou_d4": item.ceiling_iou_d4,
        "downscale_factor_d4": item.downscale_factor_d4,
    }


def build(
    items: list[RawItem],
    set_name: str,
    out_path: Path | None = None,
    *,
    data_root: Path | None = None,
    check_disjoint: bool = True,
) -> pa.Table:
    """Rows -> a validated table, written if ``out_path`` is given.

    SPEC Section 3's dev-slice disjointness is asserted here, at build time,
    because it is the only moment the whole set is in one place.
    """
    rows = [to_row(i, set_name, data_root) for i in items]
    table = pa.Table.from_pylist(rows, schema=S.ITEMS_SCHEMA)
    S.validate(table, "items")
    if check_disjoint:
        S.assert_dev_slice_disjoint(table)
    if out_path:
        S.write_table(table, out_path, "items")
    return table


def assign_dev_slice(
    items: list[RawItem], slice_name: str, n: int, *, seed: int = 0
) -> list[RawItem]:
    """Reserve ``n`` items for a dev slice and take them out of the kill set."""
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    picked = {int(i) for i in order[:n]}
    out = []
    for i, item in enumerate(items):
        if i in picked:
            out.append(
                RawItem(
                    **{
                        **item.__dict__,
                        "dev_slice": slice_name,
                        "in_kill_set": False,
                    }
                )
            )
        else:
            out.append(item)
    return out


# --- adapters ---------------------------------------------------------------
#
# One per set. Each waits for the real annotation format; none of them guesses.


def _pending(set_name: str, what: str):
    raise NotImplementedError(
        f"the {set_name} adapter is not written: {what}. Download the set "
        "(python -m shared.data.download), read its annotation format, then map it onto "
        "RawItem here. Everything downstream of RawItem is already tested."
    )


def groundingme_items(root: Path) -> list[RawItem]:  # pragma: no cover - needs the data
    """P8: positive single-box items, with the GroundingME dimension recorded."""
    _pending(
        "groundingme",
        "needs the annotation file layout at the pinned revision, the dimension field name, "
        "and which items are positive single-box",
    )


def openref_items(root: Path) -> list[RawItem]:  # pragma: no cover
    """P8: single-target positives.  Design O1: format and licence unread."""
    _pending("openref", "release format unread (design O1)")


def grefcoco_p12_items(root: Path) -> list[RawItem]:  # pragma: no cover
    """P12's dev slice: 100 no-target items and 100 positives, dev_slice='p12'."""
    _pending("grefcoco", "needs the no-target annotation format and the COCO image mapping")


def cocosearch18_items(root: Path) -> list[RawItem]:  # pragma: no cover
    """P18: all 18 categories, target-present and target-absent."""
    _pending(
        "cocosearch18",
        "needs the present/absent lists and the 1,680 x 1,050 letterbox transform, whose "
        "inverse P18 checks visually on 20 images",
    )


def openimages_pool(root: Path) -> list[RawItem]:
    """P1's pool.  Written: see ``shared/data/openimages.py``.

    ``root`` is ignored; the pool is selected from the metadata CSVs and its
    images are read from the download directory.
    """
    import pandas as pd

    from shared.data import openimages

    pool_path = paths.PREPARED / "openimages_pool" / "pool.parquet"
    pool = (
        pd.read_parquet(pool_path)
        if pool_path.is_file()
        else openimages.select_pool()
    )
    return openimages.to_raw_items(pool)


ADAPTERS = {
    "groundingme": groundingme_items,
    "openref": openref_items,
    "grefcoco": grefcoco_p12_items,
    "cocosearch18": cocosearch18_items,
    "openimages_pool": openimages_pool,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_name", required=True, choices=sorted(S.SETS))
    ap.add_argument("--check", action="store_true", help="validate the existing items.parquet")
    args = ap.parse_args()
    path = paths.items_path(args.set_name)
    if args.check:
        table = S.read_table(path)
        S.assert_dev_slice_disjoint(table)
        print(f"{path}: {table.num_rows} items, valid against SPEC v{S.SPEC_VERSION}")
        return
    adapter = ADAPTERS.get(args.set_name)
    if adapter is None:
        raise SystemExit(f"no adapter for {args.set_name}")
    build(adapter(paths.RAW / args.set_name), args.set_name, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
