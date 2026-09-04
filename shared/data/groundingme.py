"""GroundingME adapter: the pinned snapshot to :class:`RawItem` (design P8, B6).

    python -m shared.data.groundingme --extract      # images to disk, once
    python -m shared.data.groundingme --items        # prepared/groundingme/items.parquet

The dataset ships one parquet row per item with the image inlined as bytes, so
"downloading" and "extracting" are separate steps: the HF snapshot is already on
disk at a pinned revision, and this writes the 1,005 images out where the edit
stack and the harness can open them by path.

What the fields mean, read off the snapshot at revision
``78e3c7974b2b1db0ea52266969e40f664a38e330`` on 2026-09-03 rather than assumed
(OPEN-QUESTIONS Q-1):

``bbox``
    Absolute ``xyxy`` in original-image pixels, or **null** for the ``Rejection``
    subtask.  100% of the 804 positives are consistent with ``xyxy`` and 15%
    with ``xywh``, which is chance; the evaluator reads them as ``[x1, y1, x2,
    y2]``.
``description``
    The referring expression, but paragraph-shaped: median 221 characters, and
    68% open with "The object is ...".  This is what P13's T_NULL replaces.
``detection_type``
    A head noun, present for every item.  P10 asks its verifier question about
    the head noun, so on this set that needs no noun-phrase parse.
``subtask_l1``
    ``Limited`` (300), ``Spatial`` (300), ``Discriminative`` (204) and
    ``Rejection`` (201).  This is the "GroundingME dimension" P15 reports per.
    ``subtask_l2`` is the finer split and is kept in a sidecar rather than
    squeezed into the single ``dimension`` column.

The ``Rejection`` items carry no box and so are not K2 pairs -- P8 scopes K2 to
"positive single-box items".  They are still emitted, with ``in_kill_set``
false, because throwing away 201 labelled negatives to keep a loop tidy would be
worse than carrying them: they are the natural companion to P12's rejection
protocol.  Nothing includes them in a K2 statistic unless it asks for them.
"""

from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from shared import paths
from shared.data.items import RawItem

SET_NAME = "groundingme"
HF_REPO = "lirang04/GroundingME"
REVISION = "78e3c7974b2b1db0ea52266969e40f664a38e330"

#: P8: K2 is anchored on positive single-box items.
POSITIVE_SUBTASKS = ("Limited", "Spatial", "Discriminative")
REJECTION_SUBTASK = "Rejection"

#: P8's referent-area limit.  Measured at 100% pass on this set, so it excludes
#: nothing here -- kept because P8 states it and a later snapshot could differ.
MAX_AREA_FRAC = 0.30


def snapshot_dir(revision: str = REVISION) -> Path:
    """The pinned snapshot in the HF cache, or a clear failure."""
    pattern = (
        Path.home()
        / ".cache/huggingface/hub"
        / f"datasets--{HF_REPO.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not pattern.is_dir():
        raise SystemExit(
            f"GroundingME snapshot {revision} is not in the HF cache at {pattern}.\n"
            f"Fetch it with:  hf download {HF_REPO} --repo-type dataset --revision {revision}"
        )
    return pattern


def shard_paths(revision: str = REVISION) -> list[Path]:
    paths_ = sorted(Path(p) for p in glob.glob(str(snapshot_dir(revision) / "data" / "*.parquet")))
    if not paths_:
        raise SystemExit(f"no parquet shards under {snapshot_dir(revision) / 'data'}")
    return paths_


def image_dir() -> Path:
    return paths.RAW / SET_NAME / "images"


def image_path_for(item_id: str) -> Path:
    return image_dir() / f"{item_id}.jpg"


def item_id_for(row_id: int) -> str:
    return f"gme_{int(row_id):05d}"


@dataclass
class GmeRow:
    """One dataset row, with only what the adapter uses."""

    item_id: str
    description: str
    bbox: list[int] | None
    detection_type: str
    subtask_l1: str
    subtask_l2: str
    width: int
    height: int

    @property
    def is_positive(self) -> bool:
        return self.bbox is not None and len(self.bbox) == 4

    @property
    def area_frac(self) -> float:
        if not self.is_positive:
            return 0.0
        x0, y0, x1, y1 = (float(v) for v in self.bbox)
        return abs((x1 - x0) * (y1 - y0)) / float(self.width * self.height)


def read_rows(revision: str = REVISION) -> list[GmeRow]:
    """Every row's metadata, without decoding a single image."""
    columns = [
        "id", "description", "bbox", "detection_type", "subtask_l1", "subtask_l2",
        "width", "height",
    ]
    out: list[GmeRow] = []
    for shard in shard_paths(revision):
        table = pq.read_table(shard, columns=columns)
        for row in table.to_pylist():
            out.append(
                GmeRow(
                    item_id=item_id_for(row["id"]),
                    description=row["description"],
                    bbox=list(row["bbox"]) if row["bbox"] is not None else None,
                    detection_type=row["detection_type"],
                    subtask_l1=row["subtask_l1"],
                    subtask_l2=row["subtask_l2"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
    return out


def extract_images(revision: str = REVISION, *, overwrite: bool = False) -> int:
    """Write the inlined image bytes out as files, one per item.

    Resumable and idempotent: an item whose file already exists is skipped, so
    an interrupted extraction costs nothing.
    """
    directory = image_dir()
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for shard in shard_paths(revision):
        table = pq.read_table(shard, columns=["id", "image"])
        ids = table.column("id").to_pylist()
        images = table.column("image").to_pylist()
        for row_id, image in zip(ids, images):
            path = image_path_for(item_id_for(row_id))
            if path.exists() and not overwrite:
                continue
            path.write_bytes(image["bytes"])
            written += 1
        del table  # one shard at a time: the images are the bulk of 2.9 GB
    return written


def to_raw_items(rows: list[GmeRow] | None = None) -> list[RawItem]:
    """P8's K2 items, plus the rejection items marked out of the kill set."""
    rows = read_rows() if rows is None else rows
    items: list[RawItem] = []
    for row in rows:
        path = image_path_for(row.item_id)
        if row.is_positive:
            # P8 also requires a SAM 3 mask at box-to-mask IoU >= 0.5. That needs
            # the GPU, so it is the instances stage's filter, not this one; the
            # area limit is checkable here and is applied here.
            in_kill_set = row.area_frac <= MAX_AREA_FRAC
            boxes = [tuple(float(v) for v in row.bbox)]
        else:
            in_kill_set = False  # P8 scopes K2 to positive single-box items
            boxes = []
        items.append(
            RawItem(
                item_id=row.item_id,
                image_path=path,
                expr=row.description,
                gt_boxes_xyxy_px=boxes,
                dimension=row.subtask_l1,
                in_kill_set=in_kill_set,
            )
        )
    return items


def write_dimension_sidecar(rows: list[GmeRow], out: Path | None = None) -> Path:
    """``item_id -> (subtask_l1, subtask_l2, detection_type)``.

    ``items.parquet`` has one ``dimension`` column and P15 reports on
    ``subtask_l1``, so the finer split and the head noun would otherwise be lost
    between here and the K2 tables.
    """
    out = out or (paths.PREPARED / SET_NAME / "dimensions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item_id", "subtask_l1", "subtask_l2", "head_noun"])
        for row in rows:
            writer.writerow([row.item_id, row.subtask_l1, row.subtask_l2, row.detection_type])
    return out


def summary(rows: list[GmeRow]) -> str:
    positives = [r for r in rows if r.is_positive]
    rejections = [r for r in rows if not r.is_positive]
    over_area = [r for r in positives if r.area_frac > MAX_AREA_FRAC]
    lines = [
        f"GroundingME @ {REVISION[:12]}: {len(rows)} items",
        f"  positive single-box: {len(positives)}   rejection (null bbox): {len(rejections)}",
        f"  positives over P8's {MAX_AREA_FRAC:.0%} area limit: {len(over_area)}",
        "",
        "  subtask_l1:",
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.subtask_l1] = counts.get(row.subtask_l1, 0) + 1
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {name:16s} {count}")
    lines += [
        "",
        "  P8's 2,000-pair target needs OpenRef: this set tops out at "
        f"{len(positives)} before the SAM 3 mask filter (contingency O1).",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", action="store_true", help="write the inlined images to disk")
    ap.add_argument("--items", action="store_true", help="build prepared/groundingme/items.parquet")
    ap.add_argument("--overwrite", action="store_true", help="re-extract images that exist")
    args = ap.parse_args()

    rows = read_rows()
    print(summary(rows))

    if args.extract:
        written = extract_images(overwrite=args.overwrite)
        print(f"\nextracted {written} images -> {image_dir()}")

    if args.items:
        missing = [r.item_id for r in rows if not image_path_for(r.item_id).is_file()]
        if missing:
            raise SystemExit(
                f"{len(missing)} images are not on disk (first: {missing[0]}); "
                "run --extract first"
            )
        from shared.data.items import build

        items = to_raw_items(rows)
        out = paths.items_path(SET_NAME)
        build(items, SET_NAME, out)
        sidecar = write_dimension_sidecar(rows)
        kill = sum(1 for i in items if i.in_kill_set)
        print(f"\n{len(items)} items ({kill} in the kill set) -> {out}")
        print(f"dimensions -> {sidecar}")


if __name__ == "__main__":
    main()
