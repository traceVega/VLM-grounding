"""Read instances back out of the store (design 4.3, DEVIATIONS D-12).

The instance pass is expensive and the P3 sampler is not, which is the whole
reason instances are persisted: a rule change re-runs the sampler in minutes off
these rows, with the GPU untouched.

Rows arrive grouped by image and come back as the ``(referent, instances)`` pair
the sampler takes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from idea91 import masks as M
from idea91.edits.sampler import Instance


@dataclass
class StoredScene:
    """One image's instances, as read back from the store."""

    image_id: str
    set_or_pool: str
    referent: Instance
    instances: list[Instance]
    segmenter: str
    segmenter_revision: str
    kill_grade: bool
    head_noun: str | None = None
    n_head_noun_instances: int = 1
    box_to_mask_iou: float | None = None
    box_inpaint_flag: bool = False

    @property
    def hole_type(self) -> str:
        return "rect" if self.box_inpaint_flag else "mask"


def shard_paths(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("shard-*.parquet"))


def read_shards(directory: str | Path, *, validate: bool = True) -> pa.Table:
    """Every shard concatenated.  A shard left half-written by a kill is skipped."""
    tables = []
    for path in shard_paths(directory):
        try:
            tables.append(pq.read_table(path))
        except Exception as exc:
            print(f"  skipping unreadable shard {path.name}: {exc}")
    if not tables:
        raise FileNotFoundError(f"no readable instance shards in {directory}")
    table = pa.concat_tables(tables)
    if validate:
        from idea91 import schemas

        schemas.validate(table, "instances")
    return table


def scenes(table: pa.Table) -> Iterator[StoredScene]:
    """Group rows by image and rebuild the sampler's inputs.

    Masks are decoded lazily per image rather than all at once: 10,000 images of
    dense segmentation do not fit in memory as boolean arrays.
    """
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    order: dict[str, list[int]] = {}
    for i, image_id in enumerate(columns["image_id"]):
        order.setdefault(image_id, []).append(i)

    for image_id, rows in order.items():
        instances: list[Instance] = []
        referent: Instance | None = None
        for i in rows:
            inst = Instance(
                instance_id=columns["instance_id"][i],
                mask=M.decode_rle(columns["mask_rle"][i]),
                source=columns["source"][i],
                label=columns["label"][i],
            )
            instances.append(inst)
            if inst.source == "referent":
                referent = inst
        if referent is None:
            continue  # a scene without a referent is not usable
        first = rows[0]
        ref_row = next((i for i in rows if columns["source"][i] == "referent"), first)
        yield StoredScene(
            image_id=image_id,
            set_or_pool=columns["set_or_pool"][first],
            referent=referent,
            instances=instances,
            segmenter=columns["segmenter"][first],
            segmenter_revision=columns["segmenter_revision"][first],
            kill_grade=bool(columns["kill_grade"][first]),
            head_noun=columns["head_noun"][first],
            n_head_noun_instances=int(columns["n_head_noun_instances"][first]),
            box_to_mask_iou=columns["box_to_mask_iou"][ref_row],
            box_inpaint_flag=bool(columns["box_inpaint_flag"][ref_row]),
        )


def load_scenes(directory: str | Path, limit: int | None = None) -> list[StoredScene]:
    out = []
    for scene in scenes(read_shards(directory)):
        out.append(scene)
        if limit and len(out) >= limit:
            break
    return out
