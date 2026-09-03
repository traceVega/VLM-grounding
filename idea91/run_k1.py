"""K1 pipeline driver (design B3 to B5).

Stages, each resumable and each writing its own store:

    python -m idea91.run_k1 instances   # SAM 3 concept + SAM 2 automatic -> instances.parquet
    python -m idea91.run_k1 edits       # P3/P4 sampler + LaMa            -> edits/index.parquet
    python -m idea91.run_k1 status      # what exists so far

The instance pass is the expensive one (hours), and it is deliberately separate
from the sampler: masks are a property of the image, control choices are a
property of P3, and P3 is still being settled.  Persisting instances means the
sampler can be re-run in minutes when a rule changes, without touching the GPU.

Shards are written every ``--shard`` images so a crash costs one shard, and a
re-run skips images already present.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from idea91 import schemas
from idea91.edits.build import materialize, read_image
from idea91.edits.sampler import HOLE_TYPES, SamplerStats, plan_k1_image
from idea91.instances.build import build_k1_scene, scene_rows
from idea91.instances.store import load_scenes
from shared import paths

SET_NAME = "openimages_pool"

#: P7 requires the gate verdict "on both hole types", so the bank carries both.
DEFAULT_HOLE_TYPES = ("mask", "rect")


def instances_dir() -> Path:
    return paths.PREPARED / SET_NAME / "instances"


def done_image_ids() -> set[str]:
    """Image ids already written to a shard, so a re-run resumes."""
    directory = instances_dir()
    if not directory.is_dir():
        return set()
    done: set[str] = set()
    for shard in sorted(directory.glob("shard-*.parquet")):
        try:
            done |= set(pq.read_table(shard, columns=["image_id"]).column("image_id").to_pylist())
        except Exception as exc:  # a half-written shard from a kill
            print(f"  ignoring unreadable shard {shard.name}: {exc}")
    return done


def write_shard(rows: list[dict], index: int) -> Path:
    directory = instances_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shard-{index:04d}.parquet"
    table = pa.Table.from_pylist(rows, schema=schemas.INSTANCES_SCHEMA)
    schemas.validate(table, "instances")
    pq.write_table(table, path, compression="zstd")
    return path


def stage_instances(limit: int | None, shard_size: int, seed: int, device: str) -> None:
    from idea91.instances.sam import CompositeSegmenter

    pool_path = paths.PREPARED / SET_NAME / "pool.parquet"
    pool = pd.read_parquet(pool_path)
    if limit:
        pool = pool.head(limit)
    image_dir = paths.RAW / "openimages" / "images"

    already = done_image_ids()
    todo = [r for r in pool.itertuples() if r.image_id not in already]
    print(f"pool {len(pool)} images; {len(already)} already done; {len(todo)} to go")
    if not todo:
        return

    segmenter = CompositeSegmenter(device=device)
    shard_index = len(list(instances_dir().glob("shard-*.parquet"))) if instances_dir().is_dir() else 0
    buffer: list[dict] = []
    drops: collections.Counter = collections.Counter()
    sources: collections.Counter = collections.Counter()
    started = time.perf_counter()

    for n, row in enumerate(todo, 1):
        path = image_dir / f"{row.image_id}.jpg"
        try:
            image = read_image(path)
            scene = build_k1_scene(image, row.image_id, list(row.labels), segmenter, seed=seed)
            if scene.drop_reason:
                drops[scene.drop_reason] += 1
                continue
            rows = scene_rows(scene, SET_NAME)
        except Exception as exc:  # a corrupt jpeg, an OOM, an odd mask
            drops[f"error:{type(exc).__name__}"] += 1
            print(f"  {row.image_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        buffer.extend(rows)
        for r in rows:
            sources[r["source"]] += 1

        if len(buffer) >= shard_size:
            written = write_shard(buffer, shard_index)
            elapsed = time.perf_counter() - started
            rate = n / elapsed
            print(
                f"  [{n}/{len(todo)}] wrote {written.name} ({len(buffer)} rows) "
                f"{rate:.2f} img/s, eta {(len(todo)-n)/max(rate,1e-6)/3600:.1f} h",
                flush=True,
            )
            buffer.clear()
            shard_index += 1

    if buffer:
        print(f"  wrote {write_shard(buffer, shard_index).name} ({len(buffer)} rows)")

    elapsed = time.perf_counter() - started
    print(f"\ninstances done in {elapsed/3600:.2f} h ({len(todo)/max(elapsed,1e-9):.2f} img/s)")
    print(f"dropped: {dict(drops)}")
    print(f"instance sources: {dict(sources)}")


def edits_dir() -> Path:
    return paths.EDITS_ROOT


def index_dir() -> Path:
    return paths.EDITS_ROOT / "index"


def done_edit_keys() -> set[tuple[str, str]]:
    """``(image_id, operator)`` pairs already indexed, so a re-run resumes."""
    directory = index_dir()
    if not directory.is_dir():
        return set()
    done: set[tuple[str, str]] = set()
    for shard in sorted(directory.glob("shard-*.parquet")):
        try:
            table = pq.read_table(shard, columns=["image_id", "operator"])
            done |= set(
                zip(
                    table.column("image_id").to_pylist(),
                    table.column("operator").to_pylist(),
                )
            )
        except Exception as exc:
            print(f"  ignoring unreadable index shard {shard.name}: {exc}")
    return done


def write_index_shard(rows: list[dict], index: int) -> Path:
    directory = index_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shard-{index:04d}.parquet"
    table = pa.Table.from_pylist(rows, schema=schemas.INDEX_SCHEMA)
    schemas.validate(table, "index")
    pq.write_table(table, path, compression="zstd")
    return path


def stage_edits(
    limit: int | None,
    shard_size: int,
    seed: int,
    device: str,
    hole_types: tuple[str, ...],
    freeze_version: str,
) -> None:
    """P3/P4 sampling plus the LaMa edits, off the persisted instances."""
    from idea91.edits.inpaint import LamaInpainter

    scenes = load_scenes(instances_dir(), limit=limit)
    print(f"{len(scenes)} scenes from the instance store; hole types {list(hole_types)}")
    already = done_edit_keys()
    if already:
        print(f"  {len(already)} edits already indexed; resuming")

    inpainter = LamaInpainter(device=device)
    image_dir = paths.RAW / "openimages" / "images"
    stats = SamplerStats()
    drops: collections.Counter = collections.Counter()
    made: collections.Counter = collections.Counter()
    sources: collections.Counter = collections.Counter()
    buffer: list[dict] = []
    shard_index = len(list(index_dir().glob("shard-*.parquet"))) if index_dir().is_dir() else 0
    started = time.perf_counter()

    for n, scene in enumerate(scenes, 1):
        try:
            plans = []
            for hole_type in hole_types:
                planned = plan_k1_image(
                    scene.referent,
                    scene.instances,
                    hole_type=hole_type,
                    rng=np.random.default_rng(seed),
                    stats=stats,
                )
                if planned is None:
                    drops[f"no_valid_control_obj:{hole_type}"] += 1
                    continue
                plans += planned
            if not plans:
                continue

            todo = [p for p in plans if (scene.image_id, p.operator) not in already]
            if not todo:
                continue
            image = read_image(image_dir / f"{scene.image_id}.jpg")
            for plan in todo:
                built = materialize(
                    image,
                    plan,
                    inpainter,
                    image_id=scene.image_id,
                    instance_id=scene.referent.instance_id,
                    set_or_pool=SET_NAME,
                    edits_root=edits_dir(),
                    box_to_mask_iou=scene.box_to_mask_iou,
                    box_inpaint_flag=scene.box_inpaint_flag,
                    freeze_version=freeze_version,
                )
                buffer.append(built.index_row)
                made[plan.operator] += 1
                if plan.control_source:
                    sources[plan.control_source] += 1
        except Exception as exc:
            drops[f"error:{type(exc).__name__}"] += 1
            print(f"  {scene.image_id}: {type(exc).__name__}: {exc}", flush=True)
            continue

        if len(buffer) >= shard_size:
            written = write_index_shard(buffer, shard_index)
            elapsed = time.perf_counter() - started
            rate = n / elapsed
            print(
                f"  [{n}/{len(scenes)}] wrote {written.name} ({len(buffer)} edits) "
                f"{rate:.2f} img/s, eta {(len(scenes) - n) / max(rate, 1e-6) / 3600:.1f} h",
                flush=True,
            )
            buffer.clear()
            shard_index += 1

    if buffer:
        print(f"  wrote {write_index_shard(buffer, shard_index).name} ({len(buffer)} edits)")
    elapsed = time.perf_counter() - started
    print(f"\nedits done in {elapsed / 3600:.2f} h")
    print(f"operators: {dict(made)}")
    print(f"control_source mix: {dict(sources)}")
    print(f"dropped: {dict(drops)}")
    print(f"sampler rejections: {stats.as_dict()}")


def count_table() -> str:
    """The count table the design commits at each freeze (4.2).

    Per set, hole type, control type and control source.
    """
    shards = sorted(index_dir().glob("shard-*.parquet"))
    if not shards:
        return "no edit index yet"
    table = pa.concat_tables([pq.read_table(s) for s in shards])
    df = table.select(
        ["set_or_pool", "operator", "hole_type", "control_source", "mask_area_frac", "image_id"]
    ).to_pandas()
    lines = [
        "# K1 edit-bank counts",
        "",
        f"images {df.image_id.nunique():,}   edits {len(df):,}",
        "",
        "| set | hole type | operator | control source | edits | median hole area |",
        "|---|---|---|---|---|---|",
    ]
    df = df.assign(control_source=df.control_source.fillna("-"))
    for (s, h, op, src), g in df.groupby(
        ["set_or_pool", "hole_type", "operator", "control_source"], observed=True
    ):
        lines.append(
            f"| {s} | {h} | {op} | {src} | {len(g):,} | {g.mask_area_frac.median() * 100:.2f}% |"
        )
    return "\n".join(lines)


def stage_status() -> None:
    directory = instances_dir()
    shards = sorted(directory.glob("shard-*.parquet")) if directory.is_dir() else []
    print(f"instances: {len(shards)} shards in {directory}")
    if not shards:
        return
    table = pa.concat_tables([pq.read_table(s) for s in shards])
    images = table.column("image_id").unique()
    df = table.select(["source", "kill_grade", "area_frac"]).to_pandas()
    print(f"  images {len(images):,}   instance rows {table.num_rows:,}")
    print(f"  rows per image: {table.num_rows/max(1,len(images)):.1f}")
    print(f"  sources: {df.source.value_counts().to_dict()}")
    print(f"  kill_grade: {df.kill_grade.value_counts().to_dict()}")
    print(f"  area_frac: median {df.area_frac.median():.4f}  max {df.area_frac.max():.4f}")
    shards = sorted(index_dir().glob("shard-*.parquet"))
    print(f"\nedits: {len(shards)} index shards in {index_dir()}")
    if shards:
        print(count_table())


def load_instances() -> pa.Table:
    """Every shard, concatenated and validated (the sampler's input)."""
    shards = sorted(instances_dir().glob("shard-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no instance shards in {instances_dir()}; run the instances stage")
    table = pa.concat_tables([pq.read_table(s) for s in shards])
    return schemas.validate(table, "instances")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["instances", "edits", "status"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N pool images")
    ap.add_argument("--shard", type=int, default=5_000, help="rows per shard")
    ap.add_argument("--seed", type=int, default=0, help="P1's referent choice seed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hole-types", default=",".join(DEFAULT_HOLE_TYPES))
    ap.add_argument("--freeze-version", default="unfrozen")
    args = ap.parse_args()

    if args.stage == "instances":
        stage_instances(args.limit, args.shard, args.seed, args.device)
    elif args.stage == "edits":
        holes = tuple(h.strip() for h in args.hole_types.split(",") if h.strip())
        bad = [h for h in holes if h not in HOLE_TYPES]
        if bad:
            raise SystemExit(f"unknown hole types {bad}; expected {list(HOLE_TYPES)}")
        stage_edits(args.limit, args.shard, args.seed, args.device, holes, args.freeze_version)
    else:
        stage_status()


if __name__ == "__main__":
    main()
