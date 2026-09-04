"""K1 pipeline driver (design B3 to B5).

Stages, each resumable and each writing its own store:

    python -m idea91.run_k1 instances   # SAM 3 concept + SAM 2 automatic -> instances.parquet
    python -m idea91.run_k1 edits       # P3/P4 sampler + LaMa            -> edits/index.parquet
    python -m idea91.run_k1 checks      # B4a: ladders and nulls, pre-freeze
    python -m idea91.run_k1 freeze      # B0a: bind P1 to P7 (needs a sign-off)
    python -m idea91.run_k1 rows        # B4b: the classifiers that see a real REMOVE
    python -m idea91.run_k1 verify      # P10 removal success (needs the judge served)
    python -m idea91.run_k1 report      # tables/k1.md
    python -m idea91.run_k1 status      # what exists so far

The order is P20's, not a convenience.  ``checks`` runs before the freeze --
the ladders and nulls touch no real removal, so they may inform the frozen gate
resolution.  ``rows`` runs after it and refuses to start without an intact
freeze record, because those classifiers are the first to see a real removal
and the gate result must not be able to influence the gate constants.

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

from idea91 import freeze as F
from idea91 import schemas
from idea91.edits.build import materialize, read_image
from idea91.edits.sampler import HOLE_TYPES, SamplerStats, plan_k1_image
from idea91.instances.build import build_k1_scene, scene_rows
from idea91.instances.store import iter_scenes, shard_paths
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


def count_scenes(directory: Path) -> int:
    """How many images the store holds, without decoding a single mask.

    Only the ``source`` and ``image_id`` columns are read, so this costs a
    fraction of a second where materialising the scenes costs the machine.
    """
    total = 0
    for path in shard_paths(directory):
        try:
            table = pq.read_table(path, columns=["image_id", "source"])
        except Exception:
            continue
        sources = table.column("source").to_pylist()
        image_ids = table.column("image_id").to_pylist()
        total += len({i for i, s in zip(image_ids, sources) if s == "referent"})
    return total


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

    # Streamed, not loaded: the full pool's decoded masks do not fit in memory.
    n_scenes = count_scenes(instances_dir())
    scenes = iter_scenes(instances_dir(), limit=limit)
    total = min(limit, n_scenes) if limit else n_scenes
    print(f"{total} scenes from the instance store; hole types {list(hole_types)}")
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
                f"  [{n}/{total}] wrote {written.name} ({len(buffer)} edits) "
                f"{rate:.2f} img/s, eta {(total - n) / max(rate, 1e-6) / 3600:.1f} h",
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


# --- B4a, B0a, B4b, B5: the gate ---------------------------------------------


def image_dir() -> Path:
    return paths.RAW / "openimages" / "images"


def gate_cache_dir() -> Path:
    return paths.DATA_ROOT / "gate_cache"


def checks_path() -> Path:
    return paths.DATA_ROOT / "gate_checks.json"


def load_index() -> pa.Table:
    shards = sorted(index_dir().glob("shard-*.parquet"))
    if not shards:
        raise SystemExit(f"no edit index in {index_dir()}; run the edits stage first")
    return pa.concat_tables([pq.read_table(s) for s in shards])


def gate_config(args) -> "GateRunConfig":  # noqa: F821
    from idea91.gate.run import GateRunConfig

    holes = tuple(h.strip() for h in args.hole_types.split(",") if h.strip())
    return GateRunConfig(
        index=load_index(),
        image_dir=image_dir(),
        edits_root=paths.EDITS_ROOT,
        cache_dir=gate_cache_dir(),
        hole_types=holes,
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        limit_images=args.limit,
        render_cache_dir=None if args.no_render_cache else render_cache_dir(),
        render_cache_gb=args.render_cache_gb,
    )


def render_cache_dir() -> Path:
    return paths.DATA_ROOT / "gate_renders"


def stage_checks(args) -> None:
    """B4a: the JPEG ladders and the nulls.  P20 lets these run before the freeze."""
    import json

    from idea91.gate.run import run_checks

    checks = run_checks(gate_config(args))
    checks_path().write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
    print(f"\nchecks -> {checks_path()}")
    print(gate_resolution_note(checks))


def gate_resolution_note(checks: dict) -> str:
    """What check 1a says about the resolution the gate rows should run at.

    Check 1a is a sensitivity floor: if no gate row can see a whole-image q75
    re-encode, the gate is blind and a low AUROC on real removals would mean
    nothing.  The design's one declared pre-freeze contingency is raising the
    resolution in response, so this is the line the freeze sheet carries.
    """
    from idea91.gate import ladders as L

    lines = []
    for hole_type, by_quality in sorted(checks.get("global_ladder", {}).items()):
        q75 = by_quality.get(75) or by_quality.get("75") or {}
        best = max(q75.values()) if q75 else float("nan")
        verdict = "sensitive" if best >= L.GLOBAL_REQUIRED_Q75_AUROC else "NOT SENSITIVE"
        lines.append(
            f"check 1a [{hole_type}]: best q75 AUROC {best:.3f} "
            f"(needs >= {L.GLOBAL_REQUIRED_Q75_AUROC:.2f}) -- {verdict}"
        )
    return "\n".join(lines) or "check 1a: not run"


def stage_freeze(args) -> None:
    """B0a: bind P1 to P7.  The count tables must exist; a human must sign off."""
    import json

    table = count_table()
    have_counts = table != "no edit index yet"
    if args.sign_off and not have_counts:
        # Reading the sheet early is fine and useful; binding the values is not.
        raise SystemExit(
            "P20 freezes K1 'after the pool, instances, edits and count tables exist'. "
            "The edit bank is not built; run the edits stage first."
        )

    data_values: dict[str, object] = {}
    if checks_path().is_file():
        checks = json.loads(checks_path().read_text(encoding="utf-8"))
        data_values["check_1a_resolution_note"] = gate_resolution_note(checks)
        data_values["local_floors"] = {
            hole_type: {b: q for b, q in bins.items()}
            for hole_type, bins in checks.get("local_ladder", {}).items()
        }
    elif not args.sign_off:
        print(
            "note: no check-1a results yet. The ladders may run before the freeze (P20) "
            "and settle the gate resolution, so freezing now forgoes that contingency."
        )

    if not args.sign_off:
        path = paths.TABLES_ROOT / "FREEZE-B0a-sheet.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet = F.sheet(F.B0A, data_values=data_values)
        counts = (
            "\n\n## Count table committed with this freeze\n\n" + table + "\n"
            if have_counts
            else "\n\n## Count table\n\nNot yet built. P20 requires it before the freeze can be "
            "taken, so this sheet is a preview: the values are final, the counts are not.\n"
        )
        path.write_text(sheet + counts, encoding="utf-8")
        print(sheet)
        print(f"\nsheet -> {path}")
        if not have_counts:
            print("PREVIEW ONLY: the edit bank does not exist, so the freeze cannot be taken yet.")
        print("\nNothing is frozen yet. To bind these values:")
        print("  python -m idea91.run_k1 freeze --sign-off '<your name>'")
        return

    record = F.create(
        F.B0A,
        sign_off=args.sign_off,
        data_values=data_values,
        count_table=table,
        note=args.note,
        allow_dirty=args.allow_dirty,
    )
    print(f"{record.point} frozen: {record.version}")
    print(f"  commit     {record.git_commit[:12]}   tag {record.git_tag}")
    print(f"  signed off {record.signed_off_by}   {record.created_utc}")
    print(f"  record     {F.record_path(F.B0A)}")
    print("\nEvery manifest from here carries freeze_version =", record.version)


def stage_rows(args) -> None:
    """B4b: the first classifiers to see a real REMOVE.  Refuses without the freeze."""
    from idea91.gate.run import TIERS, run_gate_rows

    record = F.require(F.B0A, what="the K1 gate rows")
    print(f"freeze {record.version} intact (signed off by {record.signed_off_by})\n")

    known = {name for name, _ in TIERS}
    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip()) if args.tiers else None
    if tiers:
        unknown = [t for t in tiers if t not in known]
        if unknown:
            raise SystemExit(f"unknown tier(s) {unknown}; expected {sorted(known)}")

    results = run_gate_rows(gate_config(args), tiers=tiers)
    print(f"\n{len(results)} rows trained -> {gate_cache_dir()}")
    if tiers and set(tiers) != known:
        remaining = [t for t, _ in TIERS if t not in tiers]
        print(f"tiers not yet run: {', '.join(remaining)}")
        print("The table will mark the missing rows; the P7 verdict needs 'verdict' only.")


def load_row_results() -> list:
    """Every trained gate row from the cache, so ``report`` stands alone.

    The ``gate__`` prefix is load-bearing: a run with ``--limit`` writes its rows
    under ``limit<N>__gate__``, so a smoke test's numbers cannot reach the K1
    table by being left in the cache.
    """
    import json

    from idea91.gate.train import RowResult
    from shared.stats import Interval

    out = []
    for path in sorted(gate_cache_dir().glob("gate__*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["ci"] = Interval(**data["ci"])
        out.append(RowResult(**data))
    return out


def frequency_check(limit: int = 200) -> dict[str, tuple[float, float]]:
    """Design 4.4, once per editor: high-pass energy inside the hole versus outside."""
    from idea91.edits.build import load_edited
    from idea91.edits.composite import high_pass_residual_energy
    from idea91.masks import decode_rle

    table = load_index()
    columns = {n: table.column(n).to_pylist() for n in table.column_names}
    inside_all: list[float] = []
    outside_all: list[float] = []
    for i in range(min(limit, table.num_rows)):
        if not columns["operator"][i].endswith("REMOVE"):
            continue
        try:
            original = read_image(image_dir() / f"{columns['image_id'][i]}.jpg")
            edited = load_edited(
                original,
                {
                    "window_xyxy_px": columns["window_xyxy_px"][i],
                    "window_path": columns["window_path"][i],
                },
                paths.EDITS_ROOT,
            )
            inside, outside = high_pass_residual_energy(edited, decode_rle(columns["mask_rle"][i]))
        except Exception:
            continue
        if np.isfinite(inside) and np.isfinite(outside):
            inside_all.append(inside)
            outside_all.append(outside)
    if not inside_all:
        return {}
    return {"big_lama": (float(np.mean(inside_all)), float(np.mean(outside_all)))}


def verifier_path() -> Path:
    return paths.DATA_ROOT / "k1_removal_success.json"


def stage_verify(args) -> None:
    """P10's K1 removal success: the class-label question on both K1 classes.

    Needs the edit verifier served (``shared/judges/serve.sh edit_verifier``).
    Every answer is cached, so an interrupted pass resumes for free.
    """
    import json
    from dataclasses import asdict

    from idea91.gate import removal_success as RS

    holes = tuple(h.strip() for h in args.hole_types.split(",") if h.strip())
    result = RS.run(
        load_index(),
        image_dir(),
        paths.EDITS_ROOT,
        set_name=SET_NAME,
        limit=args.limit,
        hole_types=holes,
    )
    verifier_path().write_text(
        json.dumps([asdict(r) for r in result.rows], indent=2), encoding="utf-8"
    )
    print(f"\n{len(result.rows)} answers -> {verifier_path()}")
    for (klass, source), interval in sorted(result.by_class_and_source().items()):
        print(f"  {klass:12s} {source:22s} removal success {interval.point:.3f} (n={interval.n})")


def load_verifier_rows() -> list:
    import json

    from idea91.gate.removal_success import VerifierRow

    if not verifier_path().is_file():
        return []
    data = json.loads(verifier_path().read_text(encoding="utf-8"))
    return [VerifierRow(**{k: v for k, v in row.items() if k != "removed"}) for row in data]


def stage_report(args) -> None:
    """B5: tables/k1.md -- the verdict, its conditions, and every row behind it."""
    import json

    from idea91.analysis.k1 import CONTRAST_OBJ, K1Tables, verdict_from_results
    from idea91.gate.run import summarise_checks
    from idea91.gate.verdict import GATE_ROW_NAMES

    results = load_row_results()
    if not results:
        print(f"no trained gate rows in {gate_cache_dir()}; the verdict will read INCOMPLETE")

    checks = {}
    if checks_path().is_file():
        checks = json.loads(checks_path().read_text(encoding="utf-8"))

    holes = tuple(h.strip() for h in args.hole_types.split(",") if h.strip())
    ladder = null_1b = None
    floors: dict[str, int | None] = {}
    asserted = False
    for hole_type in holes:
        if checks:
            hole_ladder, hole_null, hole_floors = summarise_checks(checks, hole_type)
            # The verdict takes the weakest evidence across hole types: P7 asks
            # for the gate to hold "on both", so a ladder that fails on one is
            # the one that has to be reported.
            if ladder is None or not hole_ladder.passes:
                ladder = hole_ladder
            if null_1b is None or not hole_null.passes:
                null_1b = hole_null
            floors.update({f"{hole_type}/{b}": v for b, v in hole_floors.items()})
        asserted = asserted or bool(checks.get("null_1c", {}).get(hole_type, {}) and
                                    "failed" not in checks["null_1c"][hole_type])

    record = F.read(F.B0A)
    verdict = verdict_from_results(
        results,
        global_ladder=ladder,
        null_1b=null_1b,
        local_floors=floors,
        null_1c_asserted=asserted,
        freeze_version=record.version if record else "unfrozen",
        gate_resolution_note=gate_resolution_note(checks) if checks else "",
    )

    verifier_rows = load_verifier_rows()
    removal, verified_pairs = {}, {}
    if verifier_rows:
        from idea91.gate.removal_success import RemovalSuccess, auroc_on_verified_pairs

        removal = RemovalSuccess(verifier_rows).by_class_and_source()
        # P10 reports the verified-pairs AUROC of the gate, so it is read off the
        # gating row that carries the verdict -- the argmax of P7's maximum, not
        # an arbitrary row.
        gating = [
            r
            for r in results
            if r.contrast == CONTRAST_OBJ and r.scores_by_sample and r.row in GATE_ROW_NAMES
        ]
        if gating:
            best = max(gating, key=lambda r: r.auroc_mean)
            verified_pairs = auroc_on_verified_pairs(best.scores_by_sample, verifier_rows)
            print(f"verified-pairs AUROC read off {best.row} ({best.hole_type})")

    tables = K1Tables(
        results=results,
        verdict=verdict,
        removal_success=removal,
        auroc_verified_pairs=verified_pairs,
        frequency_check=frequency_check() if args.frequency_check else {},
        counts=count_summary(),
        editor="big_lama",
        editor_weights_sha256=editor_weights_sha256(),
    )
    path = tables.write(paths.TABLES_ROOT / "k1.md")
    print(verdict.report())
    print(f"\n-> {path}")


def count_summary() -> dict[str, int]:
    shards = sorted(index_dir().glob("shard-*.parquet"))
    if not shards:
        return {}
    table = pa.concat_tables([pq.read_table(s) for s in shards])
    df = table.select(["operator", "image_id", "hole_type"]).to_pandas()
    out = {"images": int(df.image_id.nunique()), "edits": len(df)}
    for name, group in df.groupby("operator", observed=True):
        out[f"edits/{name}"] = len(group)
    return out


def editor_weights_sha256() -> str:
    from idea91.edits.lama import WEIGHTS_SHA256

    return WEIGHTS_SHA256


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
    ap.add_argument(
        "stage",
        choices=["instances", "edits", "checks", "freeze", "rows", "verify", "report", "status"],
    )
    ap.add_argument("--limit", type=int, default=None, help="only the first N pool images")
    ap.add_argument("--shard", type=int, default=5_000, help="rows per shard")
    ap.add_argument("--seed", type=int, default=0, help="P1's referent choice seed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hole-types", default=",".join(DEFAULT_HOLE_TYPES))
    ap.add_argument("--freeze-version", default="unfrozen")
    ap.add_argument("--epochs", type=int, default=5, help="P6: epochs per gate row")
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="dataloader workers; 0 by default because forked workers crash this "
        "host's CUDA (DEVIATIONS D-32) and the render cache makes them near-free to lose",
    )
    ap.add_argument("--sign-off", default="", help="freeze: the name binding P1 to P7 (P20)")
    ap.add_argument("--note", default="", help="freeze: recorded with the record")
    ap.add_argument("--allow-dirty", action="store_true", help="freeze: tag a dirty tree")
    ap.add_argument("--frequency-check", action="store_true", help="report: design 4.4")
    ap.add_argument(
        "--no-render-cache",
        action="store_true",
        help="re-render every epoch instead of caching to disk (slow; for comparison)",
    )
    ap.add_argument(
        "--render-cache-gb",
        type=float,
        default=150.0,
        help="stop writing renders past this many GB (0 = unlimited); reads continue",
    )
    ap.add_argument(
        "--tiers",
        default="",
        help="rows: comma-separated subset of verdict,comparison,reported,adversary. "
        "'verdict' alone is the 24 trainings P7's PASS/FAIL needs, out of 100.",
    )
    args = ap.parse_args()

    holes = tuple(h.strip() for h in args.hole_types.split(",") if h.strip())
    bad = [h for h in holes if h not in HOLE_TYPES]
    if bad:
        raise SystemExit(f"unknown hole types {bad}; expected {list(HOLE_TYPES)}")

    if args.stage == "instances":
        stage_instances(args.limit, args.shard, args.seed, args.device)
    elif args.stage == "edits":
        stage_edits(args.limit, args.shard, args.seed, args.device, holes, args.freeze_version)
    elif args.stage == "checks":
        stage_checks(args)
    elif args.stage == "freeze":
        stage_freeze(args)
    elif args.stage == "rows":
        stage_rows(args)
    elif args.stage == "verify":
        stage_verify(args)
    elif args.stage == "report":
        stage_report(args)
    else:
        stage_status()


if __name__ == "__main__":
    main()
