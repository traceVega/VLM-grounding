"""Run the K1 gate: ladders, nulls, classifiers, verdict (design B4a, B4b, B5).

Order matters and is the design's, not a convenience:

1. **Checks 1a and 1b first** (B4a), on the CONTROL_OBJ pairs only.  They decide
   whether the gate can see anything at all, and P20 lets them run *before* the
   K1 freeze precisely because they touch no real removal.
2. **The freeze** (B0a) happens between them and step 3.
3. **Then the gate rows** (B4b), which are the first classifiers to see a real
   REMOVE.

Every trained row is cached to disk by a key covering the row, hole type,
contrast and seed, so a crash costs one row rather than the run, and a re-run
with one new row does not retrain the others.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from idea91.gate import build as B
from idea91.gate import inputs as I
from idea91.gate import ladders as L
from idea91.gate.train import ADVERSARY_SEEDS, SEEDS, RowResult, run_row
from idea91.gate.verdict import GATE_ROW_NAMES
from shared.stats import Interval

CONTRAST_OBJ = "REMOVE_vs_CONTROL_OBJ"
CONTRAST_BG = "REMOVE_vs_CONTROL_BG"


@dataclass
class GateRunConfig:
    index: pa.Table
    image_dir: Path
    edits_root: Path
    cache_dir: Path
    hole_types: tuple[str, ...] = ("mask", "rect")
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 5
    device: str = "cuda"
    workers: int = 4
    limit_images: int | None = None


def _cache_path(cache: Path, key: str) -> Path:
    return cache / f"{key}.json"


def _load_cached(cache: Path, key: str) -> RowResult | None:
    path = _cache_path(cache, key)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    data["ci"] = Interval(**data["ci"])
    return RowResult(**data)


def _save_cached(cache: Path, key: str, result: RowResult) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    data = asdict(result)
    data["ci"] = asdict(result.ci)
    _cache_path(cache, key).write_text(json.dumps(data, indent=2))


def _trim(samples: list, limit: int | None) -> list:
    if not limit:
        return samples
    keep = {s.image_id for s in samples}
    keep = set(sorted(keep)[:limit])
    return [s for s in samples if s.image_id in keep]


def train_cached(
    cfg: GateRunConfig,
    key: str,
    samples: list,
    row: I.GateRow,
    hole_type: str,
    contrast: str,
    seeds: tuple[int, ...],
) -> RowResult | None:
    """One row, cached by key.  ``None`` when the sample set is too thin."""
    cached = _load_cached(cfg.cache_dir, key)
    if cached is not None:
        print(f"  [cached] {key}")
        return cached
    if len({s.label for s in samples}) < 2 or len(samples) < 8:
        print(f"  [skip]   {key}: {len(samples)} samples, needs both classes")
        return None
    started = time.perf_counter()
    result = run_row(
        samples,
        row,
        cfg.edits_root,
        hole_type=hole_type,
        contrast=contrast,
        seeds=seeds,
        epochs=cfg.epochs,
        device=cfg.device,
        workers=cfg.workers,
    )
    _save_cached(cfg.cache_dir, key, result)
    print(f"  {key}: AUROC {result.auroc_mean:.3f} ({time.perf_counter()-started:.0f}s)")
    return result


def run_checks(cfg: GateRunConfig) -> dict:
    """B4a: the ladders and the nulls, on the CONTROL_OBJ pairs only."""
    out: dict = {"global_ladder": {}, "local_ladder": {}, "null_1b": {}, "null_1c": {}}

    for hole_type in cfg.hole_types:
        pairs = _trim(B.null_1b_samples(cfg.index, cfg.image_dir, hole_type=hole_type),
                      cfg.limit_images)
        print(f"\n[{hole_type}] 1b pairs: {len(pairs)} samples "
              f"({len({s.image_id for s in pairs})} images)")
        if not pairs:
            continue

        # check 1b: the same distribution labelled both ways must be at chance
        for row in I.GATE_ROWS:
            key = f"null1b__{hole_type}__{row.name}"
            result = train_cached(cfg, key, pairs, row, hole_type, "CONTROL_OBJ_vs_CONTROL_OBJ_2",
                                  cfg.seeds)
            if result:
                out["null_1b"].setdefault(hole_type, {})[row.name] = result.auroc_mean

        # check 1a global: can the classifier see a whole-image re-encode?
        for quality in L.GLOBAL_LADDER_QUALITIES:
            samples = _trim(
                B.ladder_samples(cfg.index, cfg.image_dir, arm="global", quality=quality,
                                 hole_type=hole_type),
                cfg.limit_images,
            )
            for row in I.GATE_ROWS:
                key = f"global__{hole_type}__q{quality}__{row.name}"
                result = train_cached(cfg, key, samples, row, hole_type,
                                      f"global_q{quality}", (cfg.seeds[0],))
                if result:
                    out["global_ladder"].setdefault(hole_type, {}).setdefault(
                        quality, {})[row.name] = result.auroc_mean

        # check 1a local: can it see damage at editor scale, per hole-area bin?
        for quality in L.LOCAL_LADDER_QUALITIES:
            samples = _trim(
                B.ladder_samples(cfg.index, cfg.image_dir, arm="local", quality=quality,
                                 hole_type=hole_type),
                cfg.limit_images,
            )
            for bin_name, subset in B.by_hole_area_bin(samples).items():
                key = f"local__{hole_type}__{bin_name}__q{quality}"
                row = I.ROWS_BY_NAME["iii_resnet18_1024"]
                result = train_cached(cfg, key, subset, row, hole_type,
                                      f"local_q{quality}_{bin_name}", (cfg.seeds[0],))
                if result:
                    out["local_ladder"].setdefault(hole_type, {}).setdefault(
                        bin_name, {})[quality] = result.auroc_mean

        # check 1c: an assert, not a trained row
        try:
            out["null_1c"][hole_type] = B.assert_null_1c(
                cfg.index, cfg.image_dir, cfg.edits_root, hole_type=hole_type
            )
            print(f"[{hole_type}] check 1c: {out['null_1c'][hole_type]}")
        except AssertionError as exc:
            out["null_1c"][hole_type] = {"failed": str(exc)}
            print(f"[{hole_type}] check 1c FAILED: {exc}")

    return out


def run_gate_rows(cfg: GateRunConfig) -> list[RowResult]:
    """B4b: the classifiers that see a real REMOVE.  Runs after the K1 freeze."""
    results: list[RowResult] = []
    for hole_type in cfg.hole_types:
        for contrast, negative in ((CONTRAST_OBJ, "CONTROL_OBJ"), (CONTRAST_BG, "CONTROL_BG")):
            samples = _trim(
                B.contrast_samples(cfg.index, cfg.image_dir, hole_type=hole_type,
                                   negative=negative),
                cfg.limit_images,
            )
            print(f"\n[{hole_type}] {contrast}: {len(samples)} samples "
                  f"({len({s.image_id for s in samples})} images)")
            rows = list(I.GATE_ROWS) + list(I.REPORTED_ROWS)
            if contrast == CONTRAST_OBJ:
                rows += list(I.ADVERSARY_ROWS)
            for row in rows:
                if row.input_kind == "paired_crop":
                    continue  # check 1c is an assert
                if row.classifier == I.FORENSIC:
                    continue  # weights unpinned (OPEN-QUESTIONS Q-8)
                seeds = ADVERSARY_SEEDS if row in I.ADVERSARY_ROWS else cfg.seeds
                key = f"gate__{hole_type}__{contrast}__{row.name}"
                result = train_cached(cfg, key, samples, row, hole_type, contrast, seeds)
                if result:
                    results.append(result)
    return results


def summarise_checks(checks: dict, hole_type: str) -> tuple[L.GlobalLadderVerdict, L.NullVerdict, dict]:
    """Fold the raw AUROCs into the verdict objects P7 reports with."""
    globals_ = checks["global_ladder"].get(hole_type, {})
    q75 = globals_.get(75, {})
    q92 = globals_.get(92, {})
    ladder = L.GlobalLadderVerdict(
        q75_auroc_by_row=q75,
        q92_auroc_by_row=q92,
        all_aurocs={(r, q): v for q, rows in globals_.items() for r, v in rows.items()},
    )
    null = L.NullVerdict(auroc_by_row=checks["null_1b"].get(hole_type, {}))
    floors = {
        bin_name: L.local_floor(by_quality)
        for bin_name, by_quality in checks["local_ladder"].get(hole_type, {}).items()
    }
    return ladder, null, floors
