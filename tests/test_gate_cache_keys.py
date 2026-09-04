"""A limited run's numbers must never wear the name of a full one."""

from __future__ import annotations

import json

import pyarrow as pa

from idea91.gate import run as GR
from idea91.gate import inputs as I
from shared.stats import Interval


def config(tmp_path, limit=None):
    return GR.GateRunConfig(
        index=pa.table({"image_id": []}),
        image_dir=tmp_path,
        edits_root=tmp_path,
        cache_dir=tmp_path / "gate_cache",
        limit_images=limit,
        render_cache_dir=None,
    )


def stub_result(**over):
    base = dict(
        row="iii_resnet18_1024", classifier="resnet18", hole_type="mask",
        contrast="global_q75", auroc_mean=0.99, auroc_by_seed=[0.99],
        ci=Interval(point=0.99, lo=0.9, hi=1.0, n=30), n_images=30,
    )
    base.update(over)
    from idea91.gate.train import RowResult

    return RowResult(**base)


def test_a_limited_run_writes_under_its_own_key(tmp_path):
    cfg = config(tmp_path, limit=30)
    GR._save_cached(cfg.cache_dir, "limit30__global__mask__q75__iii_resnet18_1024", stub_result())
    names = {p.name for p in cfg.cache_dir.iterdir()}
    assert names == {"limit30__global__mask__q75__iii_resnet18_1024.json"}


def test_a_full_run_does_not_read_a_limited_runs_result(tmp_path, monkeypatch):
    """The hazard: a --limit 30 smoke test leaves entries that the real run
    reads back and reports as check 1a -- thirty images wearing the name of a
    result over nine thousand, with nothing downstream able to tell."""
    limited = config(tmp_path, limit=30)
    GR._save_cached(limited.cache_dir, "limit30__null1b__mask__iii_resnet18_1024", stub_result())

    calls = []
    monkeypatch.setattr(GR, "run_row", lambda *a, **k: calls.append(1) or stub_result())

    full = config(tmp_path, limit=None)
    samples = [type("S", (), {"label": i % 2, "image_id": f"i{i}"})() for i in range(20)]
    GR.train_cached(full, "null1b__mask__iii_resnet18_1024", samples,
                    I.ROWS_BY_NAME["iii_resnet18_1024"], "mask", "global_q75", (0,))
    assert calls, "the full run must train rather than reuse the limited result"


def test_the_same_limit_does_reuse_its_own_cache(tmp_path, monkeypatch):
    cfg = config(tmp_path, limit=30)
    GR._save_cached(cfg.cache_dir, "limit30__null1b__mask__iii_resnet18_1024", stub_result())

    def must_not_train(*a, **k):
        raise AssertionError("should have come from the cache, not retrained")

    monkeypatch.setattr(GR, "run_row", must_not_train)
    samples = [type("S", (), {"label": i % 2, "image_id": f"i{i}"})() for i in range(20)]
    got = GR.train_cached(cfg, "null1b__mask__iii_resnet18_1024", samples,
                          I.ROWS_BY_NAME["iii_resnet18_1024"], "mask", "global_q75", (0,))
    assert got is not None and got.auroc_mean == 0.99


def test_the_k1_table_loader_skips_limited_rows(tmp_path, monkeypatch):
    from idea91 import run_k1

    cache = tmp_path / "gate_cache"
    monkeypatch.setattr(run_k1, "gate_cache_dir", lambda: cache)
    GR._save_cached(cache, "gate__mask__REMOVE_vs_CONTROL_OBJ__iii_resnet18_1024",
                    stub_result(contrast="REMOVE_vs_CONTROL_OBJ"))
    GR._save_cached(cache, "limit30__gate__mask__REMOVE_vs_CONTROL_OBJ__iii_resnet18_1024",
                    stub_result(contrast="REMOVE_vs_CONTROL_OBJ", auroc_mean=0.10))

    loaded = run_k1.load_row_results()
    assert len(loaded) == 1
    assert loaded[0].auroc_mean == 0.99, "the smoke test's number must not reach the table"


def test_the_cached_file_round_trips(tmp_path):
    cfg = config(tmp_path)
    GR._save_cached(cfg.cache_dir, "gate__mask__x__row", stub_result())
    back = GR._load_cached(cfg.cache_dir, "gate__mask__x__row")
    assert back.auroc_mean == 0.99
    payload = json.loads((cfg.cache_dir / "gate__mask__x__row.json").read_text())
    assert payload["ci"]["n"] == 30
