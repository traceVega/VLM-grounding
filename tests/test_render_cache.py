"""The gate's render cache: correct bytes, shared keys, safe on a kill."""

from __future__ import annotations

import numpy as np

from idea91.gate import inputs as I
from idea91.gate.dataset import GateDataset, RenderCache


def sample(monkeypatch_target=None, sample_id="s0", damage=None):
    from idea91.gate.dataset import GateSample

    return GateSample(
        sample_id=sample_id,
        image_id="img0",
        label=1,
        original_path="unused",
        window_path="unused",
        window_xyxy=(0, 0, 10, 10),
        damage=damage,
    )


class CountingDataset(GateDataset):
    """Counts how often a render actually happens."""

    def __init__(self, *args, array=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.renders = 0
        self._array = array if array is not None else np.arange(48, dtype=np.uint8).reshape(4, 4, 3)

    def _compose_and_render(self, s):
        self.renders += 1
        return self._array

    def render(self, s):
        if self.render_cache is not None:
            cached = self.render_cache.get(s, self.row)
            if cached is not None:
                return cached
        out = self._compose_and_render(s)
        if self.render_cache is not None:
            self.render_cache.put(s, self.row, out)
        return out


ROW_1024 = I.ROWS_BY_NAME["iii_resnet18_1024"]
ROW_1024_VIT = I.ROWS_BY_NAME["iii_vit_s16_1024"]
ROW_CAP = I.ROWS_BY_NAME["i_resnet18_full_cap"]


def test_a_second_read_does_not_re_render(tmp_path):
    cache = RenderCache(tmp_path)
    ds = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=cache)
    a = ds.render(sample())
    b = ds.render(sample())
    assert ds.renders == 1, "the second read should have come from disk"
    assert np.array_equal(a, b)


def test_the_cached_bytes_are_identical_not_merely_close(tmp_path):
    """Lossless on purpose: re-encoding artifacts are the signal the gate
    detects, so a JPEG cache would manufacture what check 1a measures."""
    array = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    cache = RenderCache(tmp_path)
    ds = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=cache, array=array)
    ds.render(sample())
    reread = ds.render(sample())
    assert np.array_equal(reread, array)
    assert reread.dtype == np.uint8


def test_rows_sharing_an_input_kind_share_the_cache(tmp_path):
    """The three 1,024 px rows render identically and differ only in the
    classifier on top, so one render serves all of them."""
    cache = RenderCache(tmp_path)
    first = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=cache)
    first.render(sample())
    second = CountingDataset([sample()], ROW_1024_VIT, tmp_path, render_cache=cache)
    second.render(sample())
    assert second.renders == 0, "a different row with the same input kind must reuse it"


def test_a_different_input_kind_does_not_collide(tmp_path):
    cache = RenderCache(tmp_path)
    a = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=cache)
    a.render(sample())
    b = CountingDataset([sample()], ROW_CAP, tmp_path, render_cache=cache)
    b.render(sample())
    assert b.renders == 1, "full_cap must not read full_1024's render"


def test_a_ladder_render_does_not_collide_with_the_undamaged_one(tmp_path):
    """check 1a re-encodes the same edit; the two must not share a key."""
    cache = RenderCache(tmp_path)
    clean = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=cache)
    clean.render(sample())
    damaged = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=cache)
    damaged.render(sample(damage=("global", 75)))
    assert damaged.renders == 1


def test_two_ladder_qualities_do_not_collide(tmp_path):
    cache = RenderCache(tmp_path)
    ds = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=cache)
    ds.render(sample(damage=("global", 75)))
    ds.render(sample(damage=("global", 92)))
    assert ds.renders == 2


def test_a_truncated_file_is_re_rendered_rather_than_trusted(tmp_path):
    cache = RenderCache(tmp_path)
    ds = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=cache)
    ds.render(sample())
    path = cache._path(sample(), ROW_1024)
    path.write_bytes(b"\x93NUMPY truncated")
    again = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=cache)
    assert np.array_equal(again.render(sample()), ds._array)
    assert again.renders == 1


def test_a_disabled_cache_renders_every_time(tmp_path):
    ds = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=RenderCache(None))
    ds.render(sample())
    ds.render(sample())
    assert ds.renders == 2


def test_the_hit_rate_line_reports_what_happened(tmp_path):
    cache = RenderCache(tmp_path)
    ds = CountingDataset([sample()], ROW_1024, tmp_path, render_cache=cache)
    ds.render(sample())
    ds.render(sample())
    assert "1 hits / 2" in cache.line()
