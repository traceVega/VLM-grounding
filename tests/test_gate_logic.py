"""Gate inputs (P5), the check-1 ladders and nulls, and the P7 verdict."""

from __future__ import annotations

import numpy as np
import pytest

from idea91.gate import inputs as I
from idea91.gate import ladders as L
from idea91.gate.verdict import GATE_ROW_NAMES, GATE_THRESHOLD, HOLE_TYPES, K1Verdict
from shared import stats

from .conftest import disc, textured_image


# --- P5 inputs ---------------------------------------------------------------


def test_full_at_cap_only_downscales_and_respects_the_2_4_mpx_cap():
    small = textured_image((600, 800))
    assert I.full_at_cap(small) is small  # below the cap, untouched

    big = textured_image((2000, 3000))  # 6 Mpx
    out = I.full_at_cap(big)
    h, w = out.shape[:2]
    assert w * h <= I.MPX_CAP
    assert abs((w / h) - (3000 / 2000)) < 0.01  # aspect preserved


def test_full_at_longer_side_only_downscales():
    img = textured_image((400, 900))
    out = I.full_at_longer_side(img, 1024)
    assert out.shape == img.shape  # already below 1024
    out = I.full_at_longer_side(img, 512)
    assert max(out.shape[:2]) == 512


def test_tiles_cover_the_image_at_native_resolution():
    img = textured_image((1100, 1500))
    ts = I.tiles(img, 512)
    assert all(t.shape[:2] == (512, 512) for t in ts)
    covered = np.zeros((1100, 1500), dtype=bool)
    for y in I.tile_positions(1100, 512):
        for x in I.tile_positions(1500, 512):
            covered[y : y + 512, x : x + 512] = True
    assert covered.all(), "tiles must cover every pixel"
    assert len(ts) == len(I.tile_positions(1100, 512)) * len(I.tile_positions(1500, 512))


def test_tiles_subsample_deterministically_for_training():
    img = textured_image((2000, 2000))
    a = I.tiles(img, 512, max_tiles=4, rng=np.random.default_rng(0))
    b = I.tiles(img, 512, max_tiles=4, rng=np.random.default_rng(0))
    assert len(a) == 4
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_shown_crop_is_1_5x_the_hole_at_256():
    img = textured_image((800, 800))
    out = I.shown_crop(img, (300.0, 300.0, 400.0, 400.0))
    assert out.shape == (256, 256, 3)


def test_paired_crop_box_avoids_every_hole():
    box = I.sample_paired_crop_box(
        (1000, 800), [(100.0, 100.0, 300.0, 300.0), (600.0, 400.0, 800.0, 600.0)],
        rng=np.random.default_rng(0),
    )
    assert box is not None
    from idea91.edits.composite import boxes_overlap

    assert not boxes_overlap(box, (100.0, 100.0, 300.0, 300.0))
    assert not boxes_overlap(box, (600.0, 400.0, 800.0, 600.0))


def test_gate_rows_are_exactly_p5_rows_i_ii_iii():
    assert set(GATE_ROW_NAMES) == {
        "i_resnet18_full_cap",
        "ii_vit_s16_tiles_native",
        "iii_resnet18_1024",
        "iii_vit_s16_1024",
    }
    assert all(r.gating for r in I.GATE_ROWS)
    assert not any(r.gating for r in I.REPORTED_ROWS + I.ADVERSARY_ROWS)


def test_the_shown_crop_row_is_never_a_gate_row():
    """P5 keeps a hole-centred crop out of the gate: it is told where to look."""
    shown = [r for r in I.ALL_ROWS if r.input_kind == "shown_crop"]
    assert shown and not any(r.gating for r in shown)


# --- ladders -----------------------------------------------------------------


def test_jpeg_reencode_damages_more_at_lower_quality():
    img = textured_image((256, 256), seed=3)
    d95 = np.abs(I._resize(img, (256, 256)).astype(int) - L.jpeg_reencode(img, 95).astype(int)).mean()
    d30 = np.abs(img.astype(int) - L.jpeg_reencode(img, 30).astype(int)).mean()
    assert d30 > d95


def test_local_jpeg_only_touches_the_hole():
    from idea91.edits.composite import assert_in_mask

    img = textured_image((400, 400), seed=4)
    hole = disc((400, 400), 200, 200, 40)
    out = L.local_jpeg(img, hole, 30)
    assert_in_mask(img, out, hole)
    assert np.any(out[hole] != img[hole])


def test_hole_area_bins():
    assert L.hole_area_bin(0.007) == "0.5-1%"
    assert L.hole_area_bin(0.015) == "1-2%"
    assert L.hole_area_bin(0.03) == "2-5%"
    assert L.hole_area_bin(0.10) == "5-15%"


def test_local_floor_is_the_highest_detected_quality():
    assert L.local_floor({75: 0.95, 50: 0.99, 30: 0.99}) == 75
    assert L.local_floor({75: 0.55, 50: 0.80, 30: 0.99}) == 50
    assert L.local_floor({75: 0.51, 50: 0.55, 30: 0.60}) is None


def test_local_floor_line_says_blindness_not_cleanliness():
    assert "blind" in L.local_floor_line("1-2%", None)
    assert "blind" in L.local_floor_line("1-2%", 30)
    assert "sees editor-scale damage" in L.local_floor_line("1-2%", 75)


def test_global_ladder_verdict_needs_both_legs():
    ok = L.GlobalLadderVerdict({"iii_resnet18_1024": 0.97}, {"iii_resnet18_1024": 0.75})
    assert ok.passes and "PASS" in ok.verdict_line()

    blind_q75 = L.GlobalLadderVerdict({"iii_resnet18_1024": 0.80}, {"iii_resnet18_1024": 0.75})
    assert not blind_q75.passes and "q75" in blind_q75.verdict_line()

    blind_q92 = L.GlobalLadderVerdict({"iii_resnet18_1024": 0.99}, {"iii_resnet18_1024": 0.55})
    assert not blind_q92.passes
    assert "Raise the gate resolution" in blind_q92.verdict_line()


def test_null_1b_tolerance_is_half_a_point():
    assert L.NullVerdict({"a": 0.52, "b": 0.47}).passes
    bad = L.NullVerdict({"a": 0.52, "b": 0.61})
    assert not bad.passes and "b=0.610" in bad.verdict_line()


# --- P7 verdict --------------------------------------------------------------


def _verdict(auroc_by_hole: dict[str, float], **kwargs) -> K1Verdict:
    aurocs = {
        (ht, row): auroc_by_hole[ht] for ht in HOLE_TYPES for row in GATE_ROW_NAMES
    }
    defaults = dict(
        global_ladder=L.GlobalLadderVerdict({"r": 0.97}, {"r": 0.75}),
        null_1b=L.NullVerdict({r: 0.5 for r in GATE_ROW_NAMES}),
        null_1c_asserted=True,
        local_floors={b: 75 for b in L.HOLE_AREA_BINS},
    )
    defaults.update(kwargs)
    return K1Verdict(auroc_remove_vs_control_obj=aurocs, **defaults)


def test_gate_passes_at_or_below_zero_point_six_on_both_hole_types():
    v = _verdict({"mask": 0.55, "rect": 0.60})
    assert v.gate_passes and v.decisive
    assert "PASS" in v.headline()


def test_gate_fails_if_either_hole_type_is_above_threshold():
    v = _verdict({"mask": 0.55, "rect": 0.61})
    assert not v.gate_passes
    assert "FAIL" in v.headline() and "no downstream consumer" in v.headline()
    assert v.max_auroc("rect") > GATE_THRESHOLD


def test_a_low_auroc_without_sensitivity_is_not_decisive():
    v = _verdict(
        {"mask": 0.51, "rect": 0.52},
        global_ladder=L.GlobalLadderVerdict({"r": 0.80}, {"r": 0.60}),
    )
    assert v.gate_passes and not v.decisive
    assert "NOT DECISIVE" in v.headline() and "blindness" in v.headline()


def test_a_pass_without_the_nulls_is_not_decisive():
    v = _verdict({"mask": 0.51, "rect": 0.52}, null_1c_asserted=False)
    assert not v.decisive and "NOT DECISIVE" in v.headline()


def test_missing_rows_are_incomplete_not_a_pass():
    v = _verdict({"mask": 0.5, "rect": 0.5})
    v.auroc_remove_vs_control_obj.pop(("rect", GATE_ROW_NAMES[0]))
    assert not v.gate_passes and "INCOMPLETE" in v.headline()


def test_dinov2_above_0_7_is_carried_into_the_report():
    v = _verdict({"mask": 0.52, "rect": 0.53}, auroc_dinov2={("mask", "adv_dinov2b_1024"): 0.83})
    assert v.decisive  # the adversary does not gate
    assert any("DINOv2-B adversary" in c for c in v.conditions())


def test_a_pass_always_reports_its_local_floor():
    v = _verdict({"mask": 0.52, "rect": 0.53}, local_floors={"1-2%": None})
    assert any("blind" in c for c in v.conditions())


# --- stats -------------------------------------------------------------------


def test_auroc_matches_a_known_case():
    assert stats.auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert stats.auroc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert stats.auroc([0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0]) == 0.5  # all ties
    assert np.isnan(stats.auroc([1, 1], [0.2, 0.5]))  # one class only


def test_clustered_bootstrap_is_wider_than_row_level():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(50), 8)
    per_cluster = rng.random(50) < 0.4
    successes = np.repeat(per_cluster, 8)  # fully correlated inside a cluster
    by_row = stats.rate_ci(successes, n_resamples=400, seed=1)
    by_cluster = stats.rate_ci(successes, clusters, n_resamples=400, seed=1)
    assert by_cluster.half_width > by_row.half_width


def test_kappa_edges():
    assert stats.cohens_kappa(np.array([1, 0, 1]), np.array([1, 0, 1])) == 1.0
    assert abs(stats.cohens_kappa(np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1])) + 1.0) < 1e-9
    counts = np.array([[3, 0], [0, 3], [3, 0]])  # 3 raters, unanimous
    assert stats.fleiss_kappa(counts) == pytest.approx(1.0)
