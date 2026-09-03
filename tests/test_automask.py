"""The class-agnostic mask generator (design P1, P3, O6).

The pure pieces are tested without weights; the generator itself is marked
``gpu`` and skips when SAM 2 is not in the cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from idea91.instances import automask as A

from .conftest import disc


def test_point_grid_is_centred_and_covers_the_image():
    grid = A.point_grid(4, 640, 480)
    assert grid.shape == (16, 2)
    assert grid[:, 0].min() == pytest.approx(80.0)  # (0+0.5)/4 * 640
    assert grid[:, 0].max() == pytest.approx(560.0)
    assert (grid[:, 0] > 0).all() and (grid[:, 0] < 640).all()
    assert (grid[:, 1] > 0).all() and (grid[:, 1] < 480).all()


def test_mask_boxes_are_tight_and_empty_masks_are_degenerate():
    a = np.zeros((50, 60), bool)
    a[10:20, 5:25] = True
    boxes = A.mask_boxes(np.stack([a, np.zeros_like(a)]))
    assert tuple(boxes[0]) == (5.0, 10.0, 25.0, 20.0)
    assert tuple(boxes[1]) == (0.0, 0.0, 0.0, 0.0)


def test_mask_nms_collapses_duplicates_that_box_nms_keeps():
    """The measured failure: mask IoU 1.000 but box IoU 0.348, from stray pixels."""
    shape = (200, 300)
    clean = disc(shape, 220, 100, 30)
    speckled = clean.copy()
    speckled[5, 5] = True  # one stray pixel, far away, blows up the bbox

    masks = np.stack([clean, speckled])
    scores = np.array([0.99, 0.98])

    boxes = A.mask_boxes(masks)
    assert len(A.nms(boxes, scores, 0.7)) == 2, "box NMS is fooled by the stray pixel"
    assert len(A.mask_nms(masks, scores, 0.7)) == 1, "mask NMS is not"


def test_mask_nms_keeps_genuinely_different_objects():
    shape = (200, 300)
    masks = np.stack([disc(shape, 60, 60, 25), disc(shape, 220, 140, 25)])
    assert len(A.mask_nms(masks, np.array([0.9, 0.8]), 0.7)) == 2


def test_mask_nms_prefers_the_higher_scoring_of_a_pair():
    shape = (100, 100)
    m = disc(shape, 50, 50, 20)
    kept = A.mask_nms(np.stack([m, m.copy()]), np.array([0.5, 0.9]), 0.7)
    assert kept == [1]


def test_stability_score_is_high_for_a_solid_mask_and_low_for_a_fuzzy_one():
    torch = pytest.importorskip("torch")
    solid = torch.full((1, 32, 32), 8.0)  # far from the threshold either way
    fuzzy = torch.zeros((1, 32, 32))  # right at the threshold
    assert float(A.stability_score(solid)[0]) == pytest.approx(1.0)
    assert float(A.stability_score(fuzzy)[0]) < 0.5


def test_the_area_bounds_are_the_p1_floor_and_our_scene_ceiling():
    s = A.AutoMaskSettings()
    assert s.min_area_frac == 0.005  # P1: "above 0.5% area"
    assert s.max_area_frac == 0.50  # ours: the background is not an object (D-23)


@pytest.mark.gpu
def test_the_generator_recovers_objects_and_drops_the_background_and_specks():
    from idea91.instances.sam import Sam2Segmenter

    h, w = 480, 640
    image = np.zeros((h, w, 3), np.uint8)
    image[:, :] = (35, 60, 150)
    circle = disc((h, w), 140, 140, 65)
    square = np.zeros((h, w), bool)
    square[280:420, 60:210] = True
    speck = disc((h, w), 500, 380, 7)
    image[circle] = (210, 40, 40)
    image[square] = (40, 190, 70)
    image[speck] = (250, 250, 250)

    from idea91 import masks as M

    out = Sam2Segmenter().generic(image)
    assert all(r.origin == "generic" for r in out)
    assert all(0.005 < r.area_frac < 0.50 for r in out), "area bounds not applied"

    for name, truth in (("circle", circle), ("square", square)):
        best = max(M.mask_iou(r.mask, truth) for r in out)
        assert best > 0.9, f"{name} not recovered (best IoU {best:.3f})"
    assert max(M.mask_iou(r.mask, speck) for r in out) < 0.5, "the speck should be below the floor"

    # no near-duplicates survive
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            assert M.mask_iou(out[i].mask, out[j].mask) < 0.7
