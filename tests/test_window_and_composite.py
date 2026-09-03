"""The window operator (P2) and the compositing assert (acceptance check 1c)."""

from __future__ import annotations

import numpy as np
import pytest

from idea91.edits import composite as C
from idea91.edits.window import (
    WINDOW_MAX_SIDE,
    WINDOW_MIN_SIDE,
    Window,
    prepare_for_lama,
    restore_from_lama,
    window_box,
)
from idea91.masks import bbox_xyxy, dilate

from .conftest import disc, textured_image


def test_window_is_four_times_the_hole_clamped_to_the_p2_range():
    # 4 * 100 = 400 -> below the 768 floor
    w = window_box((450.0, 450.0, 550.0, 550.0), (4000, 4000))
    assert (w.width, w.height) == (WINDOW_MIN_SIDE, WINDOW_MIN_SIDE)

    # 4 * 300 = 1200, inside the range
    w = window_box((1000.0, 1000.0, 1300.0, 1300.0), (4000, 4000))
    assert (w.width, w.height) == (1200, 1200)

    # 4 * 900 = 3600 -> above the 2048 ceiling
    w = window_box((1000.0, 1000.0, 1900.0, 1900.0), (4000, 4000))
    assert (w.width, w.height) == (WINDOW_MAX_SIDE, WINDOW_MAX_SIDE)


def test_window_is_clipped_to_the_image_and_still_contains_the_hole():
    hole = (10.0, 10.0, 110.0, 110.0)  # against the top-left corner
    w = window_box(hole, (1000, 1000))
    assert w.x0 == 0 and w.y0 == 0
    assert w.x1 <= 1000 and w.y1 <= 1000
    assert w.x0 <= hole[0] and w.y0 <= hole[1] and w.x1 >= hole[2] and w.y1 >= hole[3]


def test_window_of_a_small_image_is_the_whole_image():
    w = window_box((100.0, 100.0, 200.0, 200.0), (500, 400))
    assert w.as_list() == [0, 0, 500, 400]


def test_window_refuses_a_hole_it_cannot_cover():
    with pytest.raises(ValueError, match="does not fit"):
        window_box((0.0, 0.0, 3000.0, 3000.0), (4000, 4000))


def test_lama_resize_round_trip_returns_the_native_window_size():
    img = textured_image((2500, 2500), seed=1)
    hole = disc((2500, 2500), 1250, 1250, 100)
    prepared = prepare_for_lama(img, hole, max_side=2048)
    assert max(prepared.resized_size) == 2048
    assert prepared.image.shape[0] % 8 == 0 and prepared.image.shape[1] % 8 == 0
    back = restore_from_lama(prepared.image, prepared)
    assert back.shape == img.shape


def test_lama_prepare_is_a_no_op_below_the_cap():
    img = textured_image((1024, 1024), seed=2)
    hole = disc((1024, 1024), 512, 512, 50)
    prepared = prepare_for_lama(img, hole)
    assert prepared.resized_size == (1024, 1024)
    assert np.array_equal(prepared.image, img)


# --- compositing -------------------------------------------------------------


def _fake_fill(window: np.ndarray, seed: int = 7) -> np.ndarray:
    """Stand-in for LaMa: something that differs everywhere, so an in-mask
    composite that leaked would be caught."""
    rng = np.random.default_rng(seed)
    return np.clip(window.astype(np.int16) + rng.integers(40, 80, window.shape), 0, 255).astype(
        np.uint8
    )


def test_composite_changes_only_the_hole():
    img = textured_image((400, 400), seed=3)
    hole = disc((400, 400), 200, 200, 30)
    out = C.composite_in_mask(img, _fake_fill(img), hole)
    C.assert_in_mask(img, out, hole)
    assert np.any(out[hole] != img[hole])
    assert np.array_equal(out[~hole], img[~hole])


def test_assert_in_mask_catches_a_leak():
    img = textured_image((200, 200), seed=4)
    hole = disc((200, 200), 100, 100, 20)
    bad = img.copy()
    bad[5, 5] = (0, 0, 0)
    with pytest.raises(C.CompositingError, match="outside the hole"):
        C.assert_in_mask(img, bad, hole)


def test_compose_full_rebuilds_the_edited_image_from_the_window():
    img = textured_image((1200, 1600), seed=5)
    hole = disc((1200, 1600), 800, 600, 60)
    win = window_box(bbox_xyxy(hole), (1600, 1200))
    win_img = win.crop(img)
    win_hole = win.crop(hole)
    edited_win = C.composite_in_mask(win_img, _fake_fill(win_img), win_hole)

    full = C.compose_full(img, edited_win, win)
    C.assert_in_mask(img, full, hole)
    assert np.array_equal(full[win.y0 : win.y1, win.x0 : win.x1], edited_win)


def test_paired_crop_outside_both_holes_is_identical_check_1c():
    """Acceptance check 1c: this is why a hidden-crop gate is 0.5 by construction."""
    shape = (800, 1000)
    img = textured_image(shape, seed=6)
    ref_hole = dilate(disc(shape, 300, 400, 40), 5)
    ctl_hole = dilate(disc(shape, 700, 400, 44), 5)

    remove = C.composite_in_mask(img, _fake_fill(img, 11), ref_hole)
    control = C.composite_in_mask(img, _fake_fill(img, 12), ctl_hole)

    crop_box = (50.0, 50.0, 200.0, 200.0)  # outside both holes
    shared = C.assert_paired_crop_identical(
        remove, control, crop_box, bbox_xyxy(ref_hole), bbox_xyxy(ctl_hole)
    )
    assert shared.shape == (150, 150, 3)


def test_paired_crop_refuses_a_crop_that_touches_a_hole():
    shape = (400, 400)
    img = textured_image(shape, seed=8)
    ref_hole = dilate(disc(shape, 100, 100, 20), 5)
    ctl_hole = dilate(disc(shape, 300, 300, 20), 5)
    remove = C.composite_in_mask(img, _fake_fill(img), ref_hole)
    control = C.composite_in_mask(img, _fake_fill(img), ctl_hole)
    with pytest.raises(ValueError, match="touches a hole"):
        C.assert_paired_crop_identical(
            remove, control, (80.0, 80.0, 150.0, 150.0), bbox_xyxy(ref_hole), bbox_xyxy(ctl_hole)
        )


def test_high_pass_energy_is_higher_inside_a_smeared_hole():
    """Design 4.4's frequency check, on a blur that stands in for the editor."""
    import cv2

    shape = (400, 400)
    img = textured_image(shape, seed=9)
    hole = disc(shape, 200, 200, 60)
    smeared = C.composite_in_mask(img, cv2.GaussianBlur(img, (31, 31), 0), hole)
    inside, outside = C.high_pass_residual_energy(smeared, hole)
    assert inside < outside  # a smear removes high frequencies; the ratio is the signal
