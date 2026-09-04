"""Acceptance check 1c must measure the hole that was edited, not the mask.

Written after check 1c failed on the real bank with "in-mask compositing was
violated" for compositing that was correct: it took its crop box off
``mask_rle``, the undilated mask, while the editor replaces that mask dilated by
5 px (P2). A crop placed just outside the mask box could sit inside the
dilation band, where the two images differ by construction.

It matters beyond a confusing message. ``null_1c_asserted`` feeds
:class:`K1Verdict`, so a false failure here turns a passing gate into
"NOT DECISIVE".
"""

from __future__ import annotations

import numpy as np

from idea91.edits.build import hole_from_row
from idea91.edits.composite import boxes_overlap
from idea91.edits.sampler import HOLE_DILATION_PX, hole_for
from idea91.gate.inputs import sample_paired_crop_box
from idea91.masks import bbox_xyxy, encode_rle

from .conftest import disc


def mask_and_rle(shape=(600, 800), cy=300, cx=400, r=40):
    mask = disc(shape, cy, cx, r)
    return mask, encode_rle(mask)


def test_the_hole_is_larger_than_the_mask_it_came_from():
    """The premise: a box off the mask is not a box off the hole."""
    mask, _ = mask_and_rle()
    mask_box = bbox_xyxy(mask)
    hole_box = bbox_xyxy(hole_for(mask, "mask"))
    assert hole_box[0] < mask_box[0] and hole_box[2] > mask_box[2]
    assert (mask_box[0] - hole_box[0]) == HOLE_DILATION_PX


def test_hole_from_row_reproduces_the_edited_hole():
    mask, rle = mask_and_rle()
    from_row = hole_from_row({"mask_rle": rle, "hole_type": "mask"})
    assert np.array_equal(from_row, hole_for(mask, "mask"))


def test_a_rect_hole_is_its_bounding_box():
    mask, rle = mask_and_rle()
    from_row = hole_from_row({"mask_rle": rle, "hole_type": "rect"})
    assert np.array_equal(from_row, hole_for(mask, "rect"))


def test_a_crop_cleared_against_the_mask_can_still_touch_the_hole():
    """Exactly the failure seen on the real bank: 296 differing pixels, which is
    a sliver of the dilation band clipping the crop."""
    mask, _ = mask_and_rle(r=40)
    mask_box = bbox_xyxy(mask)
    hole_box = bbox_xyxy(hole_for(mask, "mask"))
    # A box hugging the mask's right edge: clear of the mask, inside the hole.
    crop = (mask_box[2] + 1.0, mask_box[1], mask_box[2] + 20.0, mask_box[1] + 20.0)
    assert not boxes_overlap(crop, mask_box), "cleared against the mask"
    assert boxes_overlap(crop, hole_box), "but it is inside the dilated hole"


def test_the_sampler_given_hole_boxes_avoids_them():
    mask, rle = mask_and_rle()
    hole_box = bbox_xyxy(hole_from_row({"mask_rle": rle, "hole_type": "mask"}))
    rng = np.random.default_rng(0)
    for _ in range(50):
        box = sample_paired_crop_box((800, 600), [hole_box], rng=rng)
        if box is None:
            continue
        assert not boxes_overlap(box, hole_box)


def test_check_1c_uses_the_hole_not_the_mask():
    """A source-level assertion, because the difference is one word and the
    consequence is a false NOT DECISIVE on the K1 verdict."""
    import inspect

    from idea91.gate.build import assert_null_1c

    source = inspect.getsource(assert_null_1c)
    assert "hole_from_row" in source
    assert "decode_rle(a.mask_rle)" not in source
