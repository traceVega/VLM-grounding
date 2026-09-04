"""Q-3's decision rule and answer parsing, without the 17 GB of weights.

The probe runs unattended, so the parts that can be wrong without a GPU are
worth pinning: which convention a ratio implies, and what counts as an answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import probe_coordinates as Q  # noqa: E402

#: What the pinned processor actually produces for the two caps, measured
#: 2026-09-03 on a 7680x7680 item: frames of 1536 and 768 px.
FRAME_RATIO = 0.5


# --- the decision -------------------------------------------------------------


def test_coordinates_that_do_not_move_are_normalised():
    convention, why = Q.decide(1.00, FRAME_RATIO, max_coordinate=812)
    assert convention == "relative_1000"
    assert "did not move" in why


def test_a_0_to_100_scale_is_percent_not_relative_1000():
    """Molmo v1's convention. Magnitude is the only thing telling them apart."""
    assert Q.decide(1.00, FRAME_RATIO, max_coordinate=87)[0] == "percent_float"


def test_coordinates_that_track_the_frame_are_absolute_resized():
    convention, why = Q.decide(FRAME_RATIO, FRAME_RATIO, max_coordinate=1400)
    assert convention == "absolute_resized"
    assert "scaled with" in why


def test_a_ratio_matching_neither_is_undecided_not_nearest_match():
    """A convention chosen by 'least unlike the data' is a guess wearing a
    measurement's clothes, and every box in K2 depends on it."""
    convention, why = Q.decide(0.73, FRAME_RATIO, max_coordinate=900)
    assert convention == "UNDECIDED"
    assert "0.730" in why and "0.500" in why


def test_the_two_readings_are_far_enough_apart_to_separate():
    """With frames of 1536 and 768 the ratio is 1.0 or 0.5; the tolerances must
    not overlap, or the probe could report both."""
    assert Q.STAYS_PUT + Q.SCALES < abs(1.0 - FRAME_RATIO)


def test_noise_around_each_reading_still_decides():
    for ratio in (0.96, 1.0, 1.04):
        assert Q.decide(ratio, FRAME_RATIO, 800)[0] == "relative_1000"
    for ratio in (0.45, 0.50, 0.55):
        assert Q.decide(ratio, FRAME_RATIO, 1400)[0] == "absolute_resized"


# --- reading an answer --------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"bbox_2d": [10, 20, 30, 40]}', [10.0, 20.0, 30.0, 40.0]),
        ('Sure. {"bbox_2d": [1.5, 2.5, 3.5, 4.5]} there.', [1.5, 2.5, 3.5, 4.5]),
        ("the box is 10 20 30 40", [10.0, 20.0, 30.0, 40.0]),
        ("10,20,30,40", [10.0, 20.0, 30.0, 40.0]),
    ],
)
def test_a_box_is_read_whatever_frame_it_is_in(text, expected):
    """The probe deliberately takes the numbers raw: converting them would
    assume the convention it is trying to measure."""
    assert Q.numbers_in(text) == expected


@pytest.mark.parametrize("text", ['{"bbox_2d": null}', "no idea", "", "just 1 2 3"])
def test_a_non_answer_is_none_so_the_item_is_skipped(text):
    assert Q.numbers_in(text) is None


def test_a_malformed_bbox_does_not_raise():
    assert Q.numbers_in('{"bbox_2d": ["a", "b", "c", "d"]}') is None


# --- the caps -----------------------------------------------------------------


def test_the_two_caps_differ_by_a_factor_of_four_in_area():
    """So the resized frame halves per side, which is the signal."""
    assert Q.BIG == 4 * Q.SMALL


def test_the_big_cap_is_p21s():
    from shared.harness import tokens as T

    assert Q.BIG == T.MAX_PIXELS
