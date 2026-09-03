"""P10's K1 removal success and the verified-removed-pairs AUROC."""

from __future__ import annotations

import numpy as np
import pytest

from idea91.gate import removal_success as R


def row(sample_id, operator, *, image_id="img0", hole_type="mask", source=None, answer="no"):
    return R.VerifierRow(
        sample_id=sample_id,
        image_id=image_id,
        operator=operator,
        hole_type=hole_type,
        control_source=source,
        class_label="cup",
        answer=answer,
    )


# --- reading the verifier's answer -------------------------------------------


def test_no_means_removed_and_yes_means_still_there():
    assert row("a", "REMOVE", answer="no").removed is True
    assert row("a", "REMOVE", answer="yes").removed is False


def test_an_unparseable_answer_counts_as_a_failed_removal():
    """P10: unparseable is counted as not clean -- never dropped, which would
    bias the rate upward by silently removing the hard cases."""
    for answer in ("perhaps", "", None, "I cannot tell"):
        assert row("a", "REMOVE", answer=answer).removed is False


def test_the_operator_prefix_is_stripped_not_matched_loosely():
    assert R.base_operator("RECT_CONTROL_OBJ") == "CONTROL_OBJ"
    assert R.base_operator("REMOVE") == "REMOVE"
    assert R.base_operator("RECT_REMOVE") == "REMOVE"
    # the check-1b twin and the background control are not P10's classes
    assert R.base_operator("CONTROL_OBJ_2") is None
    assert R.base_operator("RECT_CONTROL_OBJ_2") is None
    assert R.base_operator("CONTROL_BG") is None


# --- the rate ----------------------------------------------------------------


def test_removal_success_is_reported_per_class_and_source():
    success = R.RemovalSuccess([
        row("r1", "REMOVE", answer="no"),
        row("r2", "REMOVE", image_id="img1", answer="yes"),
        row("c1", "CONTROL_OBJ", source="labelled_other_class", answer="no"),
        row("c2", "CONTROL_OBJ", image_id="img1", source="class_agnostic", answer="yes"),
    ])
    table = success.by_class_and_source()
    assert table[("REMOVE", "-")].point == pytest.approx(0.5)
    assert table[("CONTROL_OBJ", "labelled_other_class")].point == pytest.approx(1.0)
    assert table[("CONTROL_OBJ", "class_agnostic")].point == pytest.approx(0.0)


def test_every_rate_carries_a_confidence_interval():
    success = R.RemovalSuccess([row(f"r{i}", "REMOVE", image_id=f"i{i}") for i in range(20)])
    interval = success.by_class_and_source()[("REMOVE", "-")]
    assert interval.n == 20
    assert interval.lo <= interval.point <= interval.hi


# --- the verified-pairs AUROC ------------------------------------------------


def make_pairs(n, *, source="labelled_other_class", both_removed=True):
    rows, scores = [], {}
    for i in range(n):
        rows.append(row(f"r{i}", "REMOVE", image_id=f"img{i}", answer="no"))
        rows.append(row(f"c{i}", "CONTROL_OBJ", image_id=f"img{i}", source=source,
                        answer="no" if both_removed else "yes"))
        scores[f"r{i}"] = 1.0  # a perfectly separating classifier
        scores[f"c{i}"] = 0.0
    return rows, scores


def test_verified_pairs_auroc_is_computed_per_source():
    rows, scores = make_pairs(10)
    result = R.auroc_on_verified_pairs(scores, rows)
    assert result == {"labelled_other_class": pytest.approx(1.0)}


def test_class_agnostic_controls_never_enter_the_verified_pairs_row():
    """P10: they have no label to ask about, so they cannot be verified."""
    rows, scores = make_pairs(10, source="class_agnostic")
    assert R.auroc_on_verified_pairs(scores, rows) == {}


def test_a_pair_whose_control_still_shows_its_object_is_excluded():
    rows, scores = make_pairs(10, both_removed=False)
    assert R.auroc_on_verified_pairs(scores, rows) == {}


def test_an_unpaired_remove_is_excluded():
    rows = [row("r0", "REMOVE", image_id="img0")]
    assert R.auroc_on_verified_pairs({"r0": 1.0}, rows) == {}


def test_pairs_are_keyed_by_image_and_hole_type():
    """A mask REMOVE must not pair with a rect CONTROL_OBJ: different edits."""
    rows = [
        row("r0", "REMOVE", image_id="img0", hole_type="mask"),
        row("c0", "CONTROL_OBJ", image_id="img0", hole_type="rect",
            source="labelled_other_class"),
    ]
    assert R.auroc_on_verified_pairs({"r0": 1.0, "c0": 0.0}, rows) == {}


def test_a_chance_classifier_gives_half():
    rows, scores = make_pairs(20)
    rng = np.random.default_rng(0)
    scores = {k: float(rng.random()) for k in scores}
    assert R.auroc_on_verified_pairs(scores, rows)["labelled_other_class"] == pytest.approx(
        0.5, abs=0.25
    )


def test_scores_missing_for_a_sample_drop_the_pair_not_the_run():
    rows, scores = make_pairs(10)
    del scores["r0"]
    result = R.auroc_on_verified_pairs(scores, rows)
    assert result["labelled_other_class"] == pytest.approx(1.0)


# --- the window view ---------------------------------------------------------


def test_the_window_view_is_capped_at_the_p10_display_size():
    image = np.zeros((3000, 4000, 3), dtype=np.uint8)
    view = R.window_view(image, (0, 0, 4000, 3000))
    assert max(view.shape[:2]) == R.WINDOW_DISPLAY_MAX_PX


def test_a_small_window_is_not_upscaled():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    view = R.window_view(image, (100, 100, 300, 300))
    assert view.shape[:2] == (200, 200)
