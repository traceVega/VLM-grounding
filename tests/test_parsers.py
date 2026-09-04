"""Model output to a scored row: the conventions of Q-1 and Q-2, made executable."""

from __future__ import annotations

import pytest

from shared.harness import parsers as P

WH = (1000, 800)


# --- coordinate conversion ----------------------------------------------------


def test_relative_1000_scales_against_the_original_image():
    """Molmo2: 'our points format assume coordinates are scaled by 1000'."""
    assert P.to_original_pixels((500, 500), P.RELATIVE_1000, original_wh=WH) == (500.0, 400.0)


def test_percent_float_and_relative_1000_differ_by_ten(monkeypatch):
    """The bug this guards: molmo2-8b.yaml said percent_float, Molmo v1's
    convention. Both produce in-range coordinates, so nothing downstream would
    have complained -- the point would just have been in the wrong place."""
    as_1000 = P.to_original_pixels((500, 500), P.RELATIVE_1000, original_wh=WH)
    as_pct = P.to_original_pixels((50, 50), P.PERCENT_FLOAT, original_wh=WH)
    assert as_1000 == as_pct  # same place, ten times the number
    wrong = P.to_original_pixels((500, 500), P.PERCENT_FLOAT, original_wh=WH)
    assert wrong != as_1000, "reading 0-1000 numbers as percent lands somewhere else"


def test_absolute_resized_needs_the_size_actually_sent():
    with pytest.raises(ValueError, match="size actually sent"):
        P.to_original_pixels((10, 20, 30, 40), P.ABSOLUTE_RESIZED, original_wh=WH)


def test_absolute_resized_undoes_p21s_downscale():
    """99.3% of GroundingME images are downscaled by the cap, so this is the
    common path, not the corner."""
    got = P.to_original_pixels(
        (250, 200, 500, 400), P.ABSOLUTE_RESIZED, original_wh=(2000, 1600), sent_wh=(500, 400)
    )
    assert got == (1000.0, 800.0, 2000.0, 1600.0)


def test_an_unknown_convention_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown coordinate convention"):
        P.to_original_pixels((1, 2), "vibes", original_wh=WH)


# --- Qwen3-VL, GroundingME's protocol ----------------------------------------


def qwen(text, **kw):
    kw.setdefault("convention", P.RELATIVE_1000)
    kw.setdefault("original_wh", WH)
    return P.parse_qwen3vl(text, **kw)


def test_a_bbox_json_parses_to_a_box():
    got = qwen('Sure. {"bbox_2d": [100, 100, 500, 500]}')
    assert got.output_type == P.BOX and got.parse_ok
    assert got.box_xyxy_px == (100.0, 80.0, 500.0, 400.0)
    assert got.n_boxes == 1


def test_the_benchmarks_null_is_an_abstention_not_a_failure():
    """GroundingME's own instruction offers this channel, which is what makes
    P16's abstention leg measurable under the primary protocol (Q-1)."""
    got = qwen('{"bbox_2d": null}')
    assert got.output_type == P.NONE
    assert got.parse_ok is True
    assert got.abstained


def test_a_bare_none_is_read_only_when_the_protocol_asked_for_one():
    """P12's secondary adds 'output none'; the primary never asked for it."""
    assert qwen("none").output_type == P.INVALID
    assert qwen("none", none_patterns=("none",)).output_type == P.NONE


def test_none_matching_is_on_word_boundaries():
    assert qwen("Nonetheless I see a cup", none_patterns=("none",)).output_type != P.NONE


def test_loose_numbers_are_recovered_the_way_the_evaluator_does():
    got = qwen("The box is 100 100 500 500.")
    assert got.output_type == P.BOX and got.parse_ok
    assert "loose numbers" in got.note


def test_unparseable_text_is_a_row_not_an_exception():
    got = qwen("I think it is somewhere on the left.")
    assert got.output_type == P.INVALID and got.parse_ok is False
    assert got.note


def test_a_degenerate_box_is_counted_not_dropped():
    got = qwen('{"bbox_2d": [300, 300, 300, 300]}')
    assert got.output_type == P.INVALID and got.parse_ok is False
    assert got.box_xyxy_px is not None, "P15 still needs to see what came back"


def test_a_box_hanging_off_the_edge_is_clipped_not_rejected():
    got = qwen('{"bbox_2d": [-50, -50, 2000, 2000]}')
    assert got.output_type == P.BOX
    assert got.box_xyxy_px == (0.0, 0.0, 1000.0, 800.0)


def test_a_reversed_box_is_normalised():
    got = qwen('{"bbox_2d": [500, 500, 100, 100]}')
    assert got.box_xyxy_px == (100.0, 80.0, 500.0, 400.0)


def test_a_refusal_is_its_own_output_type():
    got = qwen("I can't help with identifying people in images.")
    assert got.output_type == P.REFUSAL and got.parse_ok


# --- Molmo2 -------------------------------------------------------------------


def molmo(text, **kw):
    kw.setdefault("original_wh", WH)
    return P.parse_molmo2(text, **kw)


def test_a_single_point_parses_at_0_to_1000():
    got = molmo('<points coords="1 500 250"/>')
    assert got.output_type == P.POINT and got.parse_ok
    assert got.point_xy_px == (500.0, 200.0)


def test_several_points_become_a_set_and_keep_the_first():
    got = molmo('<points coords="1 100 100 2 900 700"/>')
    assert got.output_type == P.SET
    assert got.n_boxes == 2
    assert got.point_xy_px == (100.0, 80.0)


def test_an_empty_point_list_is_one_of_p12s_two_abstentions():
    got = molmo('<points coords=" "/>')
    assert got.output_type == P.NONE and got.parse_ok


def test_a_none_sentence_is_the_other_abstention():
    got = molmo("There is no such object.", none_patterns=("no such object",))
    assert got.output_type == P.NONE


def test_molmo_output_with_neither_is_invalid():
    assert molmo("It is near the middle.").output_type == P.INVALID


def test_a_point_outside_the_frame_is_a_bad_answer_not_an_abstention():
    """The card's decoder drops out-of-frame points, but dropping them must not
    turn the row into an abstention: P16 (a) fires on the `none` rate over clean
    REMOVE and (b) on the abstention rate over CONTROL_OBJ, so counting a wild
    point as a refusal would push both rules towards firing on a model that
    never declined anything."""
    got = molmo('<points coords="1 1500 1500"/>')
    assert got.output_type == P.INVALID
    assert not got.abstained
    assert got.n_boxes == 1, "what the model offered is still recorded"


def test_an_empty_tag_and_a_wild_point_are_told_apart():
    assert molmo('<points coords=" "/>').output_type == P.NONE
    assert molmo('<points coords="1 1500 1500"/>').output_type == P.INVALID


def test_molmo_ignores_the_sent_size_because_its_frame_is_the_original():
    a = molmo('<points coords="1 500 500"/>')
    b = molmo('<points coords="1 500 500"/>', sent_wh=(100, 80))
    assert a.point_xy_px == b.point_xy_px


# --- scoring ------------------------------------------------------------------


def test_iou_of_identical_boxes_is_one_and_disjoint_is_zero():
    assert P.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert P.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_of_half_overlap():
    assert P.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_point_in_box_is_inclusive_on_the_edge():
    assert P.point_in_box((0, 0), (0, 0, 10, 10))
    assert P.point_in_box((5, 5), (0, 0, 10, 10))
    assert not P.point_in_box((11, 5), (0, 0, 10, 10))


# --- the decision not to copy the benchmark's decoder -------------------------


def test_one_convention_is_applied_regardless_of_ground_truth():
    """GroundingME picks whichever of four readings maximises IoU with the
    ground-truth box. K2 cannot: REMOVE has no ground-truth box, and P14
    compares REMOVE against ORIGINAL, so both need one frame fixed in advance."""
    text = '{"bbox_2d": [100, 100, 500, 500]}'
    as_1000 = P.parse_qwen3vl(text, convention=P.RELATIVE_1000, original_wh=WH)
    as_abs = P.parse_qwen3vl(
        text, convention=P.ABSOLUTE_RESIZED, original_wh=WH, sent_wh=WH
    )
    assert as_1000.box_xyxy_px != as_abs.box_xyxy_px
    # Neither call was given a ground-truth box to choose between them, and the
    # signature offers no way to pass one.
    assert "gt" not in P.parse_qwen3vl.__code__.co_varnames
