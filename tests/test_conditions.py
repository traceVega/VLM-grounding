"""P13's conditions: each control must differ from ORIGINAL in exactly one way."""

from __future__ import annotations

import pytest

from shared.harness import conditions as C
from shared.harness import schema as S


# --- the split between image and text ----------------------------------------


def test_every_condition_in_the_schema_is_resolvable():
    for condition in S.CONDITIONS:
        kwargs = {"expr": "the red cup"}
        if condition == C.T_HEAD:
            kwargs |= {"head_noun": "cup", "absent_category": "accordion"}
        if condition == C.T_ATTR:
            kwargs |= {"attr_swapped_expr": "the blue cup"}
        assert C.resolve(condition, **kwargs).condition == condition


def test_image_conditions_keep_the_words():
    for condition in C.IMAGE_CONDITIONS:
        got = C.resolve(condition, expr="the red cup")
        assert got.expr == "the red cup", "an image control must not also change the text"
        assert got.uses_edited_image


def test_text_conditions_keep_the_image():
    for condition, extra in (
        (C.T_NULL, {}),
        (C.T_HEAD, {"head_noun": "cup", "absent_category": "accordion"}),
        (C.T_ATTR, {"attr_swapped_expr": "the blue cup"}),
    ):
        got = C.resolve(condition, expr="the red cup", **extra)
        assert not got.uses_edited_image, "a text control must not also change the image"


def test_original_is_the_untouched_pair():
    got = C.resolve(C.ORIGINAL, expr="the red cup")
    assert got.expr == "the red cup" and got.operator is None


def test_the_rect_hole_type_selects_the_rect_operators():
    assert C.resolve(C.REMOVE, expr="x", hole_type="rect").operator == "RECT_REMOVE"
    assert C.resolve(C.REMOVE, expr="x", hole_type="mask").operator == "REMOVE"
    assert C.resolve(C.CONTROL_BG, expr="x", hole_type="rect").operator == "RECT_CONTROL_BG"


# --- T_NULL -------------------------------------------------------------------


def test_t_null_is_p13s_literal_replacement():
    assert C.resolve(C.T_NULL, expr="the red cup on the left").expr == "the object"


def test_t_null_uses_the_same_template_as_the_real_condition():
    """P13 says 'the grounding instruction with the expression replaced'. A second
    prompt file holding a copy would be free to drift, and then T_NULL would
    differ from ORIGINAL in the instruction as well as the expression."""
    from shared.harness import model_config as MC

    for model in ("qwen3vl-8b-instruct", "molmo2-8b"):
        templates = MC.load(model, non_kill=True).prompt_templates
        assert templates["t_null"] == templates["grounding"], model


# --- T_HEAD -------------------------------------------------------------------


def test_the_head_noun_is_replaced_everywhere_it_appears():
    """GroundingME expressions are paragraphs and 68% open with 'The object
    is ...', so the head noun recurs. One surviving mention would let the model
    ground on the real noun and the control would measure nothing."""
    expr = "The car is red. The car sits behind another car."
    got, hits = C.swap_head_noun(expr, "car", "tractor")
    assert hits == 3
    assert "car" not in got.lower()
    assert got.count("tractor") == 3


def test_the_plural_is_caught():
    got, hits = C.swap_head_noun("Two cars are parked.", "car", "tractor")
    assert hits == 1 and "tractor" in got


def test_a_longer_word_that_starts_the_same_is_left_alone():
    got, hits = C.swap_head_noun("The carpet under the car.", "car", "tractor")
    assert hits == 1
    assert "carpet" in got


def test_the_case_shape_is_preserved():
    got, _ = C.swap_head_noun("Car on the left, next to a car.", "car", "tractor")
    assert got.startswith("Tractor")
    assert got.endswith("a tractor.")


def test_matching_is_case_insensitive():
    _, hits = C.swap_head_noun("The CUP and the cup.", "cup", "harp")
    assert hits == 2


def test_t_head_refuses_to_run_without_an_absent_category():
    """P13 picks it with SAM 3; nothing here may invent one."""
    with pytest.raises(ValueError, match="SAM 3"):
        C.resolve(C.T_HEAD, expr="the red cup", head_noun="cup")


def test_t_head_refuses_when_the_noun_is_not_in_the_expression():
    """Otherwise the control would be byte-identical to ORIGINAL and would quietly
    report as a passing text-side control."""
    with pytest.raises(ValueError, match="identical to ORIGINAL"):
        C.resolve(C.T_HEAD, expr="the red one", head_noun="cup", absent_category="harp")


def test_the_category_list_is_the_size_p13_states():
    assert len(C.T_HEAD_CATEGORIES) == 50
    assert len(set(C.T_HEAD_CATEGORIES)) == 50


def test_no_category_collides_with_a_common_head_noun():
    """A replacement that is itself likely to be in the image would not be absent."""
    common = {"car", "person", "man", "woman", "chair", "window", "sign", "bottle",
              "tree", "building", "boat", "dog", "cat", "table", "door"}
    assert not (set(C.T_HEAD_CATEGORIES) & common)


# --- T_ATTR -------------------------------------------------------------------


def test_t_attr_is_never_synthesised_here():
    """P13 has Qwen3.5-9B rewrite it text-only; it is reported, not scored."""
    with pytest.raises(ValueError, match="Qwen3.5-9B"):
        C.resolve(C.T_ATTR, expr="the red cup")


# --- the prior baseline -------------------------------------------------------


def test_the_prior_box_sits_at_the_centre():
    x0, y0, x1, y1 = C.prior_box((1000, 800), 0.04)
    assert (x0 + x1) / 2 == pytest.approx(500)
    assert (y0 + y1) / 2 == pytest.approx(400)


def test_the_prior_box_has_the_median_area():
    x0, y0, x1, y1 = C.prior_box((1000, 800), 0.04)
    assert (x1 - x0) * (y1 - y0) == pytest.approx(0.04 * 1000 * 800)


def test_an_unknown_condition_is_refused():
    with pytest.raises(ValueError, match="unknown condition"):
        C.resolve("VIBES", expr="x")
