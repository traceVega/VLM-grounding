"""The P3 control-edit samplers and acceptance check 2."""

from __future__ import annotations

import numpy as np
import pytest

from idea91 import masks as M
from idea91.edits import sampler as S
from idea91.edits.sampler import Instance

from .conftest import disc


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_exclusion_set_covers_the_dilated_referent_and_every_instance(scene):
    excl = S.build_exclusion(scene["referent"], scene["instances"])
    ref = scene["referent"].mask
    assert np.all(excl[M.dilate(ref, S.EXCLUSION_DILATION_PX)])
    for inst in scene["instances"][1:]:
        assert np.all(excl[inst.mask]), f"{inst.instance_id} missing from the exclusion set"


def test_small_class_agnostic_masks_are_not_excluded(scene):
    """P3/P1: only class-agnostic masks above 0.5% of the image are excluded."""
    shape = scene["shape"]
    speck = Instance("speck", disc(shape, 50, 50, 8), "class_agnostic", None)
    assert M.area_frac(speck.mask) < S.CLASS_AGNOSTIC_MIN_AREA_FRAC
    excl = S.build_exclusion(scene["referent"], scene["instances"] + [speck])
    assert not np.any(excl[speck.mask])


def test_sampler_picks_the_matched_candidate(scene):
    excl = S.build_exclusion(scene["referent"], scene["instances"])
    stats = S.SamplerStats()
    choice = S.sample_control_obj(
        scene["referent"], scene["instances"], excl, rng=rng(0), stats=stats
    )
    assert choice is not None
    assert choice.control_instance_id == "ok"
    assert choice.control_source == "labelled_other_class"
    assert S.AREA_RATIO_RANGE[0] <= choice.control_area_ratio <= S.AREA_RATIO_RANGE[1]
    assert abs(choice.control_centrality_delta) <= S.CENTRALITY_TOLERANCE
    # the other three were each rejected for the reason they were built to trip
    assert stats.rejected["area"] >= 1
    assert stats.rejected["centrality"] >= 1
    assert stats.rejected["exclusion_overlap"] >= 1


def test_area_and_centrality_filters_are_the_p3_numbers(scene):
    ref = scene["referent"].mask
    assert M.area_ratio(scene["by_id"]["too_big"].mask, ref) > S.AREA_RATIO_RANGE[1]
    assert abs(M.centrality_delta(scene["by_id"]["off_axis"].mask, ref)) > S.CENTRALITY_TOLERANCE
    assert abs(M.centrality_delta(scene["by_id"]["ok"].mask, ref)) <= S.CENTRALITY_TOLERANCE


def test_a_candidate_whose_hole_would_clip_a_neighbour_is_rejected(scene):
    """'no overlap with the exclusion set after the candidate's own mask is removed'."""
    excl = S.build_exclusion(scene["referent"], scene["instances"])
    crowded = scene["by_id"]["crowded"]
    hole = S.hole_for(crowded.mask, "mask")
    assert S._overlaps_exclusion(hole, excl, crowded.mask) > 0


def test_dropped_when_no_candidate_survives(scene):
    """Items with no valid CONTROL_OBJ are dropped and counted (P3)."""
    lonely = [scene["referent"], scene["by_id"]["too_big"], scene["by_id"]["off_axis"]]
    excl = S.build_exclusion(scene["referent"], lonely)
    stats = S.SamplerStats()
    assert (
        S.sample_control_obj(scene["referent"], lonely, excl, rng=rng(1), stats=stats) is None
    )
    assert stats.rejected["no_candidate"] == 1


def test_second_control_is_a_different_instance(scene):
    """Acceptance check 1b needs two independent CONTROL_OBJ edits per image."""
    shape = scene["shape"]
    extra = Instance("ok2", disc(shape, 300, 150, 42), "labelled_other_class", "hat")
    instances = scene["instances"] + [extra]
    excl = S.build_exclusion(scene["referent"], instances)
    first = S.sample_control_obj(scene["referent"], instances, excl, rng=rng(2))
    second = S.sample_control_obj(
        scene["referent"],
        instances,
        excl,
        rng=rng(2),
        exclude_instance_ids=(first.control_instance_id,),
    )
    assert first is not None and second is not None
    assert first.control_instance_id != second.control_instance_id


def test_class_agnostic_is_only_the_fallback(scene):
    """P3 prefers a labelled instance; class_agnostic is last."""
    shape = scene["shape"]
    agnostic = Instance("blob", disc(shape, 700, 400, 44), "class_agnostic", None)
    instances = [scene["referent"], scene["by_id"]["ok"], agnostic]
    excl = S.build_exclusion(scene["referent"], instances)
    choice = S.sample_control_obj(scene["referent"], instances, excl, rng=rng(3))
    assert choice.control_source == "labelled_other_class"

    without_labelled = [scene["referent"], agnostic]
    excl2 = S.build_exclusion(scene["referent"], without_labelled)
    choice2 = S.sample_control_obj(scene["referent"], without_labelled, excl2, rng=rng(3))
    assert choice2.control_source == "class_agnostic"


def test_control_bg_lands_clear_of_every_instance(scene):
    excl = S.build_exclusion(scene["referent"], scene["instances"])
    bg = S.sample_control_bg(scene["referent"], excl, rng=rng(4))
    assert bg is not None
    assert not np.any(bg.hole & excl)
    assert bg.hole.sum() == S.hole_for(scene["referent"].mask, "mask").sum()
    assert bg.control_source == "background"


def test_control_bg_gives_up_when_the_image_is_full(scene):
    shape = scene["shape"]
    everything = Instance("wall", np.ones(shape, dtype=bool), "labelled_other_class", "wall")
    excl = S.build_exclusion(scene["referent"], [scene["referent"], everything])
    stats = S.SamplerStats()
    assert S.sample_control_bg(scene["referent"], excl, rng=rng(5), stats=stats) is None
    assert stats.rejected["max_tries"] == 1


@pytest.mark.parametrize("hole_type", ["mask", "rect"])
def test_k1_plan_is_always_edited_and_passes_check_2(scene, hole_type):
    shape = scene["shape"]
    extra = Instance("ok2", disc(shape, 300, 150, 42), "labelled_other_class", "hat")
    instances = scene["instances"] + [extra]
    plans = S.plan_k1_image(scene["referent"], instances, hole_type=hole_type, rng=rng(6))
    assert plans is not None
    ops = [p.operator for p in plans]
    prefix = "" if hole_type == "mask" else "RECT_"
    assert f"{prefix}REMOVE" in ops and f"{prefix}CONTROL_OBJ" in ops
    assert f"{prefix}CONTROL_OBJ_2" in ops  # check 1b
    assert f"{prefix}CONTROL_BG" in ops
    S.assert_no_exclusion_overlap(plans, scene["referent"], instances)


def test_check_2_catches_a_hand_built_violation(scene):
    """The assert is an independent pass, so it must actually fail on a bad plan."""
    instances = scene["instances"]
    bad = S.EditPlan(
        operator="CONTROL_OBJ",
        hole=S.hole_for(scene["by_id"]["crowded"].mask, "mask"),
        mask=scene["by_id"]["crowded"].mask,
        hole_type="mask",
        control_instance_id="crowded",
        control_source="labelled_other_class",
    )
    with pytest.raises(AssertionError, match="acceptance check 2"):
        S.assert_no_exclusion_overlap([bad], scene["referent"], instances)


def test_k2_plan_has_no_second_control(scene):
    plans = S.plan_k2_item(scene["referent"], scene["instances"], rng=rng(7))
    assert plans is not None
    assert [p.operator for p in plans] == ["REMOVE", "CONTROL_OBJ", "CONTROL_BG"]


def test_rect_hole_is_the_bounding_box(scene):
    mask = scene["by_id"]["ok"].mask
    hole = S.hole_for(mask, "rect")
    assert np.array_equal(hole, M.mask_from_box(M.bbox_xyxy(mask), mask.shape))
    assert hole.sum() > mask.sum()


def test_rle_round_trip(scene):
    mask = scene["referent"].mask
    assert np.array_equal(M.decode_rle(M.encode_rle(mask)), mask)


# --- P3 reading B: the exclusion rule chosen for Q-12 ------------------------


def test_the_referent_is_absolute_under_reading_b(scene):
    """A control hole may not touch the referent's dilated mask at all."""
    shape = scene["shape"]
    ref = scene["referent"]
    hugging = Instance("hug", disc(shape, 355, 400, 20), "labelled_other_class", "collar")
    hole = S.hole_for(hugging.mask, "mask")
    assert np.any(hole & M.dilate(ref.mask, S.EXCLUSION_DILATION_PX))

    reason = S.exclusion_violation(hole, hugging.mask, ref, [ref, hugging], candidate_id="hug")
    assert reason == "touches the referent"


def test_a_small_nick_of_another_object_is_tolerated(scene):
    """The change that makes P3 satisfiable against a dense segmentation."""
    shape = scene["shape"]
    ref = scene["referent"]
    cand = Instance("cand", disc(shape, 700, 400, 44), "labelled_other_class", "dog")
    # a big neighbour, barely grazed by the candidate's dilation ring
    neighbour = Instance("big", disc(shape, 700, 500, 60), "labelled_other_class", "sofa")
    instances = [ref, cand, neighbour]
    hole = S.hole_for(cand.mask, "mask")

    bitten = (hole & neighbour.mask & ~cand.mask).sum() / neighbour.mask.sum()
    assert 0 < bitten <= S.CONTROL_BITE_TOLERANCE, f"fixture should graze, bit {bitten:.2%}"
    assert S.exclusion_violation(hole, cand.mask, ref, instances, candidate_id="cand") is None


def test_a_large_bite_of_another_object_is_still_refused(scene):
    shape = scene["shape"]
    ref = scene["referent"]
    cand = Instance("cand", disc(shape, 700, 400, 44), "labelled_other_class", "dog")
    swallowed = Instance("tiny", disc(shape, 745, 400, 8), "labelled_other_class", "tag")
    instances = [ref, cand, swallowed]
    hole = S.hole_for(cand.mask, "mask")

    reason = S.exclusion_violation(hole, cand.mask, ref, instances, candidate_id="cand")
    assert reason is not None and "tiny" in reason


def test_a_mask_lying_inside_the_candidate_is_a_part_of_it_not_damage(scene):
    """A nested part-mask disappears with the candidate; that is correct."""
    shape = scene["shape"]
    ref = scene["referent"]
    cand = Instance("cand", disc(shape, 700, 400, 44), "labelled_other_class", "dog")
    part = Instance("part", disc(shape, 700, 400, 20), "class_agnostic", None)
    hole = S.hole_for(cand.mask, "mask")
    assert np.all(part.mask & cand.mask == part.mask)  # fully nested
    assert S.exclusion_violation(hole, cand.mask, ref, [ref, cand, part], candidate_id="cand") is None


def test_sub_floor_class_agnostic_masks_are_ignored_by_the_rule(scene):
    shape = scene["shape"]
    ref = scene["referent"]
    cand = Instance("cand", disc(shape, 700, 400, 44), "labelled_other_class", "dog")
    speck = Instance("speck", disc(shape, 748, 400, 5), "class_agnostic", None)
    assert M.area_frac(speck.mask) < S.CLASS_AGNOSTIC_MIN_AREA_FRAC
    hole = S.hole_for(cand.mask, "mask")
    assert S.exclusion_violation(hole, cand.mask, ref, [ref, cand, speck], candidate_id="cand") is None


def test_check_2_asserts_the_same_rule_the_sampler_sampled_under(scene):
    shape = scene["shape"]
    extra = Instance("ok2", disc(shape, 300, 150, 42), "labelled_other_class", "hat")
    instances = scene["instances"] + [extra]
    plans = S.plan_k1_image(scene["referent"], instances, rng=rng(11))
    assert plans is not None
    S.assert_no_exclusion_overlap(plans, scene["referent"], instances)

    touching = S.EditPlan(
        operator="CONTROL_OBJ",
        hole=S.hole_for(disc(shape, 355, 400, 20), "mask"),  # against the referent
        mask=disc(shape, 355, 400, 20),
        hole_type="mask",
        control_instance_id="hug",
        control_source="labelled_other_class",
    )
    with pytest.raises(AssertionError, match="touches the referent"):
        S.assert_no_exclusion_overlap([touching], scene["referent"], instances)
