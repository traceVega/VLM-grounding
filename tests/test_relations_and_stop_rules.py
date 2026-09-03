"""P14/P15 relations, the P10 verifier routing, and the P16 stop rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idea91.analysis import k2
from idea91.relations import verifier_rules as V
from idea91.relations.compute import PairInput, Prediction, relate

from .conftest import disc


def pred(kind: str, box=None, point=None) -> Prediction:
    return Prediction(output_type=kind, box=box, point=point)


BOX_A = (100.0, 100.0, 200.0, 200.0)
BOX_A_NUDGED = (105.0, 105.0, 205.0, 205.0)  # IoU ~0.82 with BOX_A
BOX_FAR = (600.0, 600.0, 700.0, 700.0)


def base(**overrides) -> PairInput:
    kwargs = dict(
        pair_id="p1",
        model_id="qwen3vl-8b-instruct",
        abstain_protocol="primary",
        predictions={
            "ORIGINAL": pred("box", BOX_A),
            "REMOVE": pred("box", BOX_A_NUDGED),
            "CONTROL_OBJ": pred("box", BOX_A),
            "CONTROL_BG": pred("box", BOX_A),
        },
        gt_box=BOX_A,
        verifier_clean=True,
        image_id="img1",
    )
    kwargs.update(overrides)
    return PairInput(**kwargs)


# --- P14 ---------------------------------------------------------------------


def test_same_box_when_the_model_re_predicts_the_original_box():
    r = relate(base())
    assert r["orig_correct"] is True
    assert r["same_box_50"] is True
    assert r["unresolved"] is False


def test_not_same_box_when_the_model_moves_on_removal():
    r = relate(base(predictions={**base().predictions, "REMOVE": pred("box", BOX_FAR)}))
    assert r["same_box_50"] is False


def test_same_box_is_undefined_when_the_original_box_was_wrong():
    """P14 is defined only where the ORIGINAL box is correct."""
    r = relate(base(gt_box=BOX_FAR))
    assert r["orig_correct"] is False
    assert r["same_box_50"] is None


def test_none_on_remove_is_not_same_box():
    preds = {**base().predictions, "REMOVE": pred("none")}
    r = relate(base(predictions=preds))
    assert r["none_on_remove"] is True
    assert r["box_on_remove"] is False
    assert r["same_box_50"] is False


def test_same_box_r30_uses_the_removal_region():
    region = disc((800, 800), 150, 150, 60)
    r = relate(base(removal_region=region))
    assert r["same_box_r30"] is True
    r_far = relate(
        base(
            removal_region=region,
            predictions={**base().predictions, "REMOVE": pred("box", BOX_FAR)},
        )
    )
    assert r_far["same_box_r30"] is False


def test_molmo_point_in_region_is_the_p14_rule_for_a_pointing_model():
    region = disc((800, 800), 150, 150, 40)
    preds = {
        "ORIGINAL": pred("point", point=(150.0, 150.0)),
        "REMOVE": pred("point", point=(152.0, 149.0)),
        "CONTROL_OBJ": pred("point", point=(150.0, 150.0)),
    }
    r = relate(
        base(
            model_id="molmo2-8b",
            predictions=preds,
            removal_region=region,
            removal_region_box=(110.0, 110.0, 190.0, 190.0),
        )
    )
    assert r["orig_correct"] is True
    assert r["same_box_r30"] is True

    preds_out = {**preds, "REMOVE": pred("point", point=(700.0, 700.0))}
    r_out = relate(
        base(
            model_id="molmo2-8b",
            predictions=preds_out,
            removal_region=region,
            removal_region_box=(110.0, 110.0, 190.0, 190.0),
        )
    )
    assert r_out["same_box_r30"] is False


# --- P15 flags ---------------------------------------------------------------


def test_unresolved_when_the_box_moves_under_the_control_edit():
    preds = {**base().predictions, "CONTROL_OBJ": pred("box", BOX_FAR)}
    r = relate(base(predictions=preds))
    assert r["unresolved"] is True
    assert r["shift_on_control_obj"] is True


def test_unresolved_when_the_output_type_changes_under_the_control_edit():
    preds = {**base().predictions, "CONTROL_OBJ": pred("none")}
    r = relate(base(predictions=preds))
    assert r["unresolved"] is True
    assert r["abstain_on_control_obj"] is True


def test_control_shifts_are_only_defined_on_original_correct_items():
    r = relate(base(gt_box=BOX_FAR))
    assert r["shift_on_control_obj"] is None
    assert r["shift_on_control_bg"] is None


def test_redundant_flag():
    assert relate(base(n_head_noun_instances=3))["redundant"] is True
    assert relate(base(n_head_noun_instances=1))["redundant"] is False


def test_undetermined_is_recorded_when_the_verifier_could_not_decide():
    r = relate(base(verifier_clean=None))
    assert r["undetermined"] is True
    assert r["remove_clean_verifier"] is None


def test_text_side_controls():
    preds = {
        **base().predictions,
        "T_NULL": pred("box", BOX_A),  # did not move: fails
        "T_HEAD": pred("none"),  # abstained: passes
    }
    r = relate(base(predictions=preds, prior_box=BOX_A))
    assert r["t_null_pass"] is False
    assert r["t_head_pass"] is True
    assert r["t_null_prior_pass"] is False  # the prior box explains it


# --- P10 verifier routing ----------------------------------------------------


def test_parse_yes_no_is_a_first_word_match():
    assert V.parse_yes_no("no") is False
    assert V.parse_yes_no("  Yes, there is") is True
    assert V.parse_yes_no("Maybe") is None
    assert V.parse_yes_no("") is None


def test_unparseable_counts_as_not_clean():
    d = V.CleanDecision(V.Clean.UNDETERMINED, "window", "V1_headnoun_window")
    assert V.decide(d, "banana") is V.Clean.NOT_CLEAN
    assert V.decide(d, "no") is V.Clean.CLEAN
    assert V.decide(d, "yes") is V.Clean.NOT_CLEAN


def test_no_neighbour_asks_the_window_question():
    shape = (800, 1000)
    neighbour = disc(shape, 900, 700, 20)  # far from the window
    d = V.route((100.0, 100.0, 500.0, 500.0), disc(shape, 300, 300, 40), [neighbour], (1000, 800))
    assert d.rule == "window" and d.question == "V1_headnoun_window"


def test_a_same_noun_neighbour_in_the_window_falls_back_to_the_tight_crop():
    shape = (800, 1000)
    neighbour = disc(shape, 450, 450, 20)  # inside the window, outside the tight crop
    d = V.route((100.0, 100.0, 500.0, 500.0), disc(shape, 200, 200, 30), [neighbour], (1000, 800))
    assert d.rule == "tight_crop" and d.question == "V1_headnoun_tight_crop"
    assert d.tight_crop is not None


def test_a_neighbour_in_the_tight_crop_too_stays_undetermined():
    shape = (800, 1000)
    neighbour = disc(shape, 240, 200, 20)  # right next to the referent
    d = V.route((100.0, 100.0, 500.0, 500.0), disc(shape, 200, 200, 30), [neighbour], (1000, 800))
    assert d.rule == "neighbour_in_both" and d.question is None
    assert V.decide(d, "no") is V.Clean.UNDETERMINED


def test_tight_crop_is_at_least_256_px():
    box = V.tight_crop_box((100.0, 100.0, 120.0, 120.0), (1000, 800))
    assert max(box[2] - box[0], box[3] - box[1]) >= V.TIGHT_CROP_MIN_SIDE - 1e-6


def test_class_agnostic_controls_are_not_verifiable():
    assert V.removal_success_verifiable("labelled_other_class") is True
    assert V.removal_success_verifiable("class_agnostic") is False
    assert V.removal_success_verifiable("background") is False


def test_control_validity_requires_a_yes():
    assert V.control_is_valid("yes") is True
    assert V.control_is_valid("no") is False
    assert V.control_is_valid("hmm") is False


# --- P16 stop rules ----------------------------------------------------------


def make_relations(model: str, protocol: str, n: int, *, same_box: float, none: float,
                   box: float, shift_obj: float, shift_bg: float, abstain: float) -> pd.DataFrame:
    """A frame with exactly the requested rates, so thresholds are tested, not noise."""
    idx = np.arange(n)
    def flags(rate: float) -> np.ndarray:
        return idx < int(round(rate * n))
    return pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in idx],
            "image_id": [f"img{i}" for i in idx],
            "model_id": model,
            "abstain_protocol": protocol,
            "orig_correct": True,
            "remove_clean_verifier": True,
            "remove_clean_human": True,
            "unresolved": False,
            "redundant": False,
            "undetermined": False,
            "none_on_remove": flags(none),
            "box_on_remove": flags(box),
            "same_box_50": flags(same_box),
            "same_box_r30": flags(same_box),
            "shift_on_control_obj": flags(shift_obj),
            "shift_on_control_bg": flags(shift_bg),
            "abstain_on_control_obj": flags(abstain),
            "t_null_pass": flags(0.5),
            "t_null_prior_pass": flags(0.5),
            "t_head_pass": flags(0.5),
        }
    )


def rates_for(**kwargs) -> list[k2.ModelRates]:
    protocol = kwargs.pop("protocol", "primary")
    frames = [
        make_relations(m, protocol, 200, **kwargs)
        for m in ("qwen3vl-8b-instruct", "molmo2-8b")
    ]
    df = pd.concat(frames, ignore_index=True)
    return [
        k2.compute_rates(
            df, model_id=m, abstain_protocol=protocol, clean_column="remove_clean_verifier"
        )
        for m in ("qwen3vl-8b-instruct", "molmo2-8b")
    ]


NOMINAL = dict(same_box=0.45, none=0.02, box=0.97, shift_obj=0.15, shift_bg=0.12, abstain=0.01)


def test_no_stop_rule_fires_on_nominal_rates():
    v = k2.K2Verdict(rates=rates_for(**NOMINAL), kappa_human_vs_v1=0.7)
    assert not any(r.fires for r in v.stop_rules())
    assert v.clean_column == "remove_clean_verifier"


def test_rule_a_fires_when_both_models_abstain_on_most_removals():
    rates = rates_for(**{**NOMINAL, "none": 0.65})
    rule = k2.rule_a(rates)
    assert rule.fires
    assert not rule.unfireable  # (a) is evaluated under the primary protocol, not skipped
    assert "FIRES" in rule.line()


def test_rule_a_needs_both_models():
    df = pd.concat(
        [
            make_relations("qwen3vl-8b-instruct", "primary", 200, **{**NOMINAL, "none": 0.65}),
            make_relations("molmo2-8b", "primary", 200, **NOMINAL),
        ],
        ignore_index=True,
    )
    rates = [
        k2.compute_rates(df, model_id=m, abstain_protocol="primary", clean_column="remove_clean_verifier")
        for m in ("qwen3vl-8b-instruct", "molmo2-8b")
    ]
    assert not k2.rule_a(rates).fires


def test_rule_b_prime_fires_only_when_the_shift_exceeds_the_background_shift():
    confused = rates_for(**{**NOMINAL, "shift_obj": 0.40, "shift_bg": 0.10})
    assert k2.rule_b_prime(confused).fires

    unstable = rates_for(**{**NOMINAL, "shift_obj": 0.40, "shift_bg": 0.35})
    rule = k2.rule_b_prime(unstable)
    assert not rule.fires
    assert "instability" in rule.line()


def test_rule_b_is_secondary_protocol_only():
    primary = rates_for(**NOMINAL)
    assert k2.rule_b(primary).unfireable

    secondary = rates_for(**{**NOMINAL, "abstain": 0.35}, protocol="secondary")
    assert k2.rule_b(secondary).fires


def test_rule_c_fires_below_twenty_percent_same_box():
    assert k2.rule_c(rates_for(**{**NOMINAL, "same_box": 0.15})).fires
    assert not k2.rule_c(rates_for(**{**NOMINAL, "same_box": 0.25})).fires


def test_same_box_excludes_unresolved_items():
    df = make_relations("m", "primary", 100, **NOMINAL)
    df.loc[:49, "unresolved"] = True  # half the frame is unstable
    r = k2.compute_rates(df, model_id="m", abstain_protocol="primary", clean_column="remove_clean_verifier")
    assert r.same_box.n == 50


def test_clean_column_switches_to_human_when_kappa_fails_check_3():
    assert k2.decisive_clean_column(0.75) == "remove_clean_verifier"
    assert k2.decisive_clean_column(0.4) == "remove_clean_human"
    assert k2.decisive_clean_column(None) == "remove_clean_human"


def test_frontier_reading_is_not_a_stop():
    assert "boxes are load-bearing" in k2.frontier_reading(0.02)
    assert "no added reading" in k2.frontier_reading(0.30)


def test_report_names_every_rule():
    v = k2.K2Verdict(rates=rates_for(**NOMINAL), kappa_human_vs_v1=0.7, frontier_same_box=0.01)
    text = v.report()
    for rule in ("(a)", "(b')", "(c)"):
        assert rule in text
    assert "boxes are load-bearing" in text
