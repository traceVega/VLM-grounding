"""The B0a amendment: check 1a is read at q90, and says so.

The design wrote q92. It was amended on measured grounds -- the editor's own
footprint is 49.4 grey levels inside the hole against 3.98 at the harshest rung
the ladder tests, so every rung is more than an order of magnitude finer than
what K1 has to detect and the q92-versus-q90 distinction cannot bear on fitness.

What these tests protect is not the number but the disclosure: a PASS obtained
under an amended requirement has to keep saying so.
"""

from __future__ import annotations

import numpy as np

from idea91.gate import ladders as L


def verdict(q75=0.95, sensitivity=0.79, q92=0.64):
    return L.GlobalLadderVerdict(
        q75_auroc_by_row={"iii_resnet18_1024": q75},
        sensitivity_auroc_by_row={"iii_resnet18_1024": sensitivity},
        q92_auroc_by_row={"iii_resnet18_1024": q92},
    )


def test_the_requirement_is_read_at_q90():
    assert L.SENSITIVITY_RUNG == 90
    assert L.SENSITIVITY_RUNG in L.GLOBAL_LADDER_QUALITIES


def test_the_real_measurement_passes():
    """mask 0.945 / 0.794, rect 0.976 / 0.754 -- both hole types, as measured."""
    assert verdict(0.945, 0.794).passes
    assert verdict(0.976, 0.754).passes


def test_a_pass_still_reports_that_q92_was_not_met():
    """The amendment is disclosed on every PASS, not buried in a note."""
    line = verdict().verdict_line()
    assert "PASS" in line
    assert "q92" in line and "0.640" in line
    assert "not met" in line
    assert "amended" in line and "Q-15" in line


def test_the_disclosure_names_the_evidence_not_just_the_decision():
    line = verdict().verdict_line()
    assert "49.4" in line, "the editor's footprint"
    assert "3.98" in line, "the harshest rung it is compared against"


def test_a_pass_that_also_met_q92_says_nothing_extra():
    """No apology where none is owed."""
    line = verdict(q92=0.85).verdict_line()
    assert "PASS" in line
    assert "amended" not in line


def test_a_blind_gate_still_fails():
    """The amendment lowers the rung, not the bar: 0.7 at q90 is still required."""
    v = verdict(sensitivity=0.55)
    assert not v.passes
    assert "FAIL" in v.verdict_line() and "q90" in v.verdict_line()


def test_a_gate_that_cannot_see_q75_still_fails():
    v = verdict(q75=0.60)
    assert not v.passes
    assert "q75" in v.verdict_line()


def test_the_rung_and_its_threshold_are_both_frozen():
    """Both are values a later run must not be free to move."""
    from idea91 import freeze as F

    keys = " ".join(F.collect(F.B0A))
    assert "SENSITIVITY_RUNG" in keys
    assert "GLOBAL_REQUIRED_RUNG_AUROC" in keys


def test_an_empty_ladder_does_not_pass_by_default():
    empty = L.GlobalLadderVerdict(q75_auroc_by_row={}, sensitivity_auroc_by_row={})
    assert not empty.passes


def test_nan_rows_are_ignored_rather_than_counted():
    v = L.GlobalLadderVerdict(
        q75_auroc_by_row={"a": float("nan"), "b": 0.95},
        sensitivity_auroc_by_row={"a": float("nan"), "b": 0.79},
    )
    assert v.passes
    assert np.isfinite(max(v.sensitivity_auroc_by_row["b"], 0))
