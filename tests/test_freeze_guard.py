"""P20's freeze guard where the training happens, not only at the CLI.

Written after the guard was bypassed for real (DEVIATIONS D-31): a throughput
probe called ``run_row`` directly and trained on REMOVE vs CONTROL_OBJ before
B0a existed, because the only check lived in ``run_k1 rows``.
"""

from __future__ import annotations

import pytest

from idea91 import freeze as F
from idea91.gate import train as T


# --- which contrasts P20 protects --------------------------------------------


def test_the_real_removal_contrasts_are_recognised():
    assert T._contrast_has_real_removals("REMOVE_vs_CONTROL_OBJ")
    assert T._contrast_has_real_removals("REMOVE_vs_CONTROL_BG")


def test_the_pre_freeze_contrasts_are_let_through():
    """P20: 'the JPEG ladder and the nulls of check 1 may run before'."""
    for contrast in (
        "CONTROL_OBJ_vs_CONTROL_OBJ_2",  # check 1b null
        "global_q75",                    # check 1a global ladder
        "global_q92",
        "local_q50_1-2%",                # check 1a local ladder
    ):
        assert not T._contrast_has_real_removals(contrast), contrast


def test_an_unknown_contrast_is_not_silently_trusted():
    """Guarded by default: a contrast added later must be named safe on purpose."""
    assert not T._contrast_has_real_removals("something_new")
    # ...but anything built on REMOVE is caught without anyone remembering to.
    assert T._contrast_has_real_removals("REMOVE_vs_ANYTHING_ADDED_LATER")


# --- the guard itself ---------------------------------------------------------


def test_training_on_real_removals_refuses_without_the_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(F.paths, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="no freeze record exists"):
        T.run_row(
            [],
            T.I.ROWS_BY_NAME["iii_resnet18_1024"],
            tmp_path,
            hole_type="mask",
            contrast="REMOVE_vs_CONTROL_OBJ",
            seeds=(),
        )


def test_the_refusal_names_the_contrast_that_triggered_it(tmp_path, monkeypatch):
    monkeypatch.setattr(F.paths, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="REMOVE_vs_CONTROL_BG"):
        T.run_row(
            [],
            T.I.ROWS_BY_NAME["iii_resnet18_1024"],
            tmp_path,
            hole_type="mask",
            contrast="REMOVE_vs_CONTROL_BG",
            seeds=(),
        )


def test_a_ladder_runs_before_the_freeze_without_asking(tmp_path, monkeypatch):
    """These are exactly what P20 permits pre-freeze; the guard must not block
    them, or check 1a could never inform the frozen gate resolution."""
    monkeypatch.setattr(F.paths, "REPO_ROOT", tmp_path)
    result = T.run_row(
        [],
        T.I.ROWS_BY_NAME["iii_resnet18_1024"],
        tmp_path,
        hole_type="mask",
        contrast="global_q75",
        seeds=(),
    )
    assert result.contrast == "global_q75"


def test_an_intact_freeze_lets_a_real_removal_row_run(tmp_path, monkeypatch):
    monkeypatch.setattr(F.paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(F, "_git", lambda *a: "deadbeef")
    monkeypatch.setattr(F, "working_tree_clean", lambda: True)
    F.create(F.B0A, sign_off="A Human", tag=False)
    result = T.run_row(
        [],
        T.I.ROWS_BY_NAME["iii_resnet18_1024"],
        tmp_path,
        hole_type="mask",
        contrast="REMOVE_vs_CONTROL_OBJ",
        seeds=(),
    )
    assert result.contrast == "REMOVE_vs_CONTROL_OBJ"


def test_the_escape_hatch_is_explicit_and_named(tmp_path, monkeypatch):
    """allow_unfrozen exists for the ladders' own callers; it must be opt-in, so
    that bypassing P20 is always a visible decision in the calling code."""
    monkeypatch.setattr(F.paths, "REPO_ROOT", tmp_path)
    result = T.run_row(
        [],
        T.I.ROWS_BY_NAME["iii_resnet18_1024"],
        tmp_path,
        hole_type="mask",
        contrast="REMOVE_vs_CONTROL_OBJ",
        seeds=(),
        allow_unfrozen=True,
    )
    assert result.contrast == "REMOVE_vs_CONTROL_OBJ"
