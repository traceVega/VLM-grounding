"""P20's two-stage freeze: what it binds, and what it refuses."""

from __future__ import annotations

import json

import pytest

from idea91 import freeze as F


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Freeze records under tmp_path, and no git tagging."""
    monkeypatch.setattr(F.paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(F, "_git", lambda *a: "deadbeef" if a[:1] == ("rev-parse",) else "")
    monkeypatch.setattr(F, "working_tree_clean", lambda: True)
    return tmp_path


# --- what the registry covers ------------------------------------------------


def test_every_registered_constant_exists():
    """A renamed constant must break the freeze loudly, not silently drop out."""
    values = F.collect(F.B0A)
    assert len(values) == len(F.B0A_REGISTRY) + len(F.B0A_BEHAVIOURS)
    assert all(v is not None for v in values.values())


def test_behavioural_decisions_are_frozen_by_source(monkeypatch):
    """Q-5 and Q-7 are behaviours, not numbers; no constant could hold them."""
    values = F.collect(F.B0A)
    keys = [k for k in values if k.endswith("()")]
    assert len(keys) == len(F.B0A_BEHAVIOURS)
    for key in keys:
        assert values[key]["source_sha256"]
        assert values[key]["why"], "a frozen behaviour must say why it is frozen"


def test_editing_a_frozen_behaviour_is_detected(isolated, monkeypatch):
    F.create(F.B0A, sign_off="A Human", tag=False)
    real = F._source_digest

    def altered(module_name, qualified_name):
        if qualified_name == "local_ladder_pair":
            return "0" * 64  # someone rewrites the local ladder's reference
        return real(module_name, qualified_name)

    monkeypatch.setattr(F, "_source_digest", altered)
    _, drifts = F.verify(F.B0A)
    assert len(drifts) == 1
    assert "local_ladder_pair" in drifts[0].key


def test_a_reformat_is_not_a_change_of_substance():
    """Trailing whitespace must not read as an edit to the pre-registration."""
    import hashlib

    a = "def f():\n    return 1\n"
    b = "def f():   \n    return 1\n   "
    norm = lambda t: "\n".join(x.rstrip() for x in t.splitlines()).strip()  # noqa: E731
    assert hashlib.sha256(norm(a).encode()).digest() == hashlib.sha256(norm(b).encode()).digest()


def test_the_registry_covers_the_values_p20_names():
    """P1 to P7 -- every definition the K1 freeze binds is represented."""
    definitions = {d for d, _, _ in F.B0A_REGISTRY}
    assert definitions == {"P1", "P2", "P3", "P5", "P6", "P7"}
    # P4 is the operator set itself (REMOVE, CONTROL_OBJ, CONTROL_BG), fixed by
    # the schema enum rather than by a tunable constant.


def test_the_decisions_taken_during_the_build_are_all_frozen():
    """The values that were open questions must not be able to move afterwards."""
    keys = " ".join(F.collect(F.B0A))
    for decided in (
        "CONTROL_BITE_TOLERANCE",   # Q-12/Q-13: P3 reading B
        "MAX_AREA_FRAC",            # D-23: the class-agnostic cap
        "HOLE_AREA_BIN_EDGES",      # the local ladder's bins
        "GATE_THRESHOLD",           # P7 itself
    ):
        assert decided in keys


def test_a_missing_constant_names_the_remedy(monkeypatch):
    monkeypatch.setattr(F, "REGISTRY", {F.B0A: (("P1", "idea91.masks", "NOT_A_REAL_NAME"),)})
    with pytest.raises(AttributeError, match="update the registry"):
        F.collect(F.B0A)


# --- the digest --------------------------------------------------------------


def test_the_digest_is_stable_and_order_independent():
    a = F.digest({"x": [1, 2], "y": "z"})
    b = F.digest({"y": "z", "x": [1, 2]})
    assert a == b
    assert a != F.digest({"x": [1, 3], "y": "z"})


def test_dataclasses_and_tuples_survive_canonicalisation():
    """GATE_ROWS is a tuple of dataclasses; a change to any field must show."""
    from idea91.gate.inputs import GateRow

    one = F._canonical((GateRow("a", "resnet18", "full_cap", True),))
    two = F._canonical((GateRow("a", "resnet18", "full_cap", False),))
    assert one != two
    assert json.dumps(one)  # JSON-safe


# --- creating a freeze -------------------------------------------------------


def test_a_freeze_records_the_values_and_the_sign_off(isolated):
    record = F.create(F.B0A, sign_off="A Human", count_table="| a | b |", tag=False)
    assert record.signed_off_by == "A Human"
    assert record.values_sha256 == F.digest(F.collect(F.B0A))
    assert record.version.startswith("B0a@")
    assert record.count_table_sha256
    assert F.record_path(F.B0A).is_file()


def test_the_record_carries_the_count_table_it_was_taken_against(isolated):
    """``tables/`` is gitignored, so the committed record must hold the counts itself."""
    table = "| set | edits |\n|---|---|\n| openimages_pool | 12,345 |"
    F.create(F.B0A, sign_off="A Human", count_table=table, tag=False)
    reread = F.read(F.B0A)
    assert reread.count_table == table
    assert reread.count_table_sha256 == F.hashlib.sha256(table.encode()).hexdigest()


def test_a_freeze_needs_a_human(isolated):
    with pytest.raises(SystemExit, match="binds a human decision"):
        F.create(F.B0A, sign_off="   ", tag=False)


def test_a_dirty_tree_is_refused(isolated, monkeypatch):
    monkeypatch.setattr(F, "working_tree_clean", lambda: False)
    with pytest.raises(SystemExit, match="dirty"):
        F.create(F.B0A, sign_off="A Human", tag=False)


def test_refreezing_is_refused_because_it_forks_the_document(isolated):
    F.create(F.B0A, sign_off="A Human", tag=False)
    with pytest.raises(SystemExit, match="already frozen"):
        F.create(F.B0A, sign_off="A Human", tag=False)


# --- the guard ---------------------------------------------------------------


def test_gate_rows_refuse_to_run_before_the_freeze(isolated):
    with pytest.raises(SystemExit, match="no freeze record exists"):
        F.require(F.B0A, what="the K1 gate rows")


def test_an_intact_freeze_lets_the_stage_run(isolated):
    F.create(F.B0A, sign_off="A Human", tag=False)
    assert F.require(F.B0A).signed_off_by == "A Human"


def test_a_value_moved_after_the_freeze_stops_the_run(isolated, monkeypatch):
    """The whole point: the gate result cannot influence the gate constants."""
    F.create(F.B0A, sign_off="A Human", tag=False)

    import idea91.gate.verdict as V

    monkeypatch.setattr(V, "GATE_THRESHOLD", 0.75)  # someone "adjusts" the gate
    with pytest.raises(SystemExit) as exc:
        F.require(F.B0A, what="the K1 gate rows")
    message = str(exc.value)
    assert "GATE_THRESHOLD" in message
    assert "0.6" in message and "0.75" in message, "the message names both values"
    assert "forks the document" in message


def test_verify_names_every_drifted_value(isolated, monkeypatch):
    F.create(F.B0A, sign_off="A Human", tag=False)
    import idea91.edits.sampler as S
    import idea91.gate.verdict as V

    monkeypatch.setattr(V, "GATE_THRESHOLD", 0.75)
    monkeypatch.setattr(S, "CONTROL_BITE_TOLERANCE", 0.5)
    _, drifts = F.verify(F.B0A)
    assert {d.key.split(".")[-1] for d in drifts} == {"GATE_THRESHOLD", "CONTROL_BITE_TOLERANCE"}
    assert all(d.frozen != d.current for d in drifts)


def test_an_unfrozen_point_verifies_as_absent_rather_than_clean(isolated):
    record, drifts = F.verify(F.B0A)
    assert record is None and drifts == []


# --- the sign-off sheet ------------------------------------------------------


def test_the_sheet_groups_by_definition_and_states_the_contingencies(isolated):
    text = F.sheet(F.B0A, data_values={"gate_resolution": "1024 px"})
    for definition in ("## P1", "## P3", "## P7"):
        assert definition in text
    assert "CONTROL_BITE_TOLERANCE" in text
    assert "check 1a resolution raise" in text, "the declared contingency must be stated"
    assert "gate_resolution" in text, "data-dependent values appear too"
    assert "forks the document" in text


def test_the_sheet_flags_a_dirty_tree(isolated, monkeypatch):
    monkeypatch.setattr(F, "working_tree_clean", lambda: False)
    assert "**dirty**" in F.sheet(F.B0A)
