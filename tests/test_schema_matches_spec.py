"""SPEC v1.2 Section 7: "A test in CI diffs the enum lists in this file against
``schema.py``."  This is that test.

It parses the markdown tables of SPEC Sections 3 and 4, pulls the backticked
tokens out of the Meaning cell of each enum column, and requires set equality
with the tuples in :mod:`shared.harness.schema`.  A value added to the prose
without a code change (or the reverse) fails here.

Column names are looked up inside their own section: ``abstain_protocol``, for
one, is both a model-config field in Section 1 and an outputs column in Section
4, and only the Section 4 row is the enum.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from shared.harness import schema

SPEC_PATH = Path(__file__).resolve().parents[1] / "shared" / "harness" / "SPEC.md"

# column name -> (which SPEC section defines it, the tuple that must match)
ENUM_COLUMNS = {
    "set": ("items", schema.SETS),
    "split_half": ("items", schema.SPLIT_HALVES),
    "size_bin": ("items", schema.SIZE_BINS),
    "dev_slice": ("items", schema.DEV_SLICES),
    "condition": ("outputs", schema.CONDITIONS),
    "abstain_protocol": ("outputs", schema.ABSTAIN_PROTOCOLS),
    "output_type": ("outputs", schema.OUTPUT_TYPES),
}


def spec_text() -> str:
    assert SPEC_PATH.exists(), f"SPEC not found at {SPEC_PATH}"
    return SPEC_PATH.read_text(encoding="utf-8")


def section(kind: str, text: str | None = None) -> str:
    """The body of SPEC Section 3 (items) or Section 4 (outputs)."""
    text = text if text is not None else spec_text()
    head, tail = {
        "items": ("## 3. `items.parquet`", "## 4."),
        "outputs": ("## 4. `outputs.parquet`", "## 5."),
    }[kind]
    assert head in text, f"SPEC has no heading {head!r}"
    return text.split(head, 1)[1].split(tail, 1)[0]


def meaning_cell(column: str, kind: str) -> str:
    """The last cell of the table row whose first cell is `` `column` ``."""
    body = section(kind)
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in body.splitlines()
        if line.strip().startswith("|")
    ]
    hits = [r for r in rows if r and r[0] == f"`{column}`"]
    assert len(hits) == 1, f"expected one `{column}` row in SPEC {kind}, found {len(hits)}"
    return hits[0][-1]


@pytest.mark.parametrize("column", sorted(ENUM_COLUMNS))
def test_enum_matches_spec(column: str) -> None:
    kind, values = ENUM_COLUMNS[column]
    in_spec = set(re.findall(r"`([^`]+)`", meaning_cell(column, kind)))
    assert in_spec == set(values), (
        f"`{column}` drifted: SPEC has {sorted(in_spec)}, schema.py has {sorted(values)}"
    )


def test_spec_version_matches() -> None:
    assert f"SPEC v{schema.SPEC_VERSION}" in spec_text()


@pytest.mark.parametrize("kind", ["items", "outputs"])
def test_every_spec_column_is_in_the_pyarrow_schema(kind: str) -> None:
    body = section(kind)
    # first cell of each row, which is `col` or `col_a`, `col_b` for the paired row
    names = set()
    for line in body.splitlines():
        if not line.strip().startswith("|"):
            continue
        first = line.strip().strip("|").split("|")[0].strip()
        names.update(re.findall(r"`([a-z0-9_]+)`", first))
    missing = names - set(schema.SCHEMAS[kind].names)
    assert not missing, f"{kind}: SPEC columns absent from schema.py: {sorted(missing)}"
    extra = set(schema.SCHEMAS[kind].names) - names
    assert not extra, f"{kind}: schema.py columns absent from SPEC: {sorted(extra)}"


def test_original_only_columns_are_named_as_such_in_spec() -> None:
    body = section("outputs")
    for column in schema.ORIGINAL_ONLY_COLUMNS:
        assert "ORIGINAL" in meaning_cell(column, "outputs"), (
            f"`{column}` is enforced as ORIGINAL-only in schema.py but SPEC does not say so"
        )
