"""Executable form of ``shared/harness/SPEC.md`` Sections 3 and 4.

SPEC v1.2 Section 7 makes this file normative alongside the prose: every writer
calls :func:`validate` before closing a parquet file and every reader calls it on
open.  ``tests/test_schema_matches_spec.py`` diffs the enum lists below against
the markdown table, so the two cannot drift.

The idea-specific stores (``edits/index.parquet``, ``edits/verifier.parquet``,
``results/<run_id>/relations.parquet``) live in ``idea91/schemas.py``; SPEC only
governs ``items.parquet`` and ``outputs.parquet``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SPEC_VERSION = "1.2"

# --- Section 3 and 4 enums (diffed against SPEC.md in CI) ---------------------

SETS = (
    "groundingme",
    "openref",
    "ref_l4",
    "grefcoco",
    "cocosearch18",
    "openimages_pool",
    "sa1b_pool",
    "synthetic",
)
SPLIT_HALVES = ("coco", "objects365")
SIZE_BINS = ("tiny", "small", "medium", "large", "xl")
DEV_SLICES = ("a", "b", "p12")
CONDITIONS = (
    "ORIGINAL",
    "REMOVE",
    "CONTROL_OBJ",
    "CONTROL_BG",
    "T_NULL",
    "T_HEAD",
    "T_ATTR",
)
ABSTAIN_PROTOCOLS = ("primary", "secondary")
OUTPUT_TYPES = ("box", "point", "set", "none", "invalid", "refusal")

#: ``iou_gt``, ``point_in_gt`` and ``is_failure`` are defined for ORIGINAL rows
#: only (SPEC Section 4); relations for the other conditions are computed by the
#: idea-specific relation code, never stored here.
ORIGINAL_ONLY_COLUMNS = ("iou_gt", "point_in_gt", "is_failure")

_BOX = pa.list_(pa.float64())

ITEMS_SCHEMA = pa.schema(
    [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("set", pa.string(), nullable=False),
        pa.field("split_half", pa.string(), nullable=True),
        pa.field("image_path", pa.string(), nullable=False),
        pa.field("image_sha256", pa.string(), nullable=False),
        pa.field("image_phash", pa.string(), nullable=False),
        pa.field("expr", pa.string(), nullable=False),
        pa.field("gt_boxes_xyxy_px", pa.list_(_BOX), nullable=False),
        pa.field("n_gt", pa.int32(), nullable=False),
        pa.field("size_bin", pa.string(), nullable=True),
        pa.field("dimension", pa.string(), nullable=True),
        pa.field("in_kill_set", pa.bool_(), nullable=False),
        pa.field("dev_slice", pa.string(), nullable=True),
        pa.field("pair_id", pa.string(), nullable=True),
        pa.field("ceiling_iou_d4", pa.float64(), nullable=True),
        pa.field("downscale_factor_d4", pa.float64(), nullable=True),
    ]
)

OUTPUTS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("condition", pa.string(), nullable=False),
        pa.field("pair_id", pa.string(), nullable=True),
        pa.field("abstain_protocol", pa.string(), nullable=False),
        pa.field("prompt_version", pa.string(), nullable=False),
        pa.field("raw_text", pa.string(), nullable=False),
        pa.field("parse_ok", pa.bool_(), nullable=False),
        pa.field("output_type", pa.string(), nullable=False),
        pa.field("n_boxes", pa.int32(), nullable=False),
        pa.field("box_xyxy_px", _BOX, nullable=True),
        pa.field("point_xy_px", _BOX, nullable=True),
        pa.field("iou_gt", pa.float64(), nullable=True),
        pa.field("point_in_gt", pa.bool_(), nullable=True),
        pa.field("is_failure", pa.bool_(), nullable=True),
        pa.field("image_px_sent", pa.list_(pa.int32()), nullable=False),
        pa.field("latency_ms", pa.int64(), nullable=False),
        pa.field("tokens_in", pa.int64(), nullable=True),
        pa.field("tokens_out", pa.int64(), nullable=True),
    ]
)

SCHEMAS = {"items": ITEMS_SCHEMA, "outputs": OUTPUTS_SCHEMA}

_ENUM_COLUMNS = {
    "items": {
        "set": SETS,
        "split_half": SPLIT_HALVES,
        "size_bin": SIZE_BINS,
        "dev_slice": DEV_SLICES,
    },
    "outputs": {
        "condition": CONDITIONS,
        "abstain_protocol": ABSTAIN_PROTOCOLS,
        "output_type": OUTPUT_TYPES,
    },
}


class SchemaError(ValueError):
    """Raised when a table violates SPEC Section 3 or 4."""


@dataclass(frozen=True)
class _Problem:
    column: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.column}: {self.detail}"


def kind_of(path: str | Path) -> str:
    """``items`` or ``outputs``, from the file name."""
    stem = Path(path).stem
    if stem not in SCHEMAS:
        raise SchemaError(
            f"{path}: file name must be items.parquet or outputs.parquet, got {stem!r}"
        )
    return stem


def _check_enum(table: pa.Table, column: str, allowed: tuple[str, ...]) -> list[_Problem]:
    if column not in table.column_names:
        return []
    seen = set(table.column(column).drop_null().unique().to_pylist())
    bad = sorted(seen - set(allowed))
    if bad:
        return [_Problem(column, f"values outside SPEC enum {list(allowed)}: {bad}")]
    return []


def _check_fields(table: pa.Table, schema: pa.Schema) -> list[_Problem]:
    problems: list[_Problem] = []
    have = dict(zip(table.schema.names, table.schema.types))
    for field in schema:
        if field.name not in have:
            problems.append(_Problem(field.name, "column missing"))
            continue
        got = have[field.name]
        if not got.equals(field.type):
            problems.append(_Problem(field.name, f"type {got} != SPEC type {field.type}"))
        if not field.nullable and table.column(field.name).null_count:
            problems.append(_Problem(field.name, "nulls in a non-nullable SPEC column"))
    extra = sorted(set(have) - set(schema.names))
    if extra:
        problems.append(_Problem("<table>", f"columns outside SPEC: {extra}"))
    return problems


def _check_items_semantics(table: pa.Table) -> list[_Problem]:
    problems: list[_Problem] = []
    n_gt = table.column("n_gt").to_pylist()
    boxes = table.column("gt_boxes_xyxy_px").to_pylist()
    for i, (n, bs) in enumerate(zip(n_gt, boxes)):
        bs = bs or []
        if n != len(bs):
            problems.append(_Problem("n_gt", f"row {i}: n_gt={n} but {len(bs)} boxes"))
            break
        if any(b is None or len(b) != 4 for b in bs):
            problems.append(
                _Problem("gt_boxes_xyxy_px", f"row {i}: a box is not 4 floats xyxy")
            )
            break
    for i, h in enumerate(table.column("image_phash").to_pylist()):
        if h is None or len(h) != 16 or any(c not in "0123456789abcdef" for c in h.lower()):
            problems.append(
                _Problem("image_phash", f"row {i}: not a 16-char hex 64-bit pHash: {h!r}")
            )
            break
    for i, h in enumerate(table.column("image_sha256").to_pylist()):
        if h is None or len(h) != 64:
            problems.append(_Problem("image_sha256", f"row {i}: not a 64-char sha256"))
            break
    return problems


def _check_outputs_semantics(table: pa.Table) -> list[_Problem]:
    problems: list[_Problem] = []

    for i, px in enumerate(table.column("image_px_sent").to_pylist()):
        # "mandatory on every row" (SPEC Sections 1 and 4).
        if px is None or len(px) != 2 or any(v is None or v <= 0 for v in px):
            problems.append(
                _Problem("image_px_sent", f"row {i}: must be [width, height] > 0, got {px!r}")
            )
            break

    conditions = table.column("condition").to_pylist()
    for column in ORIGINAL_ONLY_COLUMNS:
        values = table.column(column).to_pylist()
        for i, (cond, value) in enumerate(zip(conditions, values)):
            if cond != "ORIGINAL" and value is not None:
                problems.append(
                    _Problem(
                        column,
                        f"row {i}: defined for ORIGINAL rows only (SPEC Section 4), "
                        f"but condition={cond} carries {value!r}",
                    )
                )
                break

    for i, box in enumerate(table.column("box_xyxy_px").to_pylist()):
        if box is not None and len(box) != 4:
            problems.append(_Problem("box_xyxy_px", f"row {i}: not 4 floats xyxy"))
            break
    for i, point in enumerate(table.column("point_xy_px").to_pylist()):
        if point is not None and len(point) != 2:
            problems.append(_Problem("point_xy_px", f"row {i}: not 2 floats xy"))
            break

    types = table.column("output_type").to_pylist()
    boxes = table.column("box_xyxy_px").to_pylist()
    points = table.column("point_xy_px").to_pylist()
    for i, (t, b, p) in enumerate(zip(types, boxes, points)):
        if t == "box" and b is None:
            problems.append(_Problem("box_xyxy_px", f"row {i}: output_type=box with no box"))
            break
        if t == "point" and p is None:
            problems.append(_Problem("point_xy_px", f"row {i}: output_type=point with no point"))
            break
    return problems


def validate(obj: str | Path | pa.Table, kind: str | None = None) -> pa.Table:
    """Validate a table against SPEC Sections 3 and 4.

    ``obj`` is a parquet path (``kind`` inferred from the file name) or an
    in-memory table (``kind`` required).  Returns the table so callers can write
    ``pq.write_table(validate(t, "outputs"), path)``.
    """
    if isinstance(obj, (str, Path)):
        kind = kind or kind_of(obj)
        table = pq.read_table(obj)
        where = str(obj)
    else:
        if kind is None:
            raise SchemaError("validate(table) needs kind='items' or 'outputs'")
        table = obj
        where = f"<in-memory {kind}>"
    if kind not in SCHEMAS:
        raise SchemaError(f"unknown kind {kind!r}")

    problems = _check_fields(table, SCHEMAS[kind])
    if not problems:  # semantics only make sense once the columns are right
        for column, allowed in _ENUM_COLUMNS[kind].items():
            problems += _check_enum(table, column, allowed)
        problems += (
            _check_items_semantics(table)
            if kind == "items"
            else _check_outputs_semantics(table)
        )
    if problems:
        joined = "\n  ".join(str(p) for p in problems)
        raise SchemaError(f"{where} violates SPEC v{SPEC_VERSION} {kind}:\n  {joined}")
    return table


def write_table(rows, path: str | Path, kind: str | None = None) -> Path:
    """Validate then write.  ``rows`` is a pyarrow Table or a list of dicts."""
    path = Path(path)
    kind = kind or kind_of(path)
    table = rows if isinstance(rows, pa.Table) else pa.Table.from_pylist(list(rows), schema=SCHEMAS[kind])
    validate(table, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def read_table(path: str | Path) -> pa.Table:
    """Read and validate (SPEC Section 7: every reader validates on open)."""
    return validate(path)


def empty(kind: str) -> pa.Table:
    return SCHEMAS[kind].empty_table()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def phash_hamming(a: str, b: str) -> int:
    """Hamming distance between two 64-bit hex pHashes (SPEC Section 3)."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def assert_dev_slice_disjoint(items: pa.Table, max_hamming: int = 4) -> None:
    """SPEC Section 3: dev slices are disjoint from kill sets on ``image_phash``.

    Asserted at build time, not on every read: this is O(n_dev * n_kill).
    """
    dev = [
        (i, h)
        for i, (s, h) in enumerate(
            zip(items.column("dev_slice").to_pylist(), items.column("image_phash").to_pylist())
        )
        if s is not None
    ]
    kill = [
        (i, h)
        for i, (k, h) in enumerate(
            zip(items.column("in_kill_set").to_pylist(), items.column("image_phash").to_pylist())
        )
        if k
    ]
    ids = items.column("item_id").to_pylist()
    for di, dh in dev:
        for ki, kh in kill:
            if phash_hamming(dh, kh) <= max_hamming:
                raise SchemaError(
                    f"dev-slice item {ids[di]} is within pHash Hamming {max_hamming} of "
                    f"kill-set item {ids[ki]} (SPEC Section 3 disjointness)"
                )
