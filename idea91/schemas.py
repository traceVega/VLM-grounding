"""The idea-91 stores of design Section 5.

SPEC v1.2 governs ``items.parquet`` and ``outputs.parquet``; these three are
idea-91's own and are defined here in the same executable style, with a
``validate`` per store so a half-written edit bank fails at write time rather
than in the K1 table.

* ``edits/index.parquet``   one row per materialized edit
* ``edits/verifier.parquet`` one row per judge question (P10)
* ``results/<run_id>/relations.parquet`` one row per pair and model (P14, P15)
* ``edits/human_labels.csv`` P11, a CSV because annotators' tooling exports CSV
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from shared.harness.schema import SchemaError

#: P3/P4 operators.  The ``RECT_`` variants are the rectangular hole type of P2;
#: ``CONTROL_OBJ_2`` is the second independent control edit acceptance check 1b
#: needs (one labelled each way, AUROC must be 0.5).
OPERATORS = (
    "REMOVE",
    "CONTROL_OBJ",
    "CONTROL_OBJ_2",
    "CONTROL_BG",
    "RECT_REMOVE",
    "RECT_CONTROL_OBJ",
    "RECT_CONTROL_OBJ_2",
    "RECT_CONTROL_BG",
)
HOLE_TYPES = ("mask", "rect")
CONTROL_SOURCES = ("labelled_other_class", "noun_phrase_instance", "class_agnostic", "background")

#: P10 questions.
VERIFIER_QUESTIONS = (
    "V1_headnoun_window",
    "V1_headnoun_tight_crop",
    "V1_headnoun_control_window",
    "V1_expr_full",
    "V1_classlabel_window",
    "V2",
)

#: P11 labels and strata.
HUMAN_LABELS = ("clean", "remnant", "failed")
HUMAN_STRATA = ("random", "v1_negative", "escalation", "pilot")

INDEX_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("set_or_pool", pa.string(), nullable=False),
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("pair_id", pa.string(), nullable=True),
        pa.field("operator", pa.string(), nullable=False),
        pa.field("editor", pa.string(), nullable=False),
        pa.field("editor_weights_sha256", pa.string(), nullable=False),
        pa.field("editor_settings_hash", pa.string(), nullable=False),
        pa.field("mask_rle", pa.string(), nullable=False),
        pa.field("hole_type", pa.string(), nullable=False),
        pa.field("mask_area_frac", pa.float64(), nullable=False),
        pa.field("box_to_mask_iou", pa.float64(), nullable=True),
        pa.field("box_inpaint_flag", pa.bool_(), nullable=False),
        pa.field("control_instance_id", pa.string(), nullable=True),
        pa.field("control_source", pa.string(), nullable=True),
        pa.field("control_area_ratio", pa.float64(), nullable=True),
        pa.field("control_centrality_delta", pa.float64(), nullable=True),
        pa.field("window_xyxy_px", pa.list_(pa.int32()), nullable=False),
        pa.field("window_path", pa.string(), nullable=False),
        pa.field("window_sha256", pa.string(), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
        pa.field("freeze_version", pa.string(), nullable=False),
    ]
)

VERIFIER_SCHEMA = pa.schema(
    [
        pa.field("window_sha256", pa.string(), nullable=False),
        pa.field("verifier_model", pa.string(), nullable=False),
        pa.field("verifier_commit", pa.string(), nullable=False),
        pa.field("prompt_version", pa.string(), nullable=False),
        pa.field("question", pa.string(), nullable=False),
        pa.field("answer_raw", pa.string(), nullable=False),
        pa.field("answer_bool", pa.bool_(), nullable=True),
        pa.field("parsed_ok", pa.bool_(), nullable=False),
        pa.field("logprob_yes", pa.float64(), nullable=True),
    ]
)

RELATIONS_SCHEMA = pa.schema(
    [
        pa.field("pair_id", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("abstain_protocol", pa.string(), nullable=False),
        pa.field("orig_correct", pa.bool_(), nullable=True),
        pa.field("remove_clean_verifier", pa.bool_(), nullable=True),
        pa.field("remove_clean_human", pa.bool_(), nullable=True),
        pa.field("control_obj_valid", pa.bool_(), nullable=True),
        pa.field("control_bg_valid", pa.bool_(), nullable=True),
        pa.field("none_on_remove", pa.bool_(), nullable=True),
        pa.field("box_on_remove", pa.bool_(), nullable=True),
        pa.field("same_box_50", pa.bool_(), nullable=True),
        pa.field("same_box_r30", pa.bool_(), nullable=True),
        pa.field("shift_on_control_obj", pa.bool_(), nullable=True),
        pa.field("shift_on_control_bg", pa.bool_(), nullable=True),
        pa.field("abstain_on_control_obj", pa.bool_(), nullable=True),
        pa.field("t_null_pass", pa.bool_(), nullable=True),
        pa.field("t_null_prior_pass", pa.bool_(), nullable=True),
        pa.field("t_head_pass", pa.bool_(), nullable=True),
        pa.field("t_attr_moved", pa.bool_(), nullable=True),
        pa.field("redundant", pa.bool_(), nullable=True),
        pa.field("unresolved", pa.bool_(), nullable=True),
        # carried for stratification in P15; not part of the design's field list
        pa.field("item_id", pa.string(), nullable=True),
        pa.field("set", pa.string(), nullable=True),
        pa.field("size_bin", pa.string(), nullable=True),
        pa.field("image_id", pa.string(), nullable=True),
        pa.field("undetermined", pa.bool_(), nullable=True),
    ]
)

#: Instance roles.  The first three are the ``control_source`` values of P3; the
#: last three are roles that belong in the P3 exclusion set but are never control
#: candidates and never reach ``index.control_source``.
INSTANCE_SOURCES = (
    "labelled_other_class",
    "noun_phrase_instance",
    "class_agnostic",
    "referent",
    "same_class_other_instance",
    "head_noun_neighbour",
)

#: Not in the design's Section 5 list; see notes/DEVIATIONS.md D-12.
INSTANCES_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("set_or_pool", pa.string(), nullable=False),
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=True),
        pa.field("mask_rle", pa.string(), nullable=False),
        pa.field("area_frac", pa.float64(), nullable=False),
        pa.field("bbox_xyxy_px", pa.list_(pa.float64()), nullable=False),
        pa.field("centre_xy_px", pa.list_(pa.float64()), nullable=False),
        pa.field("box_to_mask_iou", pa.float64(), nullable=True),
        pa.field("box_inpaint_flag", pa.bool_(), nullable=False),
        pa.field("segmenter", pa.string(), nullable=False),
        pa.field("segmenter_revision", pa.string(), nullable=False),
        pa.field("kill_grade", pa.bool_(), nullable=False),
        pa.field("head_noun", pa.string(), nullable=True),
        pa.field("n_head_noun_instances", pa.int32(), nullable=False),
    ]
)

SCHEMAS = {
    "index": INDEX_SCHEMA,
    "verifier": VERIFIER_SCHEMA,
    "relations": RELATIONS_SCHEMA,
    "instances": INSTANCES_SCHEMA,
}

_ENUMS = {
    "index": {
        "operator": OPERATORS,
        "hole_type": HOLE_TYPES,
        "control_source": CONTROL_SOURCES,
    },
    "verifier": {"question": VERIFIER_QUESTIONS},
    "relations": {"abstain_protocol": ("primary", "secondary")},
    "instances": {"source": INSTANCE_SOURCES},
}


def validate(obj, kind: str | None = None) -> pa.Table:
    if isinstance(obj, (str, Path)):
        kind = kind or Path(obj).stem
        table = pq.read_table(obj)
        where = str(obj)
    else:
        table, where = obj, f"<in-memory {kind}>"
    if kind not in SCHEMAS:
        raise SchemaError(f"unknown idea91 store {kind!r}; expected one of {sorted(SCHEMAS)}")
    schema = SCHEMAS[kind]
    problems: list[str] = []
    have = dict(zip(table.schema.names, table.schema.types))
    for f in schema:
        if f.name not in have:
            problems.append(f"{f.name}: column missing")
        elif not have[f.name].equals(f.type):
            problems.append(f"{f.name}: type {have[f.name]} != {f.type}")
        elif not f.nullable and table.column(f.name).null_count:
            problems.append(f"{f.name}: nulls in a non-nullable column")
    extra = sorted(set(have) - set(schema.names))
    if extra:
        problems.append(f"columns outside the design Section 5 list: {extra}")
    if not problems:
        for column, allowed in _ENUMS[kind].items():
            seen = set(table.column(column).drop_null().unique().to_pylist())
            bad = sorted(seen - set(allowed))
            if bad:
                problems.append(f"{column}: values outside {list(allowed)}: {bad}")
    if problems:
        raise SchemaError(f"{where} violates the idea91 {kind} schema:\n  " + "\n  ".join(problems))
    return table


def write(rows, path: str | Path, kind: str | None = None) -> Path:
    path = Path(path)
    kind = kind or path.stem
    table = rows if isinstance(rows, pa.Table) else pa.Table.from_pylist(list(rows), schema=SCHEMAS[kind])
    validate(table, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def read(path: str | Path) -> pa.Table:
    return validate(path)


def empty(kind: str) -> pa.Table:
    return SCHEMAS[kind].empty_table()
