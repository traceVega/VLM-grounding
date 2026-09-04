"""SPEC Section 2 (manifests, immutability, the non-kill guard), the prompt and
model-config pinning rules, and the judge lineage rule."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from shared.harness import manifest as MF
from shared.harness import model_config as MC
from shared.harness import prompts as PR
from shared.harness import schema as S
from shared.judges import lineage


def outputs_row(**over):
    row = {
        "run_id": "r1",
        "model_id": "qwen3vl-8b-instruct",
        "item_id": "i1",
        "condition": "ORIGINAL",
        "pair_id": "p1",
        "abstain_protocol": "primary",
        "prompt_version": "abc",
        "raw_text": "(10,10),(20,20)",
        "parse_ok": True,
        "output_type": "box",
        "n_boxes": 1,
        "box_xyxy_px": [10.0, 10.0, 20.0, 20.0],
        "point_xy_px": None,
        "iou_gt": 0.8,
        "point_in_gt": None,
        "is_failure": False,
        "image_px_sent": [1024, 768],
        "latency_ms": 120,
        "tokens_in": 900,
        "tokens_out": 12,
    }
    row.update(over)
    return row


def table(rows, kind="outputs"):
    return pa.Table.from_pylist(rows, schema=S.SCHEMAS[kind])


# --- SPEC Section 4 semantics ------------------------------------------------


def test_a_valid_outputs_row_passes():
    S.validate(table([outputs_row()]), "outputs")


def test_image_px_sent_is_mandatory_on_every_row():
    with pytest.raises(S.SchemaError, match="image_px_sent"):
        S.validate(table([outputs_row(image_px_sent=[0, 768])]), "outputs")


def test_iou_gt_may_not_appear_on_a_non_original_row():
    bad = outputs_row(condition="REMOVE", iou_gt=0.4, is_failure=None, point_in_gt=None)
    with pytest.raises(S.SchemaError, match="ORIGINAL rows only"):
        S.validate(table([bad]), "outputs")


def test_a_condition_outside_the_spec_enum_is_refused():
    with pytest.raises(S.SchemaError, match="condition"):
        S.validate(table([outputs_row(condition="T_COLOUR")]), "outputs")


def test_output_type_box_without_a_box_is_refused():
    with pytest.raises(S.SchemaError, match="output_type=box"):
        S.validate(table([outputs_row(box_xyxy_px=None)]), "outputs")


def test_round_trip_through_disk_validates_on_read(tmp_path):
    path = S.write_table([outputs_row()], tmp_path / "outputs.parquet")
    assert S.read_table(path).num_rows == 1


def test_a_file_named_something_else_is_refused(tmp_path):
    with pytest.raises(S.SchemaError, match="items.parquet or outputs.parquet"):
        S.kind_of(tmp_path / "results.parquet")


def test_dev_slices_must_be_disjoint_from_kill_sets_on_phash():
    def item(item_id, phash, dev, kill):
        return {
            "item_id": item_id, "set": "ref_l4", "split_half": None,
            "image_path": "a.jpg", "image_sha256": "0" * 64, "image_phash": phash,
            "expr": "the cat", "gt_boxes_xyxy_px": [[1.0, 1.0, 2.0, 2.0]], "n_gt": 1,
            "size_bin": "small", "dimension": None, "in_kill_set": kill,
            "dev_slice": dev, "pair_id": None, "ceiling_iou_d4": None,
            "downscale_factor_d4": None,
        }
    ok = table([item("a", "0f0f0f0f0f0f0f0f", "a", False), item("b", "f0f0f0f0f0f0f0f0", None, True)], "items")
    S.assert_dev_slice_disjoint(ok)

    near = table([item("a", "0f0f0f0f0f0f0f0f", "a", False), item("b", "0f0f0f0f0f0f0f0e", None, True)], "items")
    with pytest.raises(S.SchemaError, match="disjointness"):
        S.assert_dev_slice_disjoint(near)


# --- SPEC Section 2 ----------------------------------------------------------


def make_manifest(**over):
    kwargs = dict(
        run_id="r1", non_kill=False, prereg_version="k1-v1",
        model_id="qwen3vl-8b-instruct", model_revision="deadbeef", backend="hf",
    )
    kwargs.update(over)
    return MF.Manifest(**kwargs)


def test_manifest_hashes_every_file_and_is_written_last(tmp_path):
    run = tmp_path / "r1"
    S.write_table([outputs_row()], run / "outputs.parquet")
    path = MF.write(run, make_manifest())
    data = json.loads(path.read_text())
    assert "outputs.parquet" in data["file_sha256"]
    assert MF.MANIFEST_NAME not in data["file_sha256"]
    assert set(data) == set(MF.REQUIRED_FIELDS)


def test_a_result_directory_is_immutable_after_the_manifest(tmp_path):
    run = tmp_path / "r1"
    S.write_table([outputs_row()], run / "outputs.parquet")
    MF.write(run, make_manifest())
    with pytest.raises(MF.ManifestError, match="immutable"):
        MF.write(run, make_manifest())


def test_reading_detects_a_file_changed_after_the_manifest(tmp_path):
    run = tmp_path / "r1"
    S.write_table([outputs_row()], run / "outputs.parquet")
    MF.write(run, make_manifest())
    S.write_table([outputs_row(item_id="i2")], run / "outputs.parquet")
    with pytest.raises(MF.ManifestError, match="changed after the manifest"):
        MF.read(run)


def test_reading_detects_a_file_added_after_the_manifest(tmp_path):
    run = tmp_path / "r1"
    S.write_table([outputs_row()], run / "outputs.parquet")
    MF.write(run, make_manifest())
    (run / "notes.txt").write_text("later")
    with pytest.raises(MF.ManifestError, match="files added"):
        MF.read(run)


def test_a_non_kill_run_cannot_enter_a_kill_table(tmp_path):
    for name, non_kill in (("kill", False), ("smoke", True)):
        run = tmp_path / name
        S.write_table([outputs_row(run_id=name)], run / "outputs.parquet")
        MF.write(run, make_manifest(run_id=name, non_kill=non_kill))
    MF.require_kill_runs([tmp_path / "kill"])
    with pytest.raises(MF.ManifestError, match="non-kill run ids"):
        MF.require_kill_runs([tmp_path / "kill", tmp_path / "smoke"])


def test_package_versions_records_what_is_installed():
    versions = MF.package_versions(("numpy", "pyarrow"))
    assert versions["numpy"] and versions["python"]


# --- prompts -----------------------------------------------------------------


def test_a_verified_prompt_loads_and_renders():
    p = PR.load("verifier_headnoun")
    assert p.render(head_noun="cat") == "Is there a cat in this image? Answer with one word, yes or no."
    assert len(p.version) == 64


def test_an_unverified_prompt_is_refused_by_a_kill_run():
    # Molmo2's pointing instruction still needs its card read (Q-2).
    with pytest.raises(PR.PromptError, match="UNVERIFIED"):
        PR.load("pointing_molmo2_primary")
    assert PR.load("pointing_molmo2_primary", non_kill=True).text  # smoke runs may


def test_a_prompt_missing_a_field_says_which():
    with pytest.raises(PR.PromptError, match="head_noun"):
        PR.load("verifier_headnoun").render()


def test_a_wrapped_header_comment_never_reaches_the_model(tmp_path):
    """The header is ``# key: value``; a wrapped comment continues the value.

    Without continuations the second line of a wrapped comment fell through to
    the body and was sent to the model as part of a pre-registered, hashed
    prompt -- silently, because nothing downstream can tell prose from prompt.
    """
    path = tmp_path / "wrapped.txt"
    path.write_text(
        "# status: VERIFIED\n"
        "# source: a note long enough to wrap onto\n"
        "#   a second line mentioning {not_a_field}\n"
        "# fields: expr\n"
        "Find {expr}.\n",
        encoding="utf-8",
    )
    p = PR.load("wrapped", root=tmp_path)
    assert p.text == "Find {expr}."
    assert "second line" in p.meta["source"], "the continuation belongs to the header"
    assert p.render(expr="the cup") == "Find the cup."  # no stray field needed


def test_a_hash_covers_the_file_so_a_comment_edit_is_visible():
    """prompt_version is the sha256 of the whole file, header included (SPEC)."""
    p = PR.load("verifier_headnoun")
    assert len(p.version) == 64


def test_the_groundingme_instruction_is_the_benchmarks_own():
    """P12 pins the primary protocol to GroundingME's words, not a paraphrase."""
    rendered = PR.load("grounding_qwen3vl_primary").render(expr="THE_EXPR")
    assert rendered.startswith("All spatial relationships are defined from the viewer's")
    assert "THE_EXPR" in rendered
    # the benchmark's own rejection channel, which P16 (b) has to reckon with
    assert '{"bbox_2d": null}' in rendered
    assert '{"bbox_2d": [x1, y1, x2, y2]}' in rendered
    assert not rendered.startswith("#")


def test_the_secondary_protocol_is_the_primary_plus_one_sentence():
    """P12: 'the same instruction followed by ...' -- so it must be a prefix."""
    primary = PR.load("grounding_qwen3vl_primary").render(expr="X")
    secondary = PR.load("grounding_qwen3vl_secondary").render(expr="X")
    assert secondary.startswith(primary)
    assert secondary[len(primary):].strip() == "If the object is not present, output none."


# --- model configs -----------------------------------------------------------


def test_an_unpinned_model_config_is_refused_by_a_kill_run():
    with pytest.raises(MC.ModelConfigError, match="unpinned"):
        MC.load("qwen3vl-8b-instruct")
    cfg = MC.load("qwen3vl-8b-instruct", non_kill=True)
    assert "revision" in cfg.unpinned
    assert "abstain_protocol.primary.null_box" in cfg.unpinned


def test_the_config_hash_is_stable_and_the_fields_are_spec_fields():
    cfg = MC.load("molmo2-8b", non_kill=True)
    assert cfg.config_hash == MC.load("molmo2-8b", non_kill=True).config_hash
    assert set(MC.REQUIRED_FIELDS) <= set(cfg.data)
    assert cfg.backend_default == "vllm"  # P19
    assert cfg.resolution_policy["kind"] == "crops"


def test_both_models_are_registered():
    assert set(MC.list_models()) == {"qwen3vl-8b-instruct", "molmo2-8b"}


# --- judge lineage -----------------------------------------------------------


def test_the_edit_verifier_is_not_the_reward_judge():
    verifier = lineage.for_role("edit_verifier")
    reward = lineage.for_role("reward_judge")
    assert verifier.lineage_family != reward.lineage_family
    assert verifier.name == "gemma4-12b" and reward.name == "qwen3.5-9b"


def test_a_lineage_violation_is_refused():
    judges = {
        "a": lineage.Judge("a", "j", "p", "r", "same", ("edit_verifier",), 1),
        "b": lineage.Judge("b", "k", "q", "r", "same", ("reward_judge",), 1),
    }
    with pytest.raises(lineage.LineageError, match="lineage rule violated"):
        lineage.check_constraints(
            judges, {"edit_verifier": "a", "reward_judge": "b"}, [["edit_verifier", "reward_judge"]]
        )


def test_serving_args_carry_a_measured_memory_fraction():
    args = lineage.serving_args("edit_verifier")
    assert 0.5 < args["gpu_memory_utilization"] <= 0.95
    assert args["max_model_len"] == 4096
