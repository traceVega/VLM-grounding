"""Sample-set construction for the gate and for acceptance check 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from idea91 import masks as M
from idea91 import schemas
from idea91.gate import build as B
from idea91.gate import ladders as L

from .conftest import disc


def index_row(image_id: str, operator: str, hole_type: str = "mask", area: float = 0.012,
              control_source: str | None = "labelled_other_class") -> dict:
    mask = disc((200, 300), 150, 100, 20)
    return {
        "image_id": image_id,
        "set_or_pool": "openimages_pool",
        "instance_id": "ref",
        "pair_id": None,
        "operator": operator,
        "editor": "big_lama",
        "editor_weights_sha256": "a" * 64,
        "editor_settings_hash": "b" * 16,
        "mask_rle": M.encode_rle(mask),
        "hole_type": hole_type,
        "mask_area_frac": area,
        "box_to_mask_iou": None,
        "box_inpaint_flag": False,
        "control_instance_id": None if operator.endswith("REMOVE") else "c1",
        "control_source": None if operator.endswith("REMOVE") else control_source,
        "control_area_ratio": None,
        "control_centrality_delta": None,
        "window_xyxy_px": [0, 0, 300, 200],
        "window_path": f"openimages_pool/{image_id}/ref/{operator}.png",
        "window_sha256": "c" * 64,
        "created_at": "2026-09-03T00:00:00+00:00",
        "freeze_version": "test",
    }


def make_index(images: int = 4, operators=("REMOVE", "CONTROL_OBJ", "CONTROL_OBJ_2", "CONTROL_BG"),
               hole_type: str = "mask") -> pa.Table:
    prefix = "" if hole_type == "mask" else "RECT_"
    rows = [
        index_row(f"img{i}", f"{prefix}{op}", hole_type,
                  control_source="background" if op.endswith("CONTROL_BG") else "labelled_other_class")
        for i in range(images)
        for op in operators
    ]
    table = pa.Table.from_pylist(rows, schema=schemas.INDEX_SCHEMA)
    schemas.validate(table, "index")
    return table


IMAGE_DIR = Path("/images")


def test_the_gate_contrast_is_one_positive_and_one_negative_per_image():
    samples = B.contrast_samples(make_index(4), IMAGE_DIR)
    assert len(samples) == 8
    assert sum(s.label for s in samples) == 4  # balanced
    per_image = {}
    for s in samples:
        per_image.setdefault(s.image_id, []).append(s.label)
    assert all(sorted(v) == [0, 1] for v in per_image.values())


def test_an_image_missing_one_side_does_not_contribute():
    """A half pair would unbalance the per-image AUROC."""
    index = make_index(3, operators=("REMOVE", "CONTROL_OBJ"))
    partial = pa.Table.from_pylist(
        index.to_pylist() + [index_row("lonely", "REMOVE")], schema=schemas.INDEX_SCHEMA
    )
    samples = B.contrast_samples(partial, IMAGE_DIR)
    assert "lonely" not in {s.image_id for s in samples}
    assert len(samples) == 6


def test_the_control_bg_comparison_uses_the_background_edit():
    samples = B.contrast_samples(make_index(3), IMAGE_DIR, negative="CONTROL_BG")
    negatives = [s for s in samples if s.label == 0]
    assert negatives and all(s.control_source == "background" for s in negatives)


def test_the_1b_null_labels_two_edits_from_the_same_distribution():
    samples = B.null_1b_samples(make_index(4), IMAGE_DIR)
    assert len(samples) == 8
    assert sum(s.label for s in samples) == 4
    assert all("REMOVE" not in s.sample_id for s in samples), "the null must not see a removal"


@pytest.mark.parametrize("arm,quality", [("global", 75), ("global", 92), ("local", 30)])
def test_ladders_damage_one_side_and_reference_the_other(arm, quality):
    samples = B.ladder_samples(make_index(3), IMAGE_DIR, arm=arm, quality=quality)
    damaged = [s for s in samples if s.label == 1]
    reference = [s for s in samples if s.label == 0]
    assert damaged and reference
    assert all(s.damage == (arm, quality) for s in damaged)
    assert all(s.damage == (arm, L.REFERENCE_QUALITY) for s in reference)


def test_every_ladder_runs_on_the_1b_pairs_never_on_a_removal():
    """Check 1a's whole point: no real class signal may contaminate it."""
    for arm, qualities in (("global", L.GLOBAL_LADDER_QUALITIES),
                           ("local", L.LOCAL_LADDER_QUALITIES)):
        for q in qualities:
            samples = B.ladder_samples(make_index(3), IMAGE_DIR, arm=arm, quality=q)
            assert all("REMOVE" not in s.sample_id for s in samples)


def test_a_quality_off_the_ladder_is_refused():
    with pytest.raises(ValueError, match="ladder quality"):
        B.ladder_samples(make_index(2), IMAGE_DIR, arm="global", quality=50)
    with pytest.raises(ValueError, match="arm must be"):
        B.ladder_samples(make_index(2), IMAGE_DIR, arm="sideways", quality=75)


def test_local_ladder_samples_carry_the_hole_they_need():
    samples = B.ladder_samples(make_index(2), IMAGE_DIR, arm="local", quality=30)
    assert all(s.hole_rle for s in samples)
    assert all(s.hole() is not None and s.hole().any() for s in samples)


def test_samples_are_binned_by_hole_area_for_the_local_floor():
    rows = [index_row("a", "CONTROL_OBJ", area=0.007), index_row("a", "CONTROL_OBJ_2", area=0.007),
            index_row("b", "CONTROL_OBJ", area=0.08), index_row("b", "CONTROL_OBJ_2", area=0.08)]
    index = pa.Table.from_pylist(rows, schema=schemas.INDEX_SCHEMA)
    samples = B.ladder_samples(index, IMAGE_DIR, arm="local", quality=50)
    bins = B.by_hole_area_bin(samples)
    assert set(bins) == {"0.5-1%", "5-15%"}
    assert all(len(v) == 2 for v in bins.values())


def test_rect_hole_types_select_the_rect_operators():
    index = make_index(3, hole_type="rect")
    samples = B.contrast_samples(index, IMAGE_DIR, hole_type="rect")
    assert samples and all("RECT_" in s.sample_id for s in samples)
    assert B.contrast_samples(index, IMAGE_DIR, hole_type="mask") == []


def test_undamaged_samples_compose_without_touching_jpeg(tmp_path):
    """The damage hook must be inert when no ladder is running."""
    from idea91.gate.dataset import GateSample

    import cv2

    from .conftest import textured_image

    original = textured_image((120, 160), seed=2)
    cv2.imwrite(str(tmp_path / "img.jpg"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    window = tmp_path / "edits" / "w.png"
    window.parent.mkdir(parents=True)
    cv2.imwrite(str(window), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))

    sample = GateSample(
        sample_id="s", image_id="img", label=1,
        original_path=tmp_path / "img.jpg", window_path=Path("w.png"),
        window_xyxy=(0, 0, 160, 120),
    )
    composed = sample.compose(tmp_path / "edits")
    assert composed.shape == original.shape

    damaged = GateSample(
        sample_id="s", image_id="img", label=1,
        original_path=tmp_path / "img.jpg", window_path=Path("w.png"),
        window_xyxy=(0, 0, 160, 120), damage=("global", 30),
    )
    assert not np.array_equal(damaged.compose(tmp_path / "edits"), composed)


def test_a_local_damage_without_a_hole_says_so(tmp_path):
    from idea91.gate.dataset import GateSample

    sample = GateSample(
        sample_id="s", image_id="img", label=1,
        original_path=tmp_path / "missing.jpg", window_path=Path("w.png"),
        window_xyxy=(0, 0, 4, 4), damage=("local", 30),
    )
    assert sample.hole() is None
