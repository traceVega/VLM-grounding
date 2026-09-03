"""End-to-end edit materialization: plan -> window PNG -> index row -> composed
full image, with the compositing assert holding on the *composed* images.

This runs the whole P2/P3/P4 path with OpenCV's Telea fill standing in for
big-LaMa, so the bank machinery is exercised before the LaMa weights land.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from idea91 import schemas
from idea91.edits import composite as C
from idea91.edits.build import hole_from_row, load_edited, materialize
from idea91.edits.inpaint import TeleaInpainter
from idea91.edits.sampler import plan_k1_image
from idea91.masks import bbox_xyxy

from .conftest import disc


@pytest.fixture
def built(tmp_path, scene):
    from idea91.edits.sampler import Instance

    shape = scene["shape"]
    instances = scene["instances"] + [
        Instance("ok2", disc(shape, 300, 150, 42), "labelled_other_class", "hat")
    ]
    plans = plan_k1_image(
        scene["referent"], instances, rng=np.random.default_rng(0)
    )
    inpainter = TeleaInpainter()
    edits = [
        materialize(
            scene["image"],
            p,
            inpainter,
            image_id="img0001",
            instance_id=scene["referent"].instance_id,
            set_or_pool="openimages_pool",
            edits_root=tmp_path,
            freeze_version="test",
        )
        for p in plans
    ]
    return {"plans": plans, "edits": edits, "root": tmp_path, "scene": scene}


def test_every_operator_is_written_and_indexed(built):
    ops = [e.index_row["operator"] for e in built["edits"]]
    assert ops == ["REMOVE", "CONTROL_OBJ", "CONTROL_OBJ_2", "CONTROL_BG"]
    for e in built["edits"]:
        path = built["root"] / e.index_row["window_path"]
        assert path.exists() and path.suffix == ".png"
        assert len(e.index_row["window_sha256"]) == 64
        assert e.index_row["editor"] == "opencv_telea"


def test_the_index_validates_against_the_design_section_5_schema(built):
    table = pa.Table.from_pylist(
        [e.index_row for e in built["edits"]], schema=schemas.INDEX_SCHEMA
    )
    schemas.validate(table, "index")


def test_index_rejects_an_operator_outside_the_design_list(built):
    row = dict(built["edits"][0].index_row)
    row["operator"] = "PAINT_A_CAT"
    table = pa.Table.from_pylist([row], schema=schemas.INDEX_SCHEMA)
    with pytest.raises(Exception, match="operator"):
        schemas.validate(table, "index")


def test_the_edit_actually_changed_the_hole(built):
    scene = built["scene"]
    for e in built["edits"]:
        full = load_edited(scene["image"], e.index_row, built["root"])
        hole = hole_from_row(e.index_row)
        assert np.any(full[hole] != scene["image"][hole]), e.index_row["operator"]


def test_composed_full_images_changed_nothing_outside_their_hole(built):
    scene = built["scene"]
    for e in built["edits"]:
        full = load_edited(scene["image"], e.index_row, built["root"])
        C.assert_in_mask(scene["image"], full, hole_from_row(e.index_row))


def test_check_1c_on_the_composed_remove_and_control_images(built):
    """Acceptance check 1c on real stored edits, not on arrays in memory."""
    scene = built["scene"]
    rows = {e.index_row["operator"]: e.index_row for e in built["edits"]}
    remove = load_edited(scene["image"], rows["REMOVE"], built["root"])
    control = load_edited(scene["image"], rows["CONTROL_OBJ"], built["root"])
    remove_hole = bbox_xyxy(hole_from_row(rows["REMOVE"]))
    control_hole = bbox_xyxy(hole_from_row(rows["CONTROL_OBJ"]))

    shared = C.assert_paired_crop_identical(
        remove, control, (20.0, 600.0, 220.0, 780.0), remove_hole, control_hole
    )
    assert shared.size > 0
    # and the two full images do differ, in exactly the two holes
    differing = np.any(remove != control, axis=-1)
    assert differing.any()
    union = hole_from_row(rows["REMOVE"]) | hole_from_row(rows["CONTROL_OBJ"])
    assert not np.any(differing & ~union)


def test_window_offset_round_trips_through_the_index_row(built):
    e = built["edits"][0]
    x0, y0, x1, y1 = e.index_row["window_xyxy_px"]
    assert (x1 - x0, y1 - y0) == e.window_image.shape[1::-1]
