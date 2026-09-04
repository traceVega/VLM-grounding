"""The GroundingME adapter: what P8 admits, and what it deliberately does not."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.data import groundingme as G


def row(item_id="gme_00001", bbox=(10, 10, 110, 110), subtask="Limited", w=1000, h=1000):
    return G.GmeRow(
        item_id=item_id,
        description="The object is a red cup on the left of the table.",
        bbox=list(bbox) if bbox is not None else None,
        detection_type="cup",
        subtask_l1=subtask,
        subtask_l2="Appearance",
        width=w,
        height=h,
    )


# --- reading the dataset's own conventions -----------------------------------


def test_a_null_bbox_is_a_rejection_item_not_a_broken_one():
    """201 of the 1,005 items have a null bbox; that is the Rejection subtask."""
    assert row(bbox=None, subtask=G.REJECTION_SUBTASK).is_positive is False
    assert row().is_positive is True


def test_the_box_is_read_as_xyxy():
    """Confirmed on the real set: 100% of positives are consistent with xyxy."""
    r = row(bbox=(100, 200, 300, 500), w=1000, h=1000)
    assert r.area_frac == pytest.approx((300 - 100) * (500 - 200) / 1_000_000)


def test_a_rejection_item_has_no_area():
    assert row(bbox=None).area_frac == 0.0


# --- P8's scope --------------------------------------------------------------


def test_rejection_items_are_carried_but_kept_out_of_the_kill_set():
    """P8 scopes K2 to positive single-box items, so these must not be K2 pairs --
    but dropping 201 labelled negatives entirely would lose P12's natural
    companion set."""
    items = G.to_raw_items([row(), row("gme_00002", bbox=None, subtask=G.REJECTION_SUBTASK)])
    positive, rejection = items
    assert positive.in_kill_set and positive.gt_boxes_xyxy_px
    assert not rejection.in_kill_set
    assert rejection.gt_boxes_xyxy_px == [], "no box means no box, not a zero box"


def test_a_referent_over_the_area_limit_leaves_the_kill_set():
    """P8: referent area at most 30% of the image."""
    big = row(bbox=(0, 0, 900, 900), w=1000, h=1000)  # 81%
    assert big.area_frac > G.MAX_AREA_FRAC
    assert G.to_raw_items([big])[0].in_kill_set is False


def test_a_referent_just_inside_the_limit_stays():
    ok = row(bbox=(0, 0, 500, 500), w=1000, h=1000)  # 25%
    assert G.to_raw_items([ok])[0].in_kill_set is True


def test_the_dimension_is_subtask_l1_because_that_is_what_p15_reports_on():
    item = G.to_raw_items([row(subtask="Spatial")])[0]
    assert item.dimension == "Spatial"


def test_the_expression_is_the_description_verbatim():
    """P13's T_NULL replaces this text, so it must not be trimmed on the way in."""
    r = row()
    assert G.to_raw_items([r])[0].expr == r.description


# --- the sidecar -------------------------------------------------------------


def test_the_sidecar_keeps_what_items_parquet_has_no_column_for(tmp_path):
    rows = [row(subtask="Limited"), row("gme_00002", subtask="Rejection", bbox=None)]
    path = G.write_dimension_sidecar(rows, tmp_path / "dimensions.csv")
    text = path.read_text(encoding="utf-8")
    assert "subtask_l2" in text and "head_noun" in text
    assert "gme_00002" in text and "Rejection" in text


# --- failure modes -----------------------------------------------------------


def test_a_missing_snapshot_says_how_to_fetch_it():
    with pytest.raises(SystemExit, match="hf download"):
        G.snapshot_dir("0" * 40)


def test_item_ids_are_stable_and_sortable():
    assert G.item_id_for(1) == "gme_00001"
    assert G.item_id_for(1005) == "gme_01005"
    assert sorted([G.item_id_for(2), G.item_id_for(10)]) == ["gme_00002", "gme_00010"]


def test_the_image_path_follows_the_item_id():
    assert G.image_path_for("gme_00007").name == "gme_00007.jpg"
    assert isinstance(G.image_path_for("gme_00007"), Path)
