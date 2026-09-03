"""The data registry, the licence file, the item builder and the download plan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.data import download, items, licenses
from shared.data.sources import BY_KEY, MANUAL, PIN, SOURCES
from shared.harness import schema as S


def write_png(path: Path, seed: int, size=(64, 48)) -> Path:
    import cv2

    from .conftest import textured_image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = textured_image((size[1], size[0]), seed=seed)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


# --- registry ----------------------------------------------------------------


def test_every_design_4_2_source_has_a_row():
    for key in ("openimages", "sa1b", "groundingme", "openref", "ref_l4", "grefcoco",
                "cocosearch18", "coco_train2014"):
        assert key in BY_KEY, f"design 4.2 lists {key} but the registry has no row"


def test_only_openimages_may_have_its_derivatives_released():
    """P1: CC-BY images make the K1 bank the only releasable edited artefact."""
    releasable = [s.key for s in SOURCES if s.may_release_derivatives]
    assert releasable == ["openimages"]


def test_research_only_sources_carry_no_release_permission():
    for key in ("sa1b", "groundingme", "cocosearch18"):
        assert "release_derivatives" not in BY_KEY[key].allowed
        assert "release_images" not in BY_KEY[key].allowed


def test_openref_licence_is_marked_unread():
    """Design O1 is open; the registry must not pretend otherwise."""
    assert BY_KEY["openref"].licence == PIN
    assert BY_KEY["openref"].allowed == ()


# --- LICENSES.md -------------------------------------------------------------


def test_the_licence_file_names_the_releasable_artefact_and_the_gaps():
    text = licenses.render()
    assert "The OpenImages K1 bank only" in text
    assert "SAM 3" in text and "design O7" in text
    assert "no retention of inputs" in text  # P17 provider terms
    assert "Licence text not yet read" in text
    for name in ("Open Images subset (K1 pool)", "COCO-Search18 (edit-free row)"):
        assert name in text


def test_the_licence_file_writes(tmp_path):
    path = licenses.write(tmp_path / "LICENSES.md")
    assert path.exists() and path.read_text(encoding="utf-8").startswith("# Data and model")


# --- items.parquet -----------------------------------------------------------


def test_a_built_items_table_validates_against_spec(tmp_path):
    raws = [
        items.RawItem(
            item_id=f"i{i}",
            image_path=write_png(tmp_path / "img" / f"{i}.png", seed=i),
            expr="the cat",
            gt_boxes_xyxy_px=[(10.0, 10.0, 60.0, 40.0)],
            dimension="spatial",
        )
        for i in range(4)
    ]
    table = items.build(raws, "groundingme", tmp_path / "items.parquet", data_root=tmp_path)
    S.validate(table, "items")
    assert table.num_rows == 4
    assert S.read_table(tmp_path / "items.parquet").num_rows == 4


def test_derived_columns_are_filled(tmp_path):
    raw = items.RawItem(
        item_id="i0",
        image_path=write_png(tmp_path / "a.png", seed=1),
        expr="the small cat",
        gt_boxes_xyxy_px=[(0.0, 0.0, 40.0, 20.0)],
    )
    row = items.to_row(raw, "groundingme", data_root=tmp_path)
    assert len(row["image_sha256"]) == 64
    assert len(row["image_phash"]) == 16
    assert row["size_bin"] == "small"  # longer side 40 px
    assert row["n_gt"] == 1
    assert row["image_path"] == "a.png"  # relative to the data root


def test_a_no_target_item_has_no_size_bin(tmp_path):
    raw = items.RawItem(
        item_id="i0", image_path=write_png(tmp_path / "b.png", seed=2), expr="the unicorn"
    )
    row = items.to_row(raw, "grefcoco", data_root=tmp_path)
    assert row["n_gt"] == 0 and row["size_bin"] is None


def test_assign_dev_slice_takes_items_out_of_the_kill_set(tmp_path):
    raws = [
        items.RawItem(item_id=f"i{i}", image_path=write_png(tmp_path / f"{i}.png", seed=i),
                      expr="x")
        for i in range(10)
    ]
    out = items.assign_dev_slice(raws, "p12", 3, seed=0)
    dev = [r for r in out if r.dev_slice == "p12"]
    assert len(dev) == 3
    assert all(not r.in_kill_set for r in dev)
    assert all(r.in_kill_set for r in out if r.dev_slice is None)


def test_building_asserts_dev_slice_disjointness(tmp_path):
    """The same image on both sides of the split must fail at build time."""
    shared_image = write_png(tmp_path / "same.png", seed=7)
    raws = [
        items.RawItem(item_id="dev", image_path=shared_image, expr="x",
                      dev_slice="a", in_kill_set=False),
        items.RawItem(item_id="kill", image_path=shared_image, expr="x", in_kill_set=True),
    ]
    with pytest.raises(S.SchemaError, match="disjointness"):
        items.build(raws, "ref_l4", data_root=tmp_path)


def test_adapters_say_what_they_are_waiting_for():
    with pytest.raises(NotImplementedError, match="annotation file layout"):
        items.groundingme_items(Path("/nowhere"))
    with pytest.raises(NotImplementedError, match="design O1"):
        items.openref_items(Path("/nowhere"))
    with pytest.raises(NotImplementedError, match="letterbox"):
        items.cocosearch18_items(Path("/nowhere"))


# --- download plan -----------------------------------------------------------


def test_an_unpinned_source_is_blocked_not_guessed():
    step = download.plan_for(BY_KEY["openimages"])
    assert not step.ready and PIN in step.blocker


def test_a_manual_source_prints_instructions():
    assert BY_KEY["cocosearch18"].kind == MANUAL
    step = download.plan_for(BY_KEY["cocosearch18"])
    assert not step.ready and "manual download" in step.blocker


def test_a_pinned_http_source_produces_a_command():
    step = download.plan_for(BY_KEY["coco_train2014"])
    assert step.ready
    assert step.command[0] == "curl" and step.command[-1].endswith("train2014.zip")


def test_the_plan_renders_sizes_and_fetches_nothing():
    text = download.render(download.build_plan(list(SOURCES)))
    assert "Nothing was fetched" in text
    assert "fetchable now" in text and "blocked" in text
    assert "licence:" in text
