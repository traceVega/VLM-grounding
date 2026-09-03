"""The K1 pool selection (design P1), on synthetic copies of the three CSVs.

The real metadata is 38 MB on the data volume; these tests build tiny CSVs with
the same columns so the P1 filters are checked without it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shared.data import openimages as OI

CC_BY = "https://creativecommons.org/licenses/by/2.0/"
CC_BY_NC = "https://creativecommons.org/licenses/by-nc/2.0/"


def write_csvs(root: Path, images: list[dict], boxes: list[dict]) -> Path:
    d = root / "openimages"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(images).to_csv(d / "validation-images-with-rotation.csv", index=False)
    pd.DataFrame(boxes).to_csv(d / "validation-annotations-bbox.csv", index=False)
    pd.DataFrame(
        [["/m/01", "Cat"], ["/m/02", "Dog"], ["/m/03", "Table"]]
    ).to_csv(d / "class-descriptions-boxable.csv", index=False, header=False)
    return root


def image_row(image_id: str, licence: str = CC_BY) -> dict:
    return {
        "ImageID": image_id,
        "Subset": "validation",
        "OriginalURL": f"https://example.invalid/{image_id}.jpg",
        "OriginalLandingURL": f"https://example.invalid/photo/{image_id}",
        "License": licence,
        "AuthorProfileURL": "https://example.invalid/people/x",
        "Author": "A Photographer",
        "Title": f"title {image_id}",
        "OriginalSize": 1234,
        "OriginalMD5": "abc==",
        "Thumbnail300KURL": "",
        "Rotation": "",
    }


def box_row(image_id, label, x0, y0, x1, y1, *, group=0, depiction=0, conf=1) -> dict:
    return {
        "ImageID": image_id,
        "Source": "xclick",
        "LabelName": label,
        "Confidence": conf,
        "XMin": x0,
        "XMax": x1,
        "YMin": y0,
        "YMax": y1,
        "IsOccluded": 0,
        "IsTruncated": 0,
        "IsGroupOf": group,
        "IsDepiction": depiction,
        "IsInside": 0,
    }


@pytest.fixture
def root(tmp_path):
    images = [image_row(f"img{i:02d}") for i in range(6)]
    images.append(image_row("nc01", licence=CC_BY_NC))  # wrong licence
    boxes = [
        # in band (area 0.04)
        *[box_row(f"img{i:02d}", "/m/01", 0.3, 0.3, 0.5, 0.5) for i in range(6)],
        # a second class on some images -> P3 control candidates
        *[box_row(f"img{i:02d}", "/m/02", 0.6, 0.6, 0.8, 0.8) for i in range(3)],
        box_row("nc01", "/m/01", 0.3, 0.3, 0.5, 0.5),
    ]
    return write_csvs(tmp_path, images, boxes)


def test_only_cc_by_images_enter_the_pool(root):
    pool = OI.select_pool(root=root, n=0)
    assert "nc01" not in set(pool.image_id)  # P1: CC-BY, so the bank is releasable
    assert len(pool) == 6


def test_group_and_depiction_boxes_are_not_instances(tmp_path):
    images = [image_row("a"), image_row("b")]
    boxes = [
        box_row("a", "/m/01", 0.2, 0.2, 0.6, 0.6, group=1),
        box_row("b", "/m/01", 0.2, 0.2, 0.6, 0.6, depiction=1),
    ]
    pool = OI.select_pool(root=write_csvs(tmp_path, images, boxes), n=0)
    assert len(pool) == 0


def test_boxes_below_the_area_floor_are_dropped(tmp_path):
    """A mask sits inside its box, so a box under 0.5% cannot make the P1 band."""
    images = [image_row("small"), image_row("ok")]
    boxes = [
        box_row("small", "/m/01", 0.10, 0.10, 0.13, 0.13),  # 0.09% of the image
        box_row("ok", "/m/01", 0.30, 0.30, 0.50, 0.50),  # 4%
    ]
    pool = OI.select_pool(root=write_csvs(tmp_path, images, boxes), n=0)
    assert set(pool.image_id) == {"ok"}


def test_boxes_above_the_band_do_not_qualify_an_image(tmp_path):
    images = [image_row("huge")]
    boxes = [box_row("huge", "/m/01", 0.0, 0.0, 0.9, 0.9)]  # 81%, over P1's 15%
    pool = OI.select_pool(root=write_csvs(tmp_path, images, boxes), n=0)
    assert len(pool) == 0


def test_selection_is_seeded_and_reproducible(root):
    a = OI.select_pool(root=root, n=4, seed=0)
    b = OI.select_pool(root=root, n=4, seed=0)
    c = OI.select_pool(root=root, n=4, seed=1)
    assert list(a.image_id) == list(b.image_id)
    assert list(a.image_id) != list(c.image_id) or len(a) < 2


def test_pool_carries_every_class_label_of_the_image(root):
    """P1: all class labels of the image are prompted."""
    pool = OI.select_pool(root=root, n=0).set_index("image_id")
    assert set(pool.loc["img00", "labels"]) == {"Cat", "Dog"}
    assert set(pool.loc["img05", "labels"]) == {"Cat"}
    assert pool.loc["img00", "n_labels"] == 2


def test_attribution_columns_are_recorded_per_image(root):
    """P1's attribution duty, and the reason the K1 bank may be released."""
    table = OI.attribution_table(OI.select_pool(root=root, n=0))
    assert set(table.columns) == {
        "image_id", "author", "author_profile_url", "licence",
        "original_url", "landing_url", "title",
    }
    assert (table.licence == CC_BY).all()
    assert table.author.notna().all()


def test_label_map_feeds_the_instance_pass(root):
    labels = OI.label_map(OI.select_pool(root=root, n=0))
    assert labels["img00"] == ["Cat", "Dog"]


def test_to_raw_items_converts_normalised_boxes_to_pixels(root, tmp_path):
    import cv2

    from .conftest import textured_image

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(6):
        img = textured_image((400, 800), seed=i)  # h=400, w=800
        cv2.imwrite(str(image_dir / f"img{i:02d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    pool = OI.select_pool(root=root, n=0)
    items = OI.to_raw_items(pool, image_dir=image_dir)
    assert len(items) == 6
    first = next(i for i in items if i.item_id == "oi_img00")
    # the cat box was 0.3-0.5 normalised on an 800x400 image
    x0, y0, x1, y1 = first.gt_boxes_xyxy_px[0]
    assert (round(x0), round(y0), round(x1), round(y1)) == (240, 120, 400, 200)
    assert "Cat" in first.expr and "Dog" in first.expr


def test_to_raw_items_skips_images_that_were_not_downloaded(root, tmp_path):
    assert OI.to_raw_items(OI.select_pool(root=root, n=0), image_dir=tmp_path / "empty") == []


def test_image_url_points_at_the_cvdf_mirror():
    url = OI.image_url("0001eeaf4aed83f9")
    assert url == (
        "https://open-images-dataset.s3.amazonaws.com/validation/0001eeaf4aed83f9.jpg"
    )


def test_missing_metadata_says_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="--metadata"):
        OI.load_tables(root=tmp_path)


def test_fetch_treats_an_existing_file_as_cached(tmp_path):
    target = tmp_path / "already.csv"
    target.write_text("x")
    ok, detail = OI.fetch("https://example.invalid/nope", target)
    assert ok and detail == "cached"
