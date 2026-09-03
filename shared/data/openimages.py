"""The K1 pool from Open Images (design P1).

    10,000 OpenImages images (CC-BY; author, URL and licence recorded per image
    from the images CSV for attribution) with at least one annotated object whose
    SAM 3 mask, prompted with the image's class labels, covers 0.5% to 15% of the
    image; one referent instance per image with seed 0. All class labels of the
    image are prompted and class-agnostic masks above 0.5% area are added [...]
    CC-BY images make the K1 bank the only releasable edited artefact.

The **validation** split is used, not train: it is 41,620 images, *all* of them
CC-BY 2.0, with a 24 MB box file, against a 2.15 GB box file for train.  20,535
of them carry a box inside P1's band -- twice the pool P1 asks for -- so there is
headroom for images the SAM 3 pass then drops.  A pool is a pool: nothing about
K1 depends on which split the pixels came from, and the smaller download is the
difference between 38 MB of metadata and 2.8 GB of it (DEVIATIONS D-19).

Two filters run before SAM 3 is ever loaded:

* ``IsGroupOf`` and ``IsDepiction`` boxes are dropped -- P1 wants an object
  instance, not a crowd of them or a drawing of one.
* Boxes under 0.5% of the image are dropped.  A mask sits inside its box, so a
  box under 0.5% cannot produce a mask over 0.5%: this filter is sound rather
  than heuristic, and it saves SAM 3 time.  There is no upper filter, because a
  large box can still hold a small mask.
"""

from __future__ import annotations

import concurrent.futures as cf
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shared import paths
from shared.data.items import RawItem

SPLIT = "validation"

IMAGES_CSV_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
BOXES_CSV_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
CLASSES_CSV_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

METADATA_FILES = {
    "validation-images-with-rotation.csv": IMAGES_CSV_URL,
    "validation-annotations-bbox.csv": BOXES_CSV_URL,
    "class-descriptions-boxable.csv": CLASSES_CSV_URL,
}

P1_AREA_BAND = (0.005, 0.15)  # P1, on the SAM 3 mask; used here on the box
POOL_SIZE = 10_000  # P1
CC_BY = "/licenses/by/"

POOL_COLUMNS = (
    "image_id",
    "rank",
    "labels",
    "boxes",
    "n_labels",
    "author",
    "author_profile_url",
    "licence",
    "original_url",
    "landing_url",
    "title",
)


def pool_root(root: Path | None = None) -> Path:
    return (root or paths.RAW) / "openimages"


def load_tables(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(images, boxes, classes)`` from the three metadata CSVs."""
    d = pool_root(root)
    missing = [n for n in METADATA_FILES if not (d / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing Open Images metadata in {d}: {missing}. "
            "Fetch it with `python -m shared.data.openimages --metadata`."
        )
    images = pd.read_csv(d / "validation-images-with-rotation.csv")
    boxes = pd.read_csv(d / "validation-annotations-bbox.csv")
    classes = pd.read_csv(
        d / "class-descriptions-boxable.csv", header=None, names=["LabelName", "Class"]
    )
    return images, boxes, classes


def clean_boxes(boxes: pd.DataFrame, min_area: float = P1_AREA_BAND[0]) -> pd.DataFrame:
    """Instance boxes only, above the sound area floor.  Adds ``area``."""
    out = boxes[
        (boxes.IsGroupOf == 0) & (boxes.IsDepiction == 0) & (boxes.Confidence == 1)
    ].copy()
    out["area"] = (out.XMax - out.XMin) * (out.YMax - out.YMin)
    return out[out.area >= min_area]


def select_pool(
    root: Path | None = None,
    n: int = POOL_SIZE,
    seed: int = 0,
    band: tuple[float, float] = P1_AREA_BAND,
) -> pd.DataFrame:
    """P1's pool: ``n`` CC-BY images each carrying a box inside the band.

    Selection is by seeded shuffle over the qualifying image ids, so the pool is
    reproducible from the CSVs alone.  More images qualify than are needed; the
    surplus is returned in ``rank`` order so the SAM 3 pass can walk past images
    whose mask falls outside the band without re-drawing the sample.
    """
    images, boxes, classes = load_tables(root)
    label_name = dict(zip(classes.LabelName, classes.Class))

    ccby = images[images["License"].str.contains(CC_BY, na=False)]
    clean = clean_boxes(boxes, band[0])
    in_band = clean[(clean.area >= band[0]) & (clean.area <= band[1])]
    qualifying = sorted(set(in_band.ImageID) & set(ccby.ImageID))
    if n and len(qualifying) < n:
        # a short pool is a real problem worth stopping on; "give me everything"
        # (n=0) may legitimately come back empty
        raise RuntimeError(
            f"P1 asks for {n:,} images but only {len(qualifying):,} CC-BY images carry a box "
            f"inside {band}. Widen the band, add the train split, or lower n -- and record it."
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(qualifying))
    ranked = [qualifying[int(i)] for i in order]

    keep = set(ranked)
    per_image = clean[clean.ImageID.isin(keep)].groupby("ImageID")
    labels = per_image.LabelName.apply(lambda s: sorted({label_name.get(x, x) for x in s}))
    box_rows = per_image.apply(
        lambda g: [
            {
                "label": label_name.get(r.LabelName, r.LabelName),
                "xyxy_norm": [float(r.XMin), float(r.YMin), float(r.XMax), float(r.YMax)],
                "area": float(r.area),
            }
            for r in g.itertuples()
        ],
        include_groups=False,
    )

    meta = ccby.set_index("ImageID")
    rows = []
    for rank, image_id in enumerate(ranked):
        m = meta.loc[image_id]
        rows.append(
            {
                "image_id": image_id,
                "rank": rank,
                "labels": labels.get(image_id, []),
                "boxes": box_rows.get(image_id, []),
                "n_labels": len(labels.get(image_id, [])),
                "author": m["Author"],
                "author_profile_url": m["AuthorProfileURL"],
                "licence": m["License"],
                "original_url": m["OriginalURL"],
                "landing_url": m["OriginalLandingURL"],
                "title": m["Title"],
            }
        )
    pool = pd.DataFrame(rows, columns=POOL_COLUMNS).sort_values("rank").reset_index(drop=True)
    return pool.head(n) if n else pool


def image_url(image_id: str, split: str = SPLIT) -> str:
    return IMAGE_URL.format(split=split, image_id=image_id)


def fetch(url: str, destination: Path, timeout: float = 60.0) -> tuple[bool, str]:
    if destination.exists() and destination.stat().st_size > 0:
        return True, "cached"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return True, f"{len(data)} bytes"
    except Exception as exc:  # network, 404, throttling
        return False, f"{type(exc).__name__}: {exc}"


def download_images(
    image_ids: list[str],
    destination: Path | None = None,
    *,
    split: str = SPLIT,
    workers: int = 16,
    progress_every: int = 500,
) -> dict[str, str]:
    """Fetch only the pool's images from the CVDF mirror.  Returns the failures."""
    destination = destination or (pool_root() / "images")
    destination.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch, image_url(i, split), destination / f"{i}.jpg"): i
            for i in image_ids
        }
        for future in cf.as_completed(futures):
            image_id = futures[future]
            ok, detail = future.result()
            if not ok:
                failures[image_id] = detail
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(image_ids)} fetched, {len(failures)} failed", flush=True)
    return failures


def to_raw_items(pool: pd.DataFrame, image_dir: Path | None = None) -> list[RawItem]:
    """Pool rows -> ``RawItem``s, with boxes converted to original pixels.

    Open Images boxes are normalised, so the image has to be opened for its
    size; ``items.to_row`` opens it again for the hash and pHash, which is two
    header reads per image and not worth caching.

    ``expr`` carries the image's class labels joined by "; ".  A pool image has
    no referring expression: P1 prompts *every* class label, and the referent is
    chosen from the resulting SAM 3 masks, not from the text.
    """
    from PIL import Image

    image_dir = image_dir or (pool_root() / "images")
    items: list[RawItem] = []
    for row in pool.itertuples():
        path = image_dir / f"{row.image_id}.jpg"
        if not path.is_file():
            continue
        with Image.open(path) as im:
            width, height = im.size
        boxes = [
            (
                b["xyxy_norm"][0] * width,
                b["xyxy_norm"][1] * height,
                b["xyxy_norm"][2] * width,
                b["xyxy_norm"][3] * height,
            )
            for b in row.boxes
        ]
        items.append(
            RawItem(
                item_id=f"oi_{row.image_id}",
                image_path=path,
                expr="; ".join(row.labels),
                gt_boxes_xyxy_px=boxes,
                in_kill_set=True,
            )
        )
    return items


def attribution_table(pool: pd.DataFrame) -> pd.DataFrame:
    """P1's attribution duty: author, URL and licence per image.

    Kept beside the pool rather than in ``items.parquet``, whose columns are
    fixed by SPEC Section 3 (DEVIATIONS D-19).  This file must travel with the
    K1 bank if it is ever released.
    """
    return pool[
        [
            "image_id",
            "author",
            "author_profile_url",
            "licence",
            "original_url",
            "landing_url",
            "title",
        ]
    ].copy()


def label_map(pool: pd.DataFrame) -> dict[str, list[str]]:
    """image_id -> the class labels to prompt SAM 3 with (P1)."""
    return {r.image_id: list(r.labels) for r in pool.itertuples()}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", action="store_true", help="fetch the three CSVs (38 MB)")
    ap.add_argument("--select", action="store_true", help="build and summarise the pool")
    ap.add_argument("--images", action="store_true", help="fetch the pool's images (~3 GB)")
    ap.add_argument("-n", type=int, default=POOL_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = pool_root()
    if args.metadata:
        root.mkdir(parents=True, exist_ok=True)
        for name, url in METADATA_FILES.items():
            ok, detail = fetch(url, root / name)
            print(f"{'ok ' if ok else 'FAIL'} {name}: {detail}")

    if args.select or args.images:
        pool = select_pool(n=args.n, seed=args.seed)
        out = paths.PREPARED / "openimages_pool"
        out.mkdir(parents=True, exist_ok=True)
        pool.to_parquet(out / "pool.parquet")
        attribution_table(pool).to_csv(out / "attribution.csv", index=False)
        print(f"pool: {len(pool)} images -> {out/'pool.parquet'}")
        print(f"  distinct labels per image: mean {pool.n_labels.mean():.2f}, "
              f">=2 in {100 * (pool.n_labels >= 2).mean():.1f}% (P3 control candidates)")
        print(f"  attribution rows -> {out/'attribution.csv'}")

    if args.images:
        pool = pd.read_parquet(paths.PREPARED / "openimages_pool" / "pool.parquet")
        print(f"fetching {len(pool)} images to {root/'images'} ...")
        failures = download_images(list(pool.image_id))
        print(f"done; {len(failures)} failures")
        for image_id, why in list(failures.items())[:10]:
            print(f"  {image_id}: {why}")


if __name__ == "__main__":
    main()
