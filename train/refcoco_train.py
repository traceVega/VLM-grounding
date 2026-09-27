"""RefCOCO/+/g *train* items for RL (answer reward only): annotations from the jxu124 HF
repos (train split, boxes xyxy in pixels, several captions per box), images fetched one by
one from images.cocodataset.org (COCO train2014) for the sampled items only.

    python -m train.refcoco_train [--n-per-set 200] [--seed 0] -> $VLMG_DATA_ROOT/train/refcoco_train.jsonl (source "refcoco")

One item per sampled box with a random one of its captions; group "refcoco:<set>:<image id>".
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request

from PIL import Image

from train import data as D

REPOS = {"RefCOCO": ("jxu124/refcoco", "data/train-00000-of-00001-94431d5f4bd5b93f.parquet"),
         "RefCOCOplus": ("jxu124/refcocoplus", "data/train-00000-of-00001-7294665695c630ee.parquet"),
         "RefCOCOg": ("jxu124/refcocog", "data/train-00000-of-00001-4fe3e6340cfb69ed.parquet")}
OUT = D.TRAIN_ROOT / "refcoco_train.jsonl"
IMG_DIR = D.REFCOCO_DIR / "train2014"
URL = "http://images.cocodataset.org/train2014/{}"


def fetch(file_name: str):
    path = IMG_DIR / file_name
    if path.is_file():
        return path
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(URL.format(file_name), path)
            return path
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def main() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-set", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    items = []
    for sub, (repo, fname) in REPOS.items():
        p = hf_hub_download(repo, fname, repo_type="dataset")
        t = pq.read_table(p, columns=["file_name", "image_id", "bbox", "captions", "ann_id", "raw_image_info"])
        d = t.to_pydict()
        idx = list(range(t.num_rows))
        rng.shuffle(idx)
        n = 0
        t0 = time.time()
        for i in idx:
            if n >= args.n_per_set:
                break
            caps = [c for c in (d["captions"][i] or []) if c and len(c.split()) >= 1]
            if not caps:
                continue
            info = json.loads(d["raw_image_info"][i]) if isinstance(d["raw_image_info"][i], str) else (d["raw_image_info"][i] or {})
            file_name = info.get("file_name") or f"COCO_train2014_{int(d['image_id'][i]):012d}.jpg"
            path = fetch(file_name)
            if path is None:
                continue
            with Image.open(path) as im:
                W, H = im.size
            x0, y0, x1, y1 = d["bbox"][i]
            if x1 <= x0 + 1 or y1 <= y0 + 1:
                continue
            items.append({"image": str(path), "image_wh": [W, H], "group": f"refcoco:{sub}:{d['image_id'][i]}", "category": None, "label": None,
                          "run": "refcoco", "source": "refcoco", "n_candidates": None, "id": f"refcoco:{sub}:{d['ann_id'][i]}", "kind": "positive",
                          "expression": rng.choice(caps), "answer": {"bbox_2d": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]}, "target_iid": None})
            n += 1
            if n % 50 == 0:
                print(f"  {sub}: {n} items, {(time.time() - t0) / 60:.1f} min", flush=True)
        print(f"{sub}: {n} items", flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} items -> {OUT}")


if __name__ == "__main__":
    main()
