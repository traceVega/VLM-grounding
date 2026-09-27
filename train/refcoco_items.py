"""RefCOCO/+/g *train* items for RL (answer reward only; no label matrix): a seeded sample from
the downloaded train shards, images written to disk, one expression per box.

    python -m train.refcoco_items [--n-per-set 200] [--seed 0] -> $VLMG_DATA_ROOT/train/refcoco_train.jsonl (source "refcoco")

Items carry group "refcoco:<set>:<image>" so the trainers keep them apart from the datagen
scenes; train.traces has no labels for them, so they are answer-only prompts in GRPO.
"""

from __future__ import annotations

import argparse
import io
import json
import random

from PIL import Image

from train import data as D

OUT = D.TRAIN_ROOT / "refcoco_train.jsonl"
IMG_DIR = D.REFCOCO_DIR / "train_images"


def main() -> None:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-set", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for sub in ("RefCOCO", "RefCOCOplus", "RefCOCOg"):
        files = sorted((D.REFCOCO_DIR / sub / "data").glob("train-*.parquet"))
        if not files:
            print(f"{sub}: no train shard")
            continue
        t = pq.read_table(files[0])
        d = t.to_pydict()
        idx = list(range(t.num_rows))
        random.Random(args.seed).shuffle(idx)
        n = 0
        for i in idx:
            if n >= args.n_per_set:
                break
            x, y, w, h = d["bbox"][i]
            expr = (d["answer"][i] or [None])[0]
            if not expr or w <= 1 or h <= 1:
                continue
            qid = str(d["question_id"][i])
            path = IMG_DIR / f"{sub}_{qid}.jpg"
            if not path.is_file():
                im = Image.open(io.BytesIO(d["image"][i]["bytes"])).convert("RGB")
                im.save(path, quality=92)
                W, H = im.size
            else:
                with Image.open(path) as im:
                    W, H = im.size
            items.append({"image": str(path), "image_wh": [W, H], "group": f"refcoco:{sub}:{d['file_name'][i]}", "category": None, "label": None,
                          "run": "refcoco", "source": "refcoco", "n_candidates": None, "id": f"refcoco:{sub}:{qid}", "kind": "positive",
                          "expression": expr, "answer": {"bbox_2d": [round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)]}, "target_iid": None})
            n += 1
        print(f"{sub}: {n} items from {files[0].name}")
    with open(OUT, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} items -> {OUT}")


if __name__ == "__main__":
    main()
