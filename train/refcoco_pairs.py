"""Contrastive audit pairs from RefCOCO / + / g *train* annotations (labels from the annotations only, no download).

    python -m train.refcoco_pairs [--per-image 2] [--seed 0]  ->  $VLMG_DATA_ROOT/train/refcoco_pairs.jsonl

A referring expression was written about one object (A) to tell it apart from the other objects of the image.  So for
an expression of A:
  * on A's box every condition derived from the expression holds (a named mismatch there is a false accusation);
  * on the box of another referred object B of the same image and category the expression does not hold (the audit
    has to name a mismatch).
One item = (image, expression of A, A's box, the boxes of up to two such siblings B).  Only images that the cleaned
replay file refcoco_train_v2.jsonl already uses (local, no evaluation image), siblings with IoU < 0.3 to A.  Used by
train.grpo_lora --audit-replay (audit-only rollouts: train.grpo_coa.audit_pair_rollouts).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

from shared.harness import parsers as P
from train import data as D
from train import refcoco_train as RT

OUT = D.TRAIN_ROOT / "refcoco_pairs.jsonl"


def main() -> None:
    import pyarrow.parquet as pq

    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # annotations come from the local HF cache
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-image", type=int, default=2, help="expressions (target objects) kept per image and dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    clean = {Path(it["image"]).name: it["image"] for it in D.load_items(D.TRAIN_ROOT / "refcoco_train_v2.jsonl")}
    items = []
    for sub, (repo, fname) in RT.REPOS.items():
        t = pq.read_table(hf_hub_download(repo, fname, repo_type="dataset"),
                          columns=["image_id", "bbox", "captions", "ann_id", "category_id", "raw_image_info", "split"]).to_pydict()
        by_img: dict[str, dict] = defaultdict(dict)
        for i in range(len(t["ann_id"])):
            assert t["split"][i] == "train", t["split"][i]
            info = json.loads(t["raw_image_info"][i]) if isinstance(t["raw_image_info"][i], str) else (t["raw_image_info"][i] or {})
            fn = info.get("file_name") or f"COCO_train2014_{int(t['image_id'][i]):012d}.jpg"
            caps = [c for c in (t["captions"][i] or []) if c and c.strip()]
            x0, y0, x1, y1 = t["bbox"][i]
            if fn in clean and caps and x1 > x0 + 1 and y1 > y0 + 1:
                by_img[fn][t["ann_id"][i]] = {"box": [round(v, 1) for v in (x0, y0, x1, y1)], "caps": caps, "cat": t["category_id"][i]}
        n = 0
        for fn in sorted(by_img):
            objs = by_img[fn]
            ids = sorted(objs)
            rng.shuffle(ids)
            kept = 0
            with Image.open(clean[fn]) as im:
                wh = list(im.size)
            for a in ids:
                sib = [b for b in ids if b != a and objs[b]["cat"] == objs[a]["cat"] and P.iou(tuple(objs[a]["box"]), tuple(objs[b]["box"])) < 0.3]
                if not sib:
                    continue
                rng.shuffle(sib)
                items.append({"id": f"rcpair:{sub}:{a}", "image": clean[fn], "image_wh": wh, "group": f"rcpair:{sub}:{fn}", "run": "refcoco", "source": "rcpair",
                              "kind": "positive", "expression": rng.choice(objs[a]["caps"]), "captions": objs[a]["caps"], "answer": {"bbox_2d": objs[a]["box"]},
                              "neg_boxes": [objs[b]["box"] for b in sib[:2]], "category": objs[a]["cat"], "label": None, "n_candidates": None, "target_iid": None})
                kept += 1
                n += 1
                if kept >= args.per_image:
                    break
        print(f"{sub}: {n} items over {len(by_img)} local clean images", flush=True)
    rng.shuffle(items)
    with open(Path(args.out).expanduser(), "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
