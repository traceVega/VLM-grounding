"""Data for the PoC: the exported triplets, a scene-level split, the answer format the base
model itself uses, and the evaluation sets (GroundingME, own validation + gray control,
RefCOCO family samples).

Answer format mirrors the base model's own output on this prompt (fenced JSON, 0-1000
relative coordinates), so SFT does not shift the format distribution:

    ```json
    {"bbox_2d": [79, 339, 475, 797]}
    ```
    ```json
    {"bbox_2d": null}
    ```
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

from PIL import Image

from shared import paths

EXPORT = paths.DATA_ROOT / "datagen" / "exports" / "poc_v1.jsonl"
REFCOCO_DIR = paths.RAW / "refcoco"
TRAIN_ROOT = paths.DATA_ROOT / "train"
NULL_TYPES = ("none", "refusal")  # harness parser: a null answer is output_type "none", a safety refusal "refusal"
MODELS = {
    "4b": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    "8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
}


def load_items(path: Path = EXPORT) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def split(items: list[dict], n_val_scenes: int = 18, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Scene-level split: every item of a scene lands on the same side."""
    groups = sorted({it["group"] for it in items})
    rng = random.Random(seed)
    rng.shuffle(groups)
    val = set(groups[:n_val_scenes])
    return [it for it in items if it["group"] not in val], [it for it in items if it["group"] in val]


def relative_1000(box: list[float], wh: list[int]) -> list[int]:
    W, H = wh
    x0, y0, x1, y1 = box
    rel = [round(x0 * 1000 / W), round(y0 * 1000 / H), round(x1 * 1000 / W), round(y1 * 1000 / H)]
    return [min(1000, max(0, v)) for v in rel]


def target_text(item: dict) -> str:
    b = item["answer"]["bbox_2d"]
    if b is None:
        return '```json\n{"bbox_2d": null}\n```'
    return '```json\n{"bbox_2d": [%d, %d, %d, %d]}\n```' % tuple(relative_1000(b, item["image_wh"]))


def training_order(items: list[dict], seed: int) -> list[dict]:
    """Scenes shuffled; inside a scene the order is target positive, sibling positive,
    negative, so the three land in the same accumulation window."""
    rank = {"positive": 0, "sibling_positive": 1, "negative": 2}
    by_group: dict[str, list[dict]] = {}
    for it in items:
        by_group.setdefault(it["group"], []).append(it)
    groups = list(by_group)
    random.Random(seed).shuffle(groups)
    out = []
    for g in groups:
        out.extend(sorted(by_group[g], key=lambda it: rank[it["kind"]]))
    return out


def policy_4b_lookup() -> dict[tuple[str, str, str], dict]:
    """(run, image_id, which) -> the 4B base model's answer, from datagen's policy_4b.jsonl."""
    out = {}
    for run in ("test50c", "poc155"):
        f = paths.DATA_ROOT / "datagen" / run / "policy_4b.jsonl"
        if f.is_file():
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                out[(run, r["image_id"], r["which"])] = r
    return out


def base4b_view(item: dict, lookup: dict) -> dict | None:
    which = {"positive": "target", "sibling_positive": "sibling",
             "negative": "neg2" if item.get("reflipped") else "neg"}[item["kind"]]
    return lookup.get((item["run"], item["group"], which))


def gray_image(wh) -> Image.Image:
    return Image.new("RGB", tuple(wh), (128, 128, 128))


# --- evaluation sets ---------------------------------------------------------


def gme_items() -> list[dict]:
    import pyarrow.parquet as pq

    t = pq.read_table(paths.PREPARED / "groundingme" / "items.parquet")
    d = t.to_pydict()
    return [{"id": d["item_id"][i], "expr": d["expr"][i], "image": str(paths.DATA_ROOT / d["image_path"][i]),
             "gt_boxes": d["gt_boxes_xyxy_px"][i] or [], "n_gt": d["n_gt"][i],
             "dimension": d["dimension"][i], "size_bin": d["size_bin"][i]} for i in range(t.num_rows)]


def refcoco_items(n_per_set: int = 300, seed: int = 0) -> list[dict]:
    """A seeded sample from the downloaded val shard of each set; first expression per box;
    the image stays as bytes until evaluation."""
    import pyarrow.parquet as pq

    out = []
    for sub in ("RefCOCO", "RefCOCOplus", "RefCOCOg"):
        files = sorted((REFCOCO_DIR / sub / "data").glob("*.parquet"))
        if not files:
            continue
        t = pq.read_table(files[0])
        d = t.to_pydict()
        idx = list(range(t.num_rows))
        random.Random(seed).shuffle(idx)
        for i in idx[:n_per_set]:
            x, y, w, h = d["bbox"][i]
            out.append({"id": f"{sub}:{d['question_id'][i]}", "set": sub, "expr": d["answer"][i][0],
                        "image_bytes": d["image"][i]["bytes"], "gt_boxes": [[x, y, x + w, y + h]], "n_gt": 1})
    return out


def open_image(item: dict) -> Image.Image:
    if item.get("image_bytes") is not None:
        return Image.open(io.BytesIO(item["image_bytes"])).convert("RGB")
    return Image.open(item["image"]).convert("RGB")
