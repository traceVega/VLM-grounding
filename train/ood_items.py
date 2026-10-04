"""Out-of-domain evaluation sets for train.eval_suite (evaluation only; never used for training or model selection,
except the declared dev split of PR-Bench Reject).

    python -m train.ood_items            # print the size and composition of every set

Every set is a seeded sample of no-target and single-target items (the COA protocol answers one box or none, so
multi-target items are out of scope), in the eval_suite item format: id, expr, image (path) or image_bytes, gt_boxes
(xyxy pixels, [] for no target), n_gt, set (subset label for the breakdown), kind (positive / negative).

  prbench_dev  PR-Bench (renamed RefBench-PRO): 300 of the 1,000 Reject items + 50 per positive task (declared dev split)
  prbench      the other 700 Reject items + 200 per positive task (attribute, position, interaction, relation, commonsense)
  finecops     FineCops-Ref test: 700 negative expressions + 300 negative images + 1,000 positives
  grefcoco     gRefCOCO: 1,000 no-target sentences (val/testA/testB) + 1,000 single-target sentences (testA/testB); images
               used by any RefCOCO training row we ever trained on are skipped
  humanref     HumanRef: the 1,000 rejection items + 1,000 single-person items from the other five domains
  openref      OpenRef test: the 488 none-target items + the 503 single-target items of single_test_labels
  refadv       Ref-Adv-s: all 1,142 (positives only; hard distractors)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

RAW = Path.home() / "vlmg-data/raw"
SETS = ("prbench_dev", "prbench", "finecops", "grefcoco", "humanref", "openref", "refadv")


def _item(id_, expr, gt, set_, image=None, image_bytes=None):
    it = {"id": id_, "expr": expr, "gt_boxes": gt, "n_gt": len(gt), "set": set_, "kind": "positive" if gt else "negative"}
    if image_bytes is not None:
        it["image_bytes"] = image_bytes
    else:
        it["image"] = str(image)
    return it


def _xywh(b):
    return [float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3])]


def prbench(dev: bool) -> list[dict]:
    root = RAW / "refbench-pro"
    rows = [json.loads(l) for l in open(root / "hf/annotation.jsonl", encoding="utf-8")]
    for i, r in enumerate(rows):
        r["_i"] = i
    rng = random.Random(0)
    rej = [r for r in rows if r["task_type"] == "reject"]
    rng.shuffle(rej)
    out = rej[:300] if dev else rej[300:]
    for task in ("attribute", "position", "interaction", "relation", "commonsense"):
        pos = [r for r in rows if r["task_type"] == task]
        rng.shuffle(pos)
        out += pos[:50] if dev else pos[50:250]
    return [_item(f"prbench:{r['_i']}", r["expression"], [[float(v) for v in r["bbox"]]] if r["bbox"] else [], r["task_type"],
                  image=root / "images" / r["image_path"]) for r in out]


def finecops() -> list[dict]:
    root = RAW / "finecops-ref"
    d = json.load(open(root / "figshare/test_expression_all_coco_format.json", encoding="utf-8"))
    imgs = {im["id"]: im for im in d["images"]}
    neg_t, neg_i, pos = [], [], []
    for a in d["annotations"]:
        cate = a.get("negative_cate")
        (neg_t if cate == "text" else neg_i if cate == "image" else pos).append(a)
    rng = random.Random(0)
    for lst in (neg_t, neg_i, pos):
        rng.shuffle(lst)
    out = []
    for a in neg_t[:700] + neg_i[:300] + pos[:1000]:
        im = imgs[a["image_id"]]
        fn = im["file_name"]
        path = root / ("final_neg_images" if fn.startswith("neg_") else "gqa_images/images") / fn
        neg = a.get("negative_cate") is not None
        label = f"neg_{a['negative_cate']}_{a.get('negative_type')}" if neg else f"pos_level{a.get('level')}"
        out.append(_item(f"finecops:{a['id']}", a.get("expression") or a.get("caption") or im.get("caption"), [] if neg else [_xywh(a["bbox"])],
                         label, image=path))
    return out


def grefcoco() -> list[dict]:
    root = RAW / "grefcoco"
    refs = json.load(open(root / "hf/grefs(unc).json", encoding="utf-8"))
    anns = {a["id"]: a for a in json.load(open(root / "hf/instances.json", encoding="utf-8"))["annotations"]}
    skip = set()
    ov = root / "TRAIN_OVERLAP_images.json"
    if ov.is_file():
        for names in json.load(open(ov)).values():
            skip |= set(names)
    none, single = [], []
    for r in refs:
        if r["split"] not in ("val", "testA", "testB") or r["file_name"] in skip:
            continue
        for s in r["sentences"]:
            if r.get("no_target") or r["ann_id"] == [-1]:
                none.append((r, s))
            elif len(r["ann_id"]) == 1 and r["split"] != "val":
                single.append((r, s))
    rng = random.Random(0)
    rng.shuffle(none)
    rng.shuffle(single)
    out = []
    for r, s in none[:1000] + single[:1000]:
        gt = [] if r["ann_id"] == [-1] else [_xywh(anns[r["ann_id"][0]]["bbox"])]
        out.append(_item(f"grefcoco:{s['sent_id']}", s["sent"], gt, f"{r['split']}_{'none' if not gt else 'single'}",
                         image=root / "train2014" / r["file_name"]))
    return out


def humanref() -> list[dict]:
    root = RAW / "humanref/hf"
    rows = [json.loads(l) for l in open(root / "annotations.jsonl", encoding="utf-8")]
    rej = [r for r in rows if not r["answer_boxes"]]
    single = [r for r in rows if len(r["answer_boxes"]) == 1]
    rng = random.Random(0)
    rng.shuffle(single)
    return [_item(f"humanref:{r['id']}", r["referring"], [[float(v) for v in b] for b in r["answer_boxes"]], r["domain"],
                  image=root / "images" / r["image_name"]) for r in rej + single[:1000]]


def openref() -> list[dict]:
    root = RAW / "openref/test"
    where = {p.name: p for p in root.rglob("*.jpg")}
    out = []
    for fname, label in (("none_test_labels.json", "none"), ("single_test_labels.json", "single")):
        for k, r in enumerate(json.load(open(root / fname, encoding="utf-8"))):
            gt = []
            if r["bbox_num"] and label == "single":
                p = r["proposal"][0]
                W, H = p["original_width"], p["original_height"]
                gt = [[p["x"] / 100 * W, p["y"] / 100 * H, (p["x"] + p["width"]) / 100 * W, (p["y"] + p["height"]) / 100 * H]]
            out.append(_item(f"openref:{label}:{k}", r["positive"], gt, label, image=where[r["image"]]))
    return out


def refadv() -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    for f in sorted((RAW / "ref-adv/hf/data").glob("*.parquet")):
        t = pq.read_table(f, columns=["image", "normal_caption", "solution", "row_idx", "image_source"])
        d = t.to_pydict()
        for i in range(t.num_rows):
            out.append(_item(f"refadv:{d['row_idx'][i]}", d["normal_caption"][i], [[float(v) for v in d["solution"][i]]],
                             d["image_source"][i], image_bytes=d["image"][i]["bytes"]))
    return out


def items(name: str) -> list[dict]:
    return {"prbench_dev": lambda: prbench(True), "prbench": lambda: prbench(False), "finecops": finecops, "grefcoco": grefcoco,
            "humanref": humanref, "openref": openref, "refadv": refadv}[name]()


def summary(rows: list[dict]) -> dict:
    """No-target accuracy (answered none), positive accuracy at IoU 0.5 and mean over IoU 0.5:0.9, per set label."""
    def block(rs):
        neg = [r for r in rs if r["n_gt"] == 0]
        pos = [r for r in rs if r["n_gt"] > 0]
        s = {"n": len(rs), "n_neg": len(neg), "n_pos": len(pos)}
        if neg:
            s["no_target_acc"] = round(sum(r["output_type"] != "box" for r in neg) / len(neg), 4)
        if pos:
            s["pos_acc"] = round(sum(r["correct"] for r in pos) / len(pos), 4)
            s["pos_macc_50_90"] = round(sum(sum(r["iou"] >= t for t in (0.5, 0.6, 0.7, 0.8, 0.9)) / 5 for r in pos) / len(pos), 4)
            s["pos_null_rate"] = round(sum(r["output_type"] != "box" for r in pos) / len(pos), 4)
        return s

    out = {"all": block(rows)}
    for lab in sorted({r.get("set") for r in rows}):
        out[lab] = block([r for r in rows if r.get("set") == lab])
    return out


if __name__ == "__main__":
    for name in SETS:
        its = items(name)
        missing = sum(1 for it in its if "image" in it and not Path(it["image"]).is_file())
        print(f"{name:12s} {len(its):5d} items | neg {sum(it['n_gt'] == 0 for it in its):5d} | missing images {missing} | "
              f"{dict(Counter(it['set'] for it in its).most_common(8))}")
