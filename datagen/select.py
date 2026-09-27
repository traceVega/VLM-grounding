"""Stage `select` (CPU): pick crowded whole-object scenes from the OpenImages instance bank.

The bank's `n_head_noun_instances` column is 1 on every K1 row (head_noun is
null there), so same-category crowding is counted from the
`same_class_other_instance` rows instead (finding of 2026-09-21).
"""

from __future__ import annotations

import glob
import random

import pyarrow.parquet as pq

from datagen import common as C
from shared import paths

COLS = ["image_id", "source", "label", "area_frac", "bbox_xyxy_px", "mask_rle", "instance_id"]


def load_bank() -> dict[str, dict]:
    root = paths.PREPARED / "openimages_pool" / "instances"
    scenes: dict[str, dict] = {}
    for shard in sorted(glob.glob(str(root / "shard-*.parquet"))):
        t = pq.read_table(shard, columns=COLS)
        cols = {n: t.column(n).to_pylist() for n in COLS}
        for i in range(t.num_rows):
            src = cols["source"][i]
            if src not in ("referent", "same_class_other_instance"):
                continue
            s = scenes.setdefault(cols["image_id"][i], {"referent": None, "siblings": []})
            row = {
                "instance_id": cols["instance_id"][i],
                "label": cols["label"][i],
                "area": float(cols["area_frac"][i]),
                "box": [float(v) for v in cols["bbox_xyxy_px"][i]],
                "mask_rle": cols["mask_rle"][i],
            }
            if src == "referent":
                s["referent"] = row
            else:
                s["siblings"].append(row)
    return scenes


def build_scene(image_id: str, s: dict) -> dict | None:
    ref = s["referent"]
    if ref is None or ref["label"] not in C.WHITELIST:
        return None
    x0, y0, x1, y1 = ref["box"]
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    if ref["area"] < C.MIN_TARGET_AREA or max(bw / bh, bh / bw) > C.MAX_TARGET_ASPECT:
        return None
    # A small box touching the frame edge is a truncated object (test50: a 60 px van
    # slice at x=0, a 37 px person at the right edge); the writer describes the whole
    # object it imagines and the checker rightly says no.
    try:
        from PIL import Image

        with Image.open(C.image_path(image_id)) as im:
            W, H = im.size
    except Exception:
        return None
    touches = (x0 <= 2) + (y0 <= 2) + (x1 >= W - 2) + (y1 >= H - 2)
    if touches and ref["area"] < C.MIN_TARGET_AREA_AT_BORDER:
        return None
    sibs = [x for x in s["siblings"] if x["area"] >= C.MIN_SIBLING_AREA]
    # Candidate hygiene (test50b review): SAM 3 same-class rows include parts, nested
    # duplicates and objects that overlap the target so that a red outline is ambiguous.
    W0, H0 = float(W), float(H)
    for x in sibs:
        bx = x["box"]
        if C.containment(ref["box"], bx) >= C.MAX_NESTING:
            return None  # the target sits inside another same-class candidate's box: occluded / a part
        if C.box_iou(ref["box"], bx) > C.MAX_TARGET_OVERLAP:
            return None
        if (bx[2] - bx[0]) * (bx[3] - bx[1]) > 0.6 * W0 * H0:
            return None  # a "candidate" covering most of the frame is not a describable object
    keep = []
    for x in sorted(sibs, key=lambda z: -z["area"]):
        if C.containment(x["box"], ref["box"]) >= C.MAX_NESTING:
            continue  # a part of the target
        if any(C.containment(x["box"], k["box"]) >= C.MAX_NESTING or C.box_iou(x["box"], k["box"]) > 0.5
               for k in keep):
            continue  # nested in / duplicate of a kept sibling
        keep.append(x)
    sibs = keep
    n = 1 + len(sibs)
    if n < C.MIN_CANDIDATES:
        return None
    if n > C.MAX_CANDIDATES:
        sibs = sorted(sibs, key=lambda x: -x["area"])[: C.MAX_CANDIDATES - 1]
    cands = [dict(ref, is_target=True)] + [dict(x, is_target=False) for x in sibs]
    cands.sort(key=lambda x: x["box"][0])  # left to right, so ids are stable
    for k, c in enumerate(cands, 1):
        c["iid"] = k
    target = next(c["iid"] for c in cands if c["is_target"])
    wh = [W, H]
    return {
        "image_id": image_id,
        "label": ref["label"],
        "category": C.head_noun(ref["label"]),
        "image_wh": wh,
        "target_iid": target,
        "n_candidates": len(cands),
        "instances": [
            {k: c[k] for k in ("iid", "instance_id", "box", "area", "mask_rle", "is_target")}
            for c in cands
        ],
    }


def run(run: str, n: int, seed: int, per_label: int, exclude_runs: list[str] | None = None) -> None:
    root = C.run_root(run)
    out = root / "scenes.jsonl"
    if out.is_file():
        print(f"{out} exists with {len(C.read_jsonl(out))} scenes; delete it to reselect")
        return
    bank = load_bank()
    print(f"bank: {len(bank)} images")
    usable = []
    for image_id in sorted(bank):
        sc = build_scene(image_id, bank[image_id])
        if sc:
            usable.append(sc)
    print(f"usable crowded whole-object scenes: {len(usable)}")
    skip = set()
    for other in exclude_runs or []:
        skip |= {s["image_id"] for s in C.read_jsonl(paths.DATA_ROOT / "datagen" / other / "scenes.jsonl")}
    if skip:
        usable = [sc for sc in usable if sc["image_id"] not in skip]
        print(f"after excluding {exclude_runs}: {len(usable)} scenes")
    rng = random.Random(seed)
    rng.shuffle(usable)
    picked, per = [], {}
    for sc in usable:
        if per_label and per.get(sc["label"], 0) >= per_label:
            continue
        per[sc["label"]] = per.get(sc["label"], 0) + 1
        picked.append(sc)
        if len(picked) >= n:
            break
    for sc in picked:
        C.append_jsonl(out, sc)
    labels = {}
    for sc in picked:
        labels[sc["label"]] = labels.get(sc["label"], 0) + 1
    print(f"selected {len(picked)} -> {out}")
    print("by label:", dict(sorted(labels.items(), key=lambda kv: -kv[1])))
    print("candidates per scene:", sorted(sc["n_candidates"] for sc in picked))
