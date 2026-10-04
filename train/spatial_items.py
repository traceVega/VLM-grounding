"""Spatial-relation grounding items (training data for the Spatial dimension), every label judged from boxes so the RL
reward stays verifiable.

    python -m train.spatial_items [--source oiraw|oi|vg] [--n-scenes 400] [--min-cands 3] [--seed 0] [--out <jsonl>]

Sources:
  oiraw  (default) the OpenImages validation boxes of our local images: human-verified and exhaustive for each image's
      positive classes; see oi_raw_scenes.
  oi  the OpenImages instance bank (prepared/openimages_pool): the same-class candidates are the SAM 3 instances with the
      datagen hygiene (datagen.select.build_scene: no nested / duplicate / frame-sized candidates), so every instance of
      the class is a candidate; references are OpenImages-labelled objects of another class that occur once.  The own
      validation scenes (train.data.split) are excluded.
  vg  Visual Genome (no-COCO images, eval-overlap ids excluded).  Review of a test sheet (2026-09-30): VG leaves many
      instances unannotated ("the rightmost man" with ten men in the picture and three boxed), so ordinal labels are
      unreliable there; kept for reference, use oi.
Conditions, each kept only when every candidate gets a clear yes / no (ambiguous geometry -> the condition is not used):
  ordinal   "the second <name> from the left" among the candidates (centre gaps >= 5 % of the width), or the vertical
            analogue ("the <name> nearest the top");
  relation  "is to the left of / to the right of / above / below the <ref>": yes = the whole box is on that side (margin
            2 %), no = the centre is on the other side of the reference's centre.
Positive: two conditions the target alone satisfies, at least one of them not unique on its own.  Negative: one condition
replaced by its opposite (left <-> right, above <-> below, another ordinal position) so that no candidate satisfies both;
the target's observed value for the changed condition is the true relation.  Items use matrix_pre (source "spatial")
like train.gme_export.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from train import data as D

VG = Path.home() / "vlmg-data/raw/visual_genome"
OUT = D.TRAIN_ROOT / "spatial_items.jsonl"
STUFF = {"sky", "wall", "ground", "grass", "street", "road", "floor", "water", "field", "background", "shadow", "line", "ceiling",
         "snow", "sand", "dirt", "air", "cloud", "sidewalk", "pavement", "surface", "side", "part", "edge", "top", "bottom", "front",
         "back", "area", "spot", "light", "reflection", "shade", "land", "band", "soil", "hill", "mountain", "ocean", "beach", "lake",
         "river", "wave", "foot", "toe", "knee", "neck", "shoulder", "lip", "tooth", "body", "skin", "tail", "tree", "leaf", "branch",
         "window", "letter", "writing", "word", "number", "stripe", "design", "pattern", "hair", "hand", "arm", "leg", "head", "face",
         "eye", "ear", "nose", "mouth", "finger", "clothing", "footwear", "plant", "food", "building"}
ORD = ["first", "second", "third", "fourth", "fifth"]
REL_TXT = {"left": "is to the left of the {}", "right": "is to the right of the {}", "above": "is above the {}", "below": "is below the {}"}
OPP = {"left": "right", "right": "left", "above": "below", "below": "above"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def inside(a, b, tol=0.9):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy >= tol * (a[2] - a[0]) * (a[3] - a[1])


def rel_verdict(c, r, side, W, H):
    """yes / no / None (ambiguous) for candidate box c being `side` of reference box r."""
    mx, my = 0.02 * W, 0.02 * H
    cx, cy, rx, ry = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2, (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
    if side == "left":
        return "yes" if c[2] <= r[0] + mx else "no" if cx >= rx else None
    if side == "right":
        return "yes" if c[0] >= r[2] - mx else "no" if cx <= rx else None
    if side == "above":
        return "yes" if c[3] <= r[1] + my else "no" if cy >= ry else None
    return "yes" if c[1] >= r[3] - my else "no" if cy <= ry else None  # below


def ordinal_text(k: int, n: int, axis: str, name: str) -> str:
    if axis == "x":
        if n == 2:
            return f"is the {name} on the {'left' if k == 0 else 'right'}"
        if k == n - 1:
            return f"is the rightmost {name}"
        return f"is the leftmost {name}" if k == 0 else f"is the {ORD[k]} {name} from the left"
    if n == 2:
        return f"is the upper {name}" if k == 0 else f"is the lower {name}"
    return f"is the {name} nearest the top" if k == 0 else f"is the {name} nearest the bottom" if k == n - 1 else f"is the {ORD[k]} {name} from the top"


def phrase(name: str, cs: list[dict]) -> str:
    """Referring expression from two conditions: an ordinal becomes the noun phrase ("The second car from the left that is
    below the lamp."), otherwise "The car that is ... and is ..."."""
    o = next((c for c in cs if c["text"].startswith("is the ")), None)
    if o is not None:
        np_ = o["text"][3:]
        return f"{np_[0].upper()}{np_[1:]} that " + " and ".join(c["text"] for c in cs if c is not o) + "."
    return f"The {name} that " + " and ".join(c["text"] for c in cs) + "."


def scene_conditions(cands, refs, W, H, name):
    """All usable conditions of a scene: dicts with kind, text, verdicts / seen per candidate (and the opposite for relations)."""
    n = len(cands)
    out = []
    for axis, lo, hi, dim in (("x", 0, 2, W), ("y", 1, 3, H)):
        cen = [(b[lo] + b[hi]) / 2 for b in cands]
        order = sorted(range(n), key=lambda i: cen[i])
        if min(cen[order[i + 1]] - cen[order[i]] for i in range(n - 1)) < 0.05 * dim:
            continue
        rank = {ci: k for k, ci in enumerate(order)}
        for k in range(n):
            out.append({"kind": "ord", "axis": axis, "k": k, "text": ordinal_text(k, n, axis, name),
                        "verdicts": ["yes" if rank[i] == k else "no" for i in range(n)],
                        "seen": [ordinal_text(rank[i], n, axis, name) for i in range(n)]})
    for rname, rb in refs:
        for side in ("left", "right", "above", "below"):
            vs = [rel_verdict(c, rb, side, W, H) for c in cands]
            ops = [rel_verdict(c, rb, OPP[side], W, H) for c in cands]
            if None in vs:
                continue
            seen = [REL_TXT[side].format(rname) if v == "yes" else REL_TXT[OPP[side]].format(rname) if o == "yes" else
                    f"is not {REL_TXT[side].format(rname)[3:]}" for v, o in zip(vs, ops)]
            out.append({"kind": "rel", "side": side, "ref": rname, "text": REL_TXT[side].format(rname), "verdicts": vs, "seen": seen,
                        "opp": {"text": REL_TXT[OPP[side]].format(rname), "verdicts": ops} if None not in ops else None})
    return out


def make_pair(cands, refs, W, H, name, rng, base, key) -> tuple[list[dict], str] | None:
    """One positive and one negative item for a candidate set, or None when no target has a usable condition pair."""
    conds = scene_conditions(cands, refs, W, H, name)
    n = len(cands)
    for t in rng.sample(range(n), n):
        pool = [c for c in conds if c["verdicts"][t] == "yes"]
        pairs = [(a, b) for i, a in enumerate(pool) for b in pool[i + 1:]
                 if not (a["kind"] == b["kind"] == "ord" and a["axis"] == b["axis"]) and "rel" in (a["kind"], b["kind"])]
        rng.shuffle(pairs)
        if rng.random() < 0.5:  # half the time try two-relation pairs first (both conditions needed)
            pairs.sort(key=lambda ab: ab[0]["kind"] != "rel" or ab[1]["kind"] != "rel")
        for a, b in pairs:
            sat = [i for i in range(n) if a["verdicts"][i] == "yes" and b["verdicts"][i] == "yes"]
            if sat != [t] or all(x["verdicts"].count("yes") == 1 for x in (a, b)):
                continue
            flips = []
            for which, keep in ((a, b), (b, a)):
                if which["kind"] == "rel":
                    if which["opp"] is not None:  # the opposite relation must be clear on every candidate to be a verifiable negative
                        flips.append((which, keep, which["opp"]["text"], which["opp"]["verdicts"]))
                else:
                    flips += [(which, keep, c["text"], c["verdicts"]) for c in conds
                              if c["kind"] == "ord" and c["axis"] == which["axis"] and c["k"] != which["k"]]
            rng.shuffle(flips)
            neg = next(((w, k, txt, vs) for w, k, txt, vs in flips if not any(vs[i] == "yes" and k["verdicts"][i] == "yes" for i in range(n))), None)
            if neg is None:
                continue

            def rows(cs):
                return [{"iid": i + 1, "box": cands[i], "verdicts": [c["verdicts"][i] for c in cs],
                         "seen": {str(j): c["seen"][i] for j, c in enumerate(cs)}} for i in range(n)]

            conds_p = [a, b]
            pos = {**base, "id": f"spatial:{key}:{t + 1}:pos", "kind": "positive", "expression": phrase(name, conds_p),
                   "answer": {"bbox_2d": cands[t]}, "target_iid": t + 1,
                   "matrix_pre": {"conditions": [c["text"] for c in conds_p], "rows": rows(conds_p), "answer_iid": t + 1,
                                  "flipped_idx": None, "flipped_clause": None, "first_iid": t + 1, "flip_from": None, "flip_to": None}}
            w, _, txt, vs = neg
            conds_n = [{"text": txt, "verdicts": vs, "seen": w["seen"]} if c is w else c for c in conds_p]
            k_idx = conds_p.index(w)
            negi = {**base, "id": f"spatial:{key}:{t + 1}:neg", "kind": "negative", "expression": phrase(name, conds_n),
                    "answer": {"bbox_2d": None}, "target_iid": None,
                    "matrix_pre": {"conditions": [c["text"] for c in conds_n], "rows": rows(conds_n), "answer_iid": None,
                                   "flipped_idx": k_idx, "flipped_clause": txt, "first_iid": t + 1, "flip_from": w["seen"][t],
                                   "flip_to": txt, "seen_pre": {str(k_idx): w["seen"][t]}}}
            return [pos, negi], w["kind"]
    return None


def oi_scenes(args, rng):
    """(key, image, W, H, name, candidate boxes, refs) from the OpenImages bank with the datagen candidate hygiene."""
    import glob

    import pyarrow.parquet as pq

    from datagen import common as C
    from datagen import select as S
    from shared import paths

    bank = S.load_bank()
    others = defaultdict(list)  # OpenImages-labelled objects of other classes, for references
    for shard in sorted(glob.glob(str(paths.PREPARED / "openimages_pool" / "instances" / "shard-*.parquet"))):
        t = pq.read_table(shard, columns=["image_id", "source", "label", "area_frac", "bbox_xyxy_px"])
        c = {k: t.column(k).to_pylist() for k in ("image_id", "source", "label", "area_frac", "bbox_xyxy_px")}
        for i in range(t.num_rows):
            if c["source"][i] == "labelled_other_class" and c["label"][i]:
                others[c["image_id"][i]].append((c["label"][i], float(c["area_frac"][i]), [float(v) for v in c["bbox_xyxy_px"][i]]))
    _, val = D.split(D.load_items())
    val_groups = {it["group"] for it in val}
    ids = sorted(bank)
    rng.shuffle(ids)
    for image_id in ids:
        if image_id in val_groups:
            continue
        sc = S.build_scene(image_id, bank[image_id])
        if sc is None:
            continue
        name = sc["category"]
        if name in STUFF or not args.min_cands <= sc["n_candidates"] <= 5:
            continue
        W, H = sc["image_wh"]
        cands = [inst["box"] for inst in sorted(sc["instances"], key=lambda x: x["box"][0])]
        labels = Counter(lb for lb, _, _ in others.get(image_id, []))
        refs = [(C.head_noun(lb), b) for lb, a, b in others.get(image_id, []) if labels[lb] == 1 and 0.005 <= a <= 0.5
                and C.head_noun(lb) not in STUFF and C.head_noun(lb) != name and all(iou(b, c) < 0.3 for c in cands)][:4]
        yield image_id, str(C.image_path(image_id)), W, H, name, cands, refs


PERSON = {"person", "man", "woman", "boy", "girl"}
PARENT = {"vehicle", "land vehicle", "animal", "mammal", "clothing", "footwear", "plant", "food", "furniture", "building",
          "tree", "house", "tire", "wheel", "fashion accessory", "sports equipment", "toy", "tableware", "kitchenware", "auto part",
          "vehicle registration plate", "human face", "human hair", "human hand", "human arm", "human leg", "human head",
          "human eye", "human ear", "human nose", "human mouth", "human body", "human beard", "human foot", "window", "girl", "boy",
          "home appliance", "weapon", "cabinetry", "person"}


def oi_raw_scenes(args, rng):
    """OpenImages validation boxes (human-verified, every instance of an image's positive classes is boxed) for the local
    images: a class with 3-5 clean boxes is the candidate set (person-like classes merged into "person" and de-duplicated),
    references are classes boxed once.  Group, depiction and inside boxes excluded; own validation scenes excluded."""
    import csv

    from PIL import Image

    root = Path.home() / "vlmg-data/raw/openimages"
    names = {r[0]: r[1].strip().lower() for r in csv.reader(open(root / "class-descriptions-boxable.csv", encoding="utf-8"))}
    local = {p.stem: p for p in (root / "images").glob("*.jpg")}
    boxes = defaultdict(list)
    with open(root / "validation-annotations-bbox.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["ImageID"] in local:
                boxes[r["ImageID"]].append(r)
    _, val = D.split(D.load_items())
    val_groups = {it["group"] for it in val}
    ids = sorted(boxes)
    rng.shuffle(ids)
    for image_id in ids:
        if image_id in val_groups:
            continue
        rs = boxes[image_id]
        try:
            with Image.open(local[image_id]) as im:
                W, H = im.size
        except Exception:
            continue
        by = defaultdict(list)
        spoiled = set()  # a class with a crowd / depiction / inside box is not exhaustively boxed as separate objects
        for r in rs:
            nm = names.get(r["LabelName"], "")
            nm = "person" if nm in PERSON else nm
            if r["IsGroupOf"] == "1" or r["IsDepiction"] == "1" or r["IsInside"] == "1":
                spoiled.add(nm)
        for r in rs:
            nm = names.get(r["LabelName"], "")
            nm = "person" if nm in PERSON else nm
            if not nm or nm in spoiled or (nm in PARENT and nm != "person") or nm in STUFF:
                continue
            b = [float(r["XMin"]) * W, float(r["YMin"]) * H, float(r["XMax"]) * W, float(r["YMax"]) * H]
            if all(iou(b, k) < 0.6 for k in by[nm]):  # the same person boxed as Man and Person
                by[nm].append(b)
        area = W * H

        def clean(bs):
            return (args.min_cands <= len(bs) <= 5 and all(0.005 * area <= (b[2] - b[0]) * (b[3] - b[1]) <= 0.5 * area for b in bs)
                    and all(iou(a, b) < 0.3 and not inside(a, b) and not inside(b, a) for i, a in enumerate(bs) for b in bs[i + 1:]))

        cats = [nm for nm, bs in by.items() if clean(bs)]
        if not cats:
            continue
        name = rng.choice(cats)
        cands = sorted(by[name], key=lambda b: b[0])
        refs = [(nm, bs[0]) for nm, bs in by.items() if nm != name and len(bs) == 1
                and 0.005 * area <= (bs[0][2] - bs[0][0]) * (bs[0][3] - bs[0][1]) <= 0.5 * area and all(iou(bs[0], c) < 0.3 for c in cands)][:4]
        yield image_id, str(local[image_id]), W, H, name, cands, refs


def vg_scenes(args, rng):
    meta = {m["image_id"]: m for m in json.load(open(VG / "annotations/image_data.json"))}
    excl = set(json.load(open(VG / "EVAL_OVERLAP_image_ids.json")))
    extra = VG / "EVAL_OVERLAP_prbench_openref.json"
    if extra.is_file():
        for hits in json.load(open(extra)).values():
            excl |= {int(Path(v).stem) for _, v in hits}
    objs = json.load(open(VG / "annotations/objects.json"))
    freq = Counter(o["synsets"][0] for rec in objs for o in rec["objects"] if o.get("synsets"))
    shown = defaultdict(Counter)
    for rec in objs:
        for o in rec["objects"]:
            if o.get("synsets") and o.get("names") and freq[o["synsets"][0]] >= args.min_freq:
                shown[o["synsets"][0]][o["names"][0].strip().lower()] += 1
    display = {s: c.most_common(1)[0][0] for s, c in shown.items()}
    common = {s for s in display if display[s] not in STUFF}
    rng.shuffle(objs)
    for rec in objs:
        iid = rec["image_id"]
        m = meta.get(iid)
        if m is None or m.get("coco_id") or iid in excl:
            continue
        W, H = m["width"], m["height"]
        img = VG / "images" / ("VG_100K_2" if "VG_100K_2" in m["url"] else "VG_100K") / f"{iid}.jpg"
        if not img.is_file():
            continue
        by = defaultdict(list)
        for o in rec["objects"]:
            if o.get("synsets"):
                by[o["synsets"][0]].append([float(o["x"]), float(o["y"]), float(o["x"] + o["w"]), float(o["y"] + o["h"])])
        area = W * H

        def clean(bs):
            return (args.min_cands <= len(bs) <= 5 and all(0.005 * area <= (b[2] - b[0]) * (b[3] - b[1]) <= 0.5 * area for b in bs)
                    and all(iou(a, b) < 0.3 and not inside(a, b) and not inside(b, a) for i, a in enumerate(bs) for b in bs[i + 1:]))

        cats = [s for s, bs in by.items() if s in common and clean(bs)]
        if not cats:
            continue
        syn = rng.choice(cats)
        cands = sorted(by[syn], key=lambda b: b[0])
        refs = [(display[s], bs[0]) for s, bs in by.items() if len(bs) == 1 and s in common and s != syn
                and 0.005 * area <= (bs[0][2] - bs[0][0]) * (bs[0][3] - bs[0][1]) <= 0.5 * area and all(iou(bs[0], c) < 0.3 for c in cands)][:4]
        yield f"vg{iid}", str(img), W, H, display[syn], cands, refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["oiraw", "oi", "vg"], default="oiraw")
    ap.add_argument("--n-scenes", type=int, default=400)
    ap.add_argument("--min-cands", type=int, default=3, help="minimum same-category candidates per scene")
    ap.add_argument("--min-freq", type=int, default=2000, help="vg: category / reference synsets need this many boxes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    st, items = Counter(), []
    gen = {"oiraw": oi_raw_scenes, "oi": oi_scenes, "vg": vg_scenes}[args.source]
    for key, img, W, H, name, cands, refs in gen(args, rng):
        if st["scenes"] >= args.n_scenes:
            break
        base = {"image": img, "image_wh": [W, H], "group": f"spatial:{key}", "category": name, "label": name, "run": f"spatial_{args.source}",
                "source": "spatial", "n_candidates": len(cands)}
        got = make_pair(cands, refs, W, H, name, rng, base, key)
        if got is None:
            st["no_pair"] += 1
            continue
        items += got[0]
        st["scenes"] += 1
        st["neg_" + got[1]] += 1
        st[f"n{len(cands)}"] += 1
    out = Path(args.out).expanduser()
    with open(out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(dict(st), f"-> {len(items)} items -> {out}", flush=True)


if __name__ == "__main__":
    main()
