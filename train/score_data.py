"""Training set of the supervised scoring stage (notes/DESIGN-SCORE-SELECT-2026-10-04.md): 6,000 items, 3,000 of them flagged
as the half-size subset with the same mix.  Local data only, labels from annotations, label tables and rules.

    python -m train.score_data [--n 6000] [--seed 0]   ->  $VLMG_DATA_ROOT/train/score_items.jsonl

One item = an expression, its candidate boxes and what is known about them:
  cands[k] = {"box", "pass": True / False, "role", "lines": None | "all_true" | {"false_word", "seen"} | {"verdicts": {clause: yes / no}}}
  answer   = index of the candidate that is the answer, or None (no candidate fits)
Sources and their labels:
  own      our scenes, cross-scene / alternative negatives, long, GME-style, spatial (train.traces.label_matrix): the described
           instance carries line labels, every row of the box-geometry data too; other rows only "must be vetoed"
  human    a RefCOCO / + / g train expression on its own object (every line true) plus a same-category sibling (must be vetoed)
  twin     the same kind of expression with one word flipped by rule (colour, left / right, top / bottom, front / back,
           ordinal) on the same object: the line carrying the flipped word is false, the original word is what is seen
Validation scenes and evaluation images are excluded (the human part reuses the cleaned replay images).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from train import data as D
from train import traces as TR

OUT = D.TRAIN_ROOT / "score_items.jsonl"
PAIRS = D.TRAIN_ROOT / "refcoco_pairs_v2.jsonl"  # python -m train.refcoco_pairs --per-image 4 --out <this> (keeps every caption of the target)
OWN_FILES = (("own", None), ("cross", "augment/cross_v1.jsonl"), ("alt", "alt_negatives.jsonl"), ("long", "long_items.jsonl"),
             ("gmestyle", "gme_style_train.jsonl"), ("spatial", "spatial_items.jsonl"))

COLOUR_SWAP = {"red": "blue", "blue": "red", "green": "red", "yellow": "blue", "purple": "yellow", "pink": "green", "brown": "blue",
               "black": "white", "white": "black", "gray": "red", "grey": "red"}
WORD_SWAP = {"left": "right", "right": "left", "top": "bottom", "bottom": "top", "upper": "lower", "lower": "upper",
             "front": "back", "back": "front", "first": "second", "second": "third", "third": "second", "1st": "2nd", "2nd": "3rd", "3rd": "2nd"}
EDIT_TYPE = {**{w: "colour" for w in COLOUR_SWAP}, "left": "left/right", "right": "left/right", "top": "vertical", "bottom": "vertical",
             "upper": "vertical", "lower": "vertical", "front": "depth", "back": "depth", "first": "ordinal", "second": "ordinal",
             "third": "ordinal", "1st": "ordinal", "2nd": "ordinal", "3rd": "ordinal"}
_SKIP = re.compile(r"black and white|white and black|back to|backs? of|right (?:hand|arm|leg|foot|side up|now)|left (?:hand|arm|leg|foot)|"
                   r"top of the|on top of|in front of", re.I)


def flip(expr: str, all_caps: list[str], rng: random.Random) -> dict | None:
    """One rule edit that makes the expression false of its own object: {"expression", "from", "to", "type"} or None.
    A colour is never replaced by a colour some caption of the same object uses; phrases whose flipped form is not a
    clean contradiction (body parts, 'black and white', 'in front of', 'on top of') are skipped."""
    if _SKIP.search(expr):
        return None
    words = re.findall(r"[A-Za-z0-9]+", expr)
    used = {w.lower() for c in all_caps for w in re.findall(r"[A-Za-z0-9]+", c)}
    options = []
    for w in words:
        lw = w.lower()
        to = COLOUR_SWAP.get(lw) or WORD_SWAP.get(lw)
        if to is None or (lw in COLOUR_SWAP and to in used):
            continue
        if len(re.findall(rf"\b{re.escape(w)}\b", expr)) != 1:  # the word has to be unique in the expression
            continue
        options.append((w, to))
    if not options:
        return None
    w, to = rng.choice(options)
    to_cased = to.upper() if w.isupper() else (to.capitalize() if w[0].isupper() else to)
    return {"expression": re.sub(rf"\b{re.escape(w)}\b", to_cased, expr, count=1), "from": w.lower(), "to": to, "type": EDIT_TYPE[w.lower()]}


def own_items() -> list[dict]:
    train, val = D.split(D.load_items(), 18, 0)
    val_groups = {it["group"] for it in val}
    out = []
    for name, f in OWN_FILES:
        its = train if f is None else [it for it in D.load_items(D.TRAIN_ROOT / f) if it["group"] not in val_groups and it.get("src_group") not in val_groups]
        for it in its:
            if it["kind"] not in ("positive", "negative"):
                continue
            m, _ = TR.label_matrix(it)
            if m is None:
                continue
            rows = TR.ordered_rows(m, "commit")[:4]
            described = m.get("answer_iid") if m.get("answer_iid") is not None else m.get("first_iid")
            cands, answer = [], None
            for k, r in enumerate(rows):
                exact = name == "spatial" or r["iid"] == described  # labels that hold by construction
                lines = {"verdicts": {c: v for c, v in zip(m["conditions"], r["verdicts"]) if v in ("yes", "no")}} if exact else None
                is_answer = m.get("answer_iid") is not None and r["iid"] == m["answer_iid"]
                if is_answer:
                    answer = k
                cands.append({"box": [round(float(v), 1) for v in r["box"]], "pass": bool(is_answer), "lines": lines,
                              "role": "described" if r["iid"] == described else "distractor"})
            if it["kind"] == "positive" and answer is None:
                continue  # the answer row fell outside the four kept rows
            out.append({"id": f"score:{it['id']}", "image": it["image"], "image_wh": it["image_wh"], "expression": it["expression"], "kind": it["kind"],
                        "source": "own:" + name, "cands": cands, "answer": answer})
    return out


def human_items(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """(positives with a sibling, rule-falsified twins) from the contrastive pair file (train.refcoco_pairs)."""
    pairs = D.load_items(PAIRS)
    pos, twins = [], []
    for p in pairs:
        a = [round(float(v), 1) for v in p["answer"]["bbox_2d"]]
        b = [round(float(v), 1) for v in p["neg_boxes"][0]]
        sub = p["id"].split(":")[1]
        base = {"image": p["image"], "image_wh": p["image_wh"]}
        pos.append({**base, "id": "score:" + p["id"], "expression": p["expression"], "kind": "positive", "source": "human:" + sub, "answer": 0,
                    "cands": [{"box": a, "pass": True, "lines": "all_true", "role": "target"}, {"box": b, "pass": False, "lines": None, "role": "sibling"}]})
        e = flip(p["expression"], p.get("captions") or [p["expression"]], rng)
        if e:
            twins.append({**base, "id": "score:" + p["id"] + ":twin", "expression": e["expression"], "kind": "negative", "source": "twin:" + sub, "answer": None,
                          "edit": {k: e[k] for k in ("from", "to", "type")}, "twin_of": "score:" + p["id"],
                          "cands": [{"box": a, "pass": False, "lines": {"false_word": e["to"], "seen": e["from"]}, "role": "target"}]})
    return pos, twins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    own = own_items()
    pos, twins = human_items(rng)
    rng.shuffle(own)
    n_own = len(own)
    own_pos, own_neg = sum(i["kind"] == "positive" for i in own), sum(i["kind"] == "negative" for i in own)
    rest = args.n - n_own
    n_pos = (rest + own_neg - own_pos) // 2  # positives and negatives balanced over the whole set
    n_twin = rest - n_pos
    # twins first (grouped by edit type, round robin over the datasets), then the positives: every twin's positive is kept
    by = {}
    for t in twins:
        by.setdefault((t["source"], t["edit"]["type"]), []).append(t)
    for v in by.values():
        rng.shuffle(v)
    keys = sorted(by)
    chosen_twins = []
    while len(chosen_twins) < n_twin and any(by[k] for k in keys):
        for k in keys:
            if by[k] and len(chosen_twins) < n_twin:
                chosen_twins.append(by[k].pop())
    need = {t["twin_of"] for t in chosen_twins}
    first = [p for p in pos if p["id"] in need]
    others = [p for p in pos if p["id"] not in need]
    rng.shuffle(others)
    chosen_pos = (first + others)[:n_pos]
    items = own + chosen_pos + chosen_twins
    # the half-size subset: every source halved (twins with their positives first)
    half = set()
    for src in sorted({i["source"] for i in items}):
        grp = [i for i in items if i["source"] == src]
        if src.startswith("human:"):
            grp.sort(key=lambda i: i["id"] not in need)
        half.update(i["id"] for i in grp[: (len(grp) + 1) // 2])
    for i in items:
        i["half"] = i["id"] in half
    rng.shuffle(items)
    with open(Path(args.out).expanduser(), "w", encoding="utf-8") as fh:
        for i in items:
            fh.write(json.dumps(i, ensure_ascii=False) + "\n")
    for name, sel in (("all", items), ("half", [i for i in items if i["half"]])):
        c = Counter(i["source"].split(":")[0] for i in sel)
        k = Counter(i["kind"] for i in sel)
        print(f"{name}: {len(sel)} items {dict(c)} | pos {k['positive']} neg {k['negative']} | candidates {sum(len(i['cands']) for i in sel)} | images {len({i['image'] for i in sel})}")
    print("twin edit types:", dict(Counter(i["edit"]["type"] for i in items if "edit" in i)), "| available twins", len(twins), "positives", len(pos), "own", n_own)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
