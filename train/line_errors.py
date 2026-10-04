"""Per-line audit errors on training scenes (does on-policy correction have a signal to learn?).

    python -m train.eval_suite --tag <tag> --adapter <dir> --trace rat --turns 2 --audit coa --sets trainprobe ...
    python -m train.line_errors <tag>

The probe set is a seeded sample of RL training items that carry a label matrix (GME-style, spatial, own data).  For every
candidate the model proposed itself that matches a labelled instance (IoU >= 0.7), each audit line is aligned to a label
clause (train.traces.align on the model's condition text) and its verdict compared with the label: false accusation
(mismatch on a true clause), sycophancy (match on a false clause), unsure; broken down by data source, clause type and
whether the candidate is the described instance or a distractor.  Lines of unmatched candidates and "unclear" labels are
not counted.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from shared.harness import parsers as P
from train import coa as COA
from train import data as D
from train import traces as TR

SOURCES = (("gmestyle", D.TRAIN_ROOT / "gme_style_train.jsonl", 120), ("spatial", D.TRAIN_ROOT / "spatial_items.jsonl", 90), ("own", None, 90))
CATS = [("relation", r"\b(left|right|behind|front|next to|beside|near|between|above|below|under|over|on top|closest|farthest|leftmost|rightmost|"
                      r"middle|center|background|foreground|nearest|upper|lower)\b"),
        ("text", r"(\"|'|\btext\b|\breads\b|\bwritten\b|\blogo\b|\bsign\b|\bletter|\bnumber\b)"),
        ("count/order", r"\b(\d+|one|two|three|four|five|first|second|third|fourth|row|pair)\b"),
        ("color", r"\b(red|blue|green|yellow|white|black|gray|grey|brown|orange|pink|purple|silver|gold|dark|light|beige|tan)\b"),
        ("action/pose", r"\b(holding|wearing|sitting|standing|walking|looking|carrying|riding|lying|leaning|hanging|raised|bent)\b")]


def cat(s: str) -> str:
    s = s.lower()
    return next((n for n, p in CATS if re.search(p, s)), "other")


def _sources():
    rng = random.Random(0)
    out = []
    for name, path, n in SOURCES:
        if path is None:
            train, _ = D.split(D.load_items())
            its = [it for it in train if it.get("kind") in ("positive", "negative")]
        else:
            its = D.load_items(path)
        its = [it for it in its if TR.label_matrix(it)[0] is not None]
        rng.shuffle(its)
        out += [(name, it) for it in its[:n]]
    return out


def probe_items() -> list[dict]:
    items = []
    for name, it in _sources():
        box = (it.get("answer") or {}).get("bbox_2d")
        items.append({"id": f"probe:{it['id']}", "expr": it["expression"], "image": it["image"], "gt_boxes": [box] if box else [],
                      "n_gt": int(box is not None), "kind": it["kind"], "set": name})
    return items


def main(tag: str) -> None:
    src = {f"probe:{it['id']}": (name, it) for name, it in _sources()}
    rows = [json.loads(l) for l in open(D.TRAIN_ROOT / "eval" / tag / "trainprobe.jsonl") if l.strip()]
    analyze(rows, src, tag)


def analyze(rows: list[dict], src: dict, tag: str) -> None:
    c = defaultdict(lambda: defaultdict(int))
    unmatched = cands = 0
    for r in rows:
        name, it = src.get(r["id"], (None, None))
        if it is None or r.get("coa") is None:
            continue
        m, _ = TR.label_matrix(it)
        if m is None:
            continue
        with Image.open(it["image"]) as im:
            wh = im.size
        raw = r["raw"]
        i = raw.find("<coa>")
        pr = TR.parse(raw[: len(raw) if i < 0 else i], wh)
        conds = pr["conditions"]
        boxes = [cd["box"] for cd in pr["candidates"] if cd.get("box")]
        audits = [COA.parse_audit(t) for t in r["coa"]["audits"]]
        target = m.get("first_iid") or m.get("answer_iid")
        for k, b in enumerate(boxes):
            if k >= len(audits) or not audits[k]["format_ok"]:
                continue
            cands += 1
            best = max(m["rows"], key=lambda row: P.iou(tuple(b), tuple(row["box"])))
            if P.iou(tuple(b), tuple(best["box"])) < 0.7:
                unmatched += 1
                continue
            role = "described" if best["iid"] == target else "distractor"
            for ln in audits[k]["lines"]:
                j = ln["idx"]
                if not 1 <= j <= len(conds):
                    continue
                li = TR.align(m["conditions"], conds[j - 1])
                if li is None or li > len(best["verdicts"]):
                    continue
                lab = best["verdicts"][li - 1]
                if lab not in ("yes", "no"):
                    continue
                key = (name, cat(m["conditions"][li - 1]), role, lab)
                c[key]["n"] += 1
                c[key][ln["verdict"]] += 1
    print(f"[{tag}] items {len(rows)}, candidates {cands}, unmatched to a labelled instance {unmatched}")
    print("source    clause       candidate   label | n     wrong   unsure   (wrong = mismatch on a true clause / match on a false one)")
    for key in sorted(c):
        name, ct, role, lab = key
        d = c[key]
        wrong = d["mismatch"] if lab == "yes" else d["match"]
        print(f"{name:9s} {ct:12s} {role:11s} {lab:5s} | {d['n']:5d} {100 * wrong / d['n']:6.1f}% {100 * d['unsure'] / d['n']:6.1f}%")


if __name__ == "__main__":
    main(sys.argv[1])
