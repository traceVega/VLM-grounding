"""Answer-Audit-Veto (notes/DESIGN-AAV-2026-10-01.md): the pieces shared by the RL loop and the evaluation.

    python -m train.eval_suite --tag base_rlpool --sets rlpool --batch 8       # the base model's direct answers on the RL pool
    python -m train.grpo_lora ... --c1-from base_rlpool --r-line 0.2           # those answers as candidate 1, per-line rewards

pool_items()    every item of the RL pool (own scenes + the --extra files the COA runs draw from) in the eval_suite format,
                so the base model's direct answer can be computed once and injected as candidate 1 during training
load_c1(tags)   item id -> that answer's box in pixels (null answers are not injected)
inject_c1()     candidate list with c1 first and the model's own proposals after it (IoU >= 0.7 duplicates of c1 dropped)
line_rewards()  per audited candidate that matches a labelled instance (IoU >= 0.7): the mean over its label-aligned lines of
                +1 (match on a true clause / named mismatch on a false one), -1 (named mismatch on a true clause / match on a
                false one), 0 (unsure, or a mismatch without a named value); "unclear" labels are not counted.  Only rows
                whose labels hold by construction are used: the described instance (the expression was written about it; a
                negative's falsified clause is false of it), every row of the box-geometry data (exact_rows), and the
                ground-truth box of answer-only positives (RefCOCO replay: all lines true).  Distractor rows of the other
                sources carry single-checker labels and are left to the outcome reward.
"""

from __future__ import annotations

import json

from shared.harness import parsers as P
from train import coa as COA
from train import data as D
from train import traces as TR

POOL = (("own", None), ("cross", "augment/cross_v1.jsonl"), ("alt", "alt_negatives.jsonl"), ("long", "long_items.jsonl"),
        ("gmestyle", "gme_style_train.jsonl"), ("refcoco", "refcoco_train_v2.jsonl"), ("spatial", "spatial_items.jsonl"))


def pool_items() -> list[dict]:
    out = []
    for name, f in POOL:
        its = D.load_items() if f is None else D.load_items(D.TRAIN_ROOT / f)
        for it in its:
            box = (it.get("answer") or {}).get("bbox_2d")
            out.append({"id": it["id"], "expr": it["expression"], "image": it["image"], "image_wh": it["image_wh"],
                        "gt_boxes": [box] if box else [], "n_gt": int(box is not None), "kind": it["kind"], "set": name})
    return out


def load_c1(tags: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for tag in [t for t in tags.split(",") if t]:
        for f in sorted((D.TRAIN_ROOT / "eval" / tag).glob("*.jsonl")):
            if f.stem in ("gmegray", "gray"):  # gray-image controls share ids with the real sets
                continue
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r.get("output_type") == "box" and r.get("box"):
                    out.setdefault(r["id"], list(r["box"]))
    return out


def inject_c1(boxes: list, c1, k_max: int) -> list:
    return ([list(c1)] + [b for b in boxes if P.iou(tuple(b), tuple(c1)) < 0.7])[:k_max]


def line_rewards(t1: str, boxes: list, audits: list[dict], m: dict | None, conditions_of, gt_box=None, exact_rows: bool = False) -> dict[int, tuple[int, float, int]]:
    """candidate index -> (labelled instance id, mean line score in [-1, 1], number of label-aligned lines).
    Without a label matrix, a positive's ground-truth box still labels its own lines: the expression describes that
    object, so every condition the model derived from it holds there (a named mismatch on it is a false accusation)."""
    conds = conditions_of(t1)
    if not conds:
        return {}
    if (not m or not m.get("rows")) and gt_box is not None:
        m = {"conditions": list(conds), "rows": [{"iid": "gt", "box": list(gt_box), "verdicts": ["yes"] * len(conds)}], "answer_iid": "gt"}
    if not m or not m.get("rows"):
        return {}
    trusted = m.get("answer_iid") if m.get("answer_iid") is not None else m.get("first_iid")  # the described instance
    lab_of = [TR.align(m["conditions"], c) for c in conds]  # model condition j (0-based) -> 1-based label clause or None
    out = {}
    for k, (b, a) in enumerate(zip(boxes, audits)):
        if not a.get("format_ok"):
            continue
        row = max(m["rows"], key=lambda r: P.iou(tuple(b), tuple(r["box"])))
        if P.iou(tuple(b), tuple(row["box"])) < 0.7 or not (exact_rows or (trusted is not None and row["iid"] == trusted)):
            continue
        named = {id(ln) for ln in COA.named_mismatches(a["lines"])}
        s, n = 0, 0
        for ln in a["lines"]:
            j = ln["idx"]
            if not 1 <= j <= len(conds) or lab_of[j - 1] is None or lab_of[j - 1] > len(row["verdicts"]):
                continue
            lab = row["verdicts"][lab_of[j - 1] - 1]
            if lab not in ("yes", "no"):
                continue
            n += 1
            if lab == "yes":
                s += 1 if ln["verdict"] == "match" else (-1 if id(ln) in named else 0)
            else:
                s += 1 if id(ln) in named else (-1 if ln["verdict"] == "match" else 0)
        if n:
            out[k] = (row["iid"], s / n, n)
    return out
