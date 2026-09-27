"""Soft null decisions from a trace evaluation that recorded verdict-cell probabilities
(train.eval_suite --trace --cell-probs): on the committed (first) candidate, reject iff the
largest p(no) over its conditions exceeds a threshold; otherwise answer its box.  Reports the
AUROC of that score between rejection items and positives, and the operating curve, on the
GME dev subset (and the threshold chosen on the own validation set when present).

    python -m train.soft_decision --tag scr_grpo_v3b_pair_cp
"""

from __future__ import annotations

import argparse
import json

from train import data as D
from train.crop_probe import auroc


def commit_score(r: dict) -> float | None:
    cp = r.get("cell_pno") or {}
    first = [k for k in cp if k.startswith("1:")]
    if not first:
        return None
    return max(cp[k]["no"] for k in first)


def curve(rows: list[dict], taus):
    rej = [r for r in rows if r["n_gt"] == 0]
    pos = [r for r in rows if r["n_gt"] > 0]
    out = []
    for tau in taus:
        def null(r):
            s = commit_score(r)
            return (s is not None and s > tau) or (s is None and r["output_type"] != "box")
        rej_acc = sum(null(r) for r in rej) / max(1, len(rej))
        pos_null = sum(null(r) for r in pos) / max(1, len(pos))
        pos_acc = sum((not null(r)) and r.get("commit_iou", 0.0) >= 0.5 for r in pos) / max(1, len(pos))
        out.append({"tau": tau, "rejection_acc": round(rej_acc, 4), "positive_null_rate": round(pos_null, 4),
                    "positive_acc": round(pos_acc, 4), "net_rejection": round(rej_acc - pos_null, 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    d = D.TRAIN_ROOT / "eval" / args.tag
    gme = {it["id"]: it for it in D.gme_items()}
    rows = [json.loads(l) for l in open(d / "gme.jsonl", encoding="utf-8") if l.strip()]
    from PIL import Image

    from shared.harness import parsers as P
    from train import traces as TR

    for r in rows:  # IoU of the committed candidate with the ground truth
        it = gme[r["id"]]
        pr = TR.parse(r["raw"], Image.open(it["image"]).size)
        c1 = pr["candidates"][0]["box"] if pr["candidates"] else None
        r["commit_iou"] = max((P.iou(tuple(c1), tuple(g)) for g in it["gt_boxes"]), default=0.0) if c1 else 0.0
    rej = [r for r in rows if r["n_gt"] == 0]
    pos = [r for r in rows if r["n_gt"] > 0]
    s_rej = [commit_score(r) for r in rej]
    s_pos = [commit_score(r) for r in pos]
    s_pos_ok = [commit_score(r) for r in pos if r["commit_iou"] >= 0.5]
    f = lambda xs: [x for x in xs if x is not None]
    summary = {"n": len(rows), "scored": sum(x is not None for x in s_rej + s_pos),
               "hard_rule": {"rejection_acc": round(sum(r["correct"] for r in rej) / max(1, len(rej)), 4),
                             "positive_acc": round(sum(r["correct"] for r in pos) / max(1, len(pos)), 4),
                             "positive_null_rate": round(sum(r["output_type"] in D.NULL_TYPES for r in pos) / max(1, len(pos)), 4)},
               "commit_recall_pos": round(sum(r["commit_iou"] >= 0.5 for r in pos) / max(1, len(pos)), 4),
               "auroc_rej_vs_pos": auroc(f(s_rej), f(s_pos)), "auroc_rej_vs_pos_correct_commit": auroc(f(s_rej), f(s_pos_ok)),
               "curve": curve(rows, [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])}
    (d / "soft_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
