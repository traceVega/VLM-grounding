"""Score-based decisions recomputed offline from stored verdict probabilities (eval_suite --verdict-probs).

    python -m train.score_rules <tag>[+<tag>...] [...] [--base base4b,prdev_base4b]

Every audited candidate gets a pass probability, prod over its lines of (1 - p(mismatch)); candidates without parsed lines get 0.
Decision rules, for a threshold t:
  first   the first candidate whose pass probability is >= t, else no object (order has priority, the threshold replaces the hard veto)
  best    the candidate with the highest pass probability if it is >= t, else no object (selection by score)
The table gives, per threshold, the positives' accuracy (RefCOCO / + / g, PR-Bench dev positives, GME positives) and the rejection
accuracy (PR-Bench dev Reject, GME Rejection), next to the run's recorded answer (hard rule) and the base model; the last block
gives, for each rule, the best rejection whose positives stay within 1 and within 3 points of the base on the same benchmark.
"""

from __future__ import annotations

import argparse
import json
import math

from shared.harness import parsers as P
from train import data as D

GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
COLS = ("RefCOCO", "RefCOCOplus", "RefCOCOg", "pr_pos", "gme_pos", "pr_rej", "gme_rej")


def load(tags: str, name: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tag in [t for t in tags.replace(",", "+").split("+") if t]:
        f = D.TRAIN_ROOT / "eval" / tag / f"{name}.jsonl"
        if f.is_file():
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    out.setdefault(r["id"], r)
    return out


def gts(name: str) -> dict[str, list]:
    if name == "gme":
        return {it["id"]: it["gt_boxes"] for it in D.gme_items()}
    if name == "refcoco":
        return {it["id"]: it["gt_boxes"] for it in D.refcoco_items(300, 0)}
    from train import ood_items as OOD

    return {it["id"]: it["gt_boxes"] for it in OOD.items("prbench_dev")}


def group(name: str, r: dict) -> str:
    if name == "gme":
        return "gme_rej" if r.get("dimension") == "Rejection" else "gme_pos"
    if name == "refcoco":
        return r.get("set")
    return "pr_rej" if r.get("set") == "reject" else "pr_pos"


def pass_probs(r: dict) -> list[float] | None:
    coa = r.get("coa") or {}
    if not coa.get("probs") or coa.get("boxes") is None:
        return None
    return [math.prod(1.0 - lp[1] for lp in pr) if pr else 0.0 for pr in coa["probs"]]


def decide(pp: list[float], boxes: list, rule: str, t: float):
    if not pp:
        return None
    if rule == "first":
        return next((b for p, b in zip(pp, boxes) if p >= t), None)
    k = max(range(len(pp)), key=lambda i: (pp[i], -i))
    return boxes[k] if pp[k] >= t else None


def ok(ans, gt: list) -> bool:
    return (ans is None) if not gt else (ans is not None and max(P.iou(tuple(ans), tuple(g)) for g in gt) >= 0.5)


def analyse(tags: str, base_tags: str | None) -> None:
    rows = {}  # group -> list of (pass probs, boxes, gt, recorded correct, base correct)
    for name in ("refcoco", "prbench_dev", "gme"):
        recs = load(tags, name)
        if not recs:
            continue
        gt = gts(name)
        base = load(base_tags, name) if base_tags else {}
        for i, r in recs.items():
            pp = pass_probs(r)
            if pp is None or i not in gt:
                continue
            rows.setdefault(group(name, r), []).append((pp, r["coa"]["boxes"], gt[i], r["correct"], base.get(i, {}).get("correct")))
    cols = [c for c in COLS if c in rows]
    if not cols:
        print(f"[{tags}] no records with verdict probabilities")
        return
    acc = lambda g, f: 100 * sum(f(x) for x in rows[g]) / len(rows[g])
    print(f"[{tags}]  n: " + ", ".join(f"{c} {len(rows[c])}" for c in cols))
    print(f"  {'rule':14s} " + " ".join(f"{c[:11]:>11s}" for c in cols))
    if base_tags:
        print(f"  {'base':14s} " + " ".join(f"{acc(c, lambda x: bool(x[4])):11.1f}" for c in cols))
    print(f"  {'recorded':14s} " + " ".join(f"{acc(c, lambda x: x[3]):11.1f}" for c in cols))
    table = {}
    for rule in ("first", "best"):
        for t in GRID:
            table[(rule, t)] = {c: acc(c, lambda x: ok(decide(x[0], x[1], rule, t), x[2])) for c in cols}
            print(f"  {rule:5s} t={t:<6} " + " ".join(f"{table[(rule, t)][c]:11.1f}" for c in cols))
    # oracle: some audited candidate is correct (positives) - the ceiling of selection
    print(f"  {'oracle select':14s} " + " ".join((f"{acc(c, lambda x: any(ok(b, x[2]) for b in x[1])):11.1f}" if not c.endswith('rej') else f"{'-':>11s}") for c in cols))
    if base_tags:
        for pos_c, rej_c in (("gme_pos", "gme_rej"), ("pr_pos", "pr_rej")):
            if pos_c in rows and rej_c in rows:
                b = acc(pos_c, lambda x: bool(x[4]))
                for rule in ("first", "best"):
                    for margin in (1, 3):
                        cand = [(table[(rule, t)][rej_c], t, table[(rule, t)][pos_c]) for t in GRID if table[(rule, t)][pos_c] >= b - margin]
                        best = max(cand) if cand else None
                        print(f"  {rule:5s} {rej_c}: best rejection with {pos_c} >= base - {margin} ({b - margin:.1f}): "
                              + (f"{best[0]:.1f} at t={best[1]} (positives {best[2]:.1f})" if best else "none on the grid"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--base", default="base4b,prdev_base4b")
    a = ap.parse_args()
    for t in a.tags:
        analyse(t, a.base)
