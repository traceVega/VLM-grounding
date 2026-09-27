"""One-line-per-run scoreboard over evaluation tags (screening or dev): GME rejection / positives /
positive null / net, the three positive dimensions, own negatives / positives, RefCOCO, and the
candidate recall of two-turn runs.

    python -m train.scoreboard [tag ...]        (default: every eval dir with a summary.json, newest last)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from train import data as D

EVAL = D.TRAIN_ROOT / "eval"


def recall(tag: str) -> str:
    p = EVAL / tag / "gme.jsonl"
    if not p.is_file():
        return "-"
    pos = [r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip()) if r.get("dimension") != "Rejection"]
    if not pos or "cand_recall_iou" not in pos[0]:
        return "-"
    hit = sum(1 for r in pos if (r.get("cand_recall_iou") or 0) >= 0.5)
    return f"{100 * hit / len(pos):.0f}"


def row(tag: str) -> str:
    s = json.load(open(EVAL / tag / "summary.json", encoding="utf-8"))
    g, o, rc = s.get("gme") or {}, s.get("own") or {}, s.get("refcoco") or {}
    f = lambda x: "-" if x is None else f"{100 * x:.1f}"
    dims = g.get("by_dimension") or {}
    ref = "/".join(f"{100 * (rc.get(k) or {}).get('acc', float('nan')):.0f}" for k in ("RefCOCO", "RefCOCOplus", "RefCOCOg")) if rc else "-"  # val / + / g
    return (f"{tag:26s} rej {f(g.get('rejection_acc')):>5} pos {f(g.get('positive_acc')):>5} null {f(g.get('positive_null_rate')):>5} net {f(g.get('net_rejection')):>5} | "
            f"D/L/S {f(dims.get('Discriminative')):>5}/{f(dims.get('Limited')):>5}/{f(dims.get('Spatial')):>5} | recall {recall(tag):>3} | "
            f"own neg {f(o.get('neg_null_rate')):>5} pos {f(o.get('pos_acc')):>5} | refcoco {ref}")


def main() -> None:
    tags = sys.argv[1:] or sorted((p.parent.name for p in EVAL.glob("*/summary.json")), key=lambda t: (EVAL / t / "summary.json").stat().st_mtime)
    for t in tags:
        try:
            print(row(t))
        except Exception as e:  # noqa: BLE001
            print(f"{t:26s} ({e})")


if __name__ == "__main__":
    main()
