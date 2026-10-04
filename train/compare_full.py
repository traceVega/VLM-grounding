"""Full-set comparison of GroundingME runs with paired McNemar tests.

    python -m train.compare_full <tag> [<tag> ...] [--ref base4b,full_grpo_v3d_hint,full_grpo_v3e_long]

Per tag: Rejection accuracy (201), positive accuracy (804) and null rate, accuracy by dimension, and the GME
gray control (gmegray: null rate on Rejection vs positive items with a gray image; gap = rejection - positive).
Per tag x reference: exact two-sided McNemar on the discordant pairs, separately for Rejection, positives and
all items (only ids present in both runs).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EV = Path.home() / "vlmg-data/train/eval"


def load(tag: str, name: str) -> dict:
    f = EV / tag / f"{name}.jsonl"
    if not f.is_file():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in open(f) if l.strip())}


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided p for b vs c discordant pairs (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def stats(recs: dict) -> dict:
    rej = [r for r in recs.values() if r["dimension"] == "Rejection"]
    pos = [r for r in recs.values() if r["dimension"] != "Rejection"]
    out = {"n": len(recs), "rej": 100 * sum(r["correct"] for r in rej) / max(1, len(rej)), "rej_n": len(rej),
           "pos": 100 * sum(r["correct"] for r in pos) / max(1, len(pos)), "pos_n": len(pos),
           "pos_null": 100 * sum(r["output_type"] == "none" for r in pos) / max(1, len(pos))}
    for d in ("Discriminative", "Limited", "Spatial"):
        rs = [r for r in pos if r["dimension"] == d]
        out[d[0]] = 100 * sum(r["correct"] for r in rs) / max(1, len(rs))
    return out


def gray_gap(recs: dict) -> str:
    if not recs:
        return "gmegray: -"
    rej = [r for r in recs.values() if r["dimension"] == "Rejection"]
    pos = [r for r in recs.values() if r["dimension"] != "Rejection"]
    rn = 100 * sum(r["output_type"] == "none" for r in rej) / max(1, len(rej))
    pn = 100 * sum(r["output_type"] == "none" for r in pos) / max(1, len(pos))
    return f"gmegray null rej {rn:5.1f} pos {pn:5.1f} gap {rn - pn:+5.1f} (n {len(rej)}/{len(pos)})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--ref", default="base4b,full_grpo_v3d_hint,full_grpo_v3e_long")
    a = ap.parse_args()
    refs = [t for t in a.ref.split(",") if t]
    for t in refs + [t for t in a.tags if t not in refs]:
        g = load(t, "gme")
        if not g:
            print(f"{t:24s} (no gme.jsonl)")
            continue
        s = stats(g)
        print(f"{t:24s} n {s['n']:4d} | rej {s['rej']:5.1f} ({s['rej_n']}) pos {s['pos']:5.1f} ({s['pos_n']}) null {s['pos_null']:5.1f} "
              f"net {s['rej'] - s['pos_null']:5.1f} | D/L/S {s['D']:5.1f}/{s['L']:5.1f}/{s['S']:5.1f} | {gray_gap(load(t, 'gmegray'))}")
    for t in a.tags:
        g = load(t, "gme")
        for ref in refs:
            if ref == t:
                continue
            h = load(ref, "gme")
            ids = sorted(set(g) & set(h))
            parts = []
            for name, sel in (("rej", lambda r: r["dimension"] == "Rejection"), ("pos", lambda r: r["dimension"] != "Rejection"), ("all", lambda r: True)):
                ii = [i for i in ids if sel(g[i])]
                b = sum(g[i]["correct"] and not h[i]["correct"] for i in ii)
                c = sum(h[i]["correct"] and not g[i]["correct"] for i in ii)
                parts.append(f"{name} +{b}/-{c} p={mcnemar(b, c):.3g}")
            print(f"  {t} vs {ref} (n {len(ids)}): " + " | ".join(parts))


if __name__ == "__main__":
    main()
