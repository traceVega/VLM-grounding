"""One table over evaluation tags (train.eval_suite summaries), plus paired per-item deltas
against a reference tag on the ids both tags scored.

    python -m train.night_report base4b base_trace scr_sft_trace_v1 scr_grpo_v2_indep ... [--ref base_trace]
"""

from __future__ import annotations

import argparse
import json

from train import data as D


def load(tag: str):
    d = D.TRAIN_ROOT / "eval" / tag
    s = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").is_file() else {}
    rows = {}
    for name in ("gme", "own", "refcoco"):
        f = d / f"{name}.jsonl"
        if f.is_file():
            rows[name] = {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8") if l.strip()}
    return s, rows


def g(s, *keys, default=None):
    for k in keys:
        if not isinstance(s, dict) or k not in s:
            return default
        s = s[k]
    return s


def fmt(v, pct=True):
    if v is None:
        return "  -  "
    if isinstance(v, float) and pct:
        return f"{100 * v:5.1f}"
    return f"{v:>5}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--ref", default=None, help="paired deltas against this tag")
    args = ap.parse_args()
    cols = [("net", ("gme", "net_rejection")), ("rej", ("gme", "rejection_acc")), ("pos", ("gme", "positive_acc")),
            ("posNull#", ("gme", "positive_null_n")), ("Disc", ("gme", "by_dimension", "Discriminative")), ("Lim", ("gme", "by_dimension", "Limited")),
            ("Spat", ("gme", "by_dimension", "Spatial")), ("fmt", ("gme", "positive_trace", "format_ok_rate")),
            ("cons", ("gme", "positive_trace", "consistent_rate")), ("fNo", ("gme", "positive_trace", "false_no_cell_rate")),
            ("freeRej", ("gme", "free_rejection_acc")), ("freePos", ("gme", "free_positive_acc")),
            ("ownNeg", ("own", "neg_null_rate")), ("ownPos", ("own", "pos_acc")), ("ownPosNull", ("own", "pos_null_rate")),
            ("pair", ("own", "pair_acc")), ("accuse", ("own", "accusation_acc")), ("ownFNo", ("own", "pos_trace", "false_no_cell_rate")),
            ("rc", ("refcoco", "RefCOCO", "acc")), ("rc+", ("refcoco", "RefCOCOplus", "acc")), ("rcg", ("refcoco", "RefCOCOg", "acc"))]
    print("tag".ljust(22) + " ".join(c.rjust(8) for c, _ in cols))
    data = {}
    for tag in args.tags:
        s, rows = load(tag)
        data[tag] = (s, rows)
        vals = []
        for name, keys in cols:
            v = g(s, *keys)
            vals.append(fmt(v, pct=name != "posNull#"))
        print(tag.ljust(22) + " ".join(v.rjust(8) for v in vals))
    if args.ref and args.ref in data:
        print(f"\npaired deltas vs {args.ref} (common ids; points):")
        _, ref_rows = data[args.ref]
        for tag in args.tags:
            if tag == args.ref:
                continue
            _, rows = data[tag]
            out = []
            for name in ("gme", "own", "refcoco"):
                a, b = ref_rows.get(name, {}), rows.get(name, {})
                ids = sorted(set(a) & set(b))
                if not ids:
                    continue
                if name == "gme":
                    for label, pred in (("rej", lambda r: r["n_gt"] == 0), ("pos", lambda r: r["n_gt"] > 0)):
                        sub = [i for i in ids if pred(a[i])]
                        if sub:
                            d = sum(b[i]["correct"] - a[i]["correct"] for i in sub) / len(sub)
                            out.append(f"{label} {100 * d:+.1f} (n={len(sub)})")
                else:
                    d = sum(b[i]["correct"] - a[i]["correct"] for i in ids) / len(ids)
                    out.append(f"{name} {100 * d:+.1f} (n={len(ids)})")
            print(f"  {tag.ljust(20)} " + "; ".join(out))


if __name__ == "__main__":
    main()
