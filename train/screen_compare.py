"""Compare screening evals against the baseline on exactly the same items.

    python -m train.screen_compare scr_sft_v1 [more screening tags] [--base base4b]

The baseline was evaluated on the full sets; each screening tag saw a subset.  For every
screening tag the baseline's per-item rows are filtered to that tag's ids, summarised the
same way, and printed as the first column next to the tag.
"""

from __future__ import annotations

import argparse
import json
import shutil

from train import data as D
from train.compare import ROWS, get
from train.eval_suite import summarize


def rows(tag: str, name: str) -> list[dict]:
    f = D.TRAIN_ROOT / "eval" / tag / f"{name}.jsonl"
    return [json.loads(l) for l in open(f, encoding="utf-8")] if f.is_file() else []


def base_on(tag: str, base: str) -> dict:
    out_dir = D.TRAIN_ROOT / "eval" / f"{base}@{tag}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name in ("gme", "gmegray", "own", "gray", "refcoco"):
        ids = {r["id"] for r in rows(tag, name)}
        if not ids:
            continue
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows(base, name):
                if r["id"] in ids:
                    fh.write(json.dumps(r) + "\n")
    return summarize(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--base", default="base4b")
    args = ap.parse_args()
    cols: list[tuple[str, dict]] = []
    for t in args.tags:
        cols.append((f"{args.base}@{t[:8]}", base_on(t, args.base)))
        cols.append((t, summarize(D.TRAIN_ROOT / "eval" / t)))
    w = max(len(r[0]) for r in ROWS) + 2
    print("| 指标".ljust(w) + "".join(f"| {name[:22]:>22} " for name, _ in cols) + "|")
    print("|" + "-" * (w - 1) + "".join("|" + "-" * 24 for _ in cols) + "|")
    for label, path in ROWS:
        cells = []
        for _, s in cols:
            if path is None:
                rc = get(s, "refcoco") or {}
                vals = [v.get("null_rate") for v in rc.values() if v.get("null_rate") is not None]
                v = sum(vals) / len(vals) if vals else None
            else:
                v = get(s, path)
            cells.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        print(f"| {label}".ljust(w) + "".join(f"| {c:>22} " for c in cells) + "|")
    n = {name: (get(s, "gme.positive_n"), get(s, "gme.rejection_n"), get(s, "gmegray.n"), get(s, "own.n")) for name, s in cols}
    print("n (GME pos, GME rej, gmegray, own):", n)


if __name__ == "__main__":
    main()
