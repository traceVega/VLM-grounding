"""Compare rationale files (train.rationales vs train.rationales_gemini) cell by cell.

    python -m train.rat_compare [--files a.jsonl,b.jsonl]

Per file: blind-verdict agreement with the label by label value, kept cells (blind agrees and a
rationale exists), no-cells kept, sentence reuse within an instance, and the share of kept
"no" rationales that add no content word beyond the clause (bare negations such as
"not wearing grey baseball cap").  Also the agreement between the two files' blind verdicts.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from train import data as D

STOP = {"not", "no", "the", "a", "an", "is", "are", "it", "its", "has", "have", "with", "of", "in", "on", "and", "visible", "none", "there", "any", "this", "that", "does", "do", "wearing", "be"}


def words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower()) if w not in STOP and len(w) > 2}


def stats(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    agree, tot = collections.Counter(), collections.Counter()
    kept = collections.Counter()
    reuse_inst, reuse_cells, n_cells, bare, kept_no = 0, 0, 0, 0, 0
    lens = []
    for r in rows:
        seen = collections.Counter(re.sub(r"\W+", " ", (x or "").lower()).strip() for x in r["rationale"] if x)
        if any(v > 1 for v in seen.values()):
            reuse_inst += 1
        reuse_cells += sum(v - 1 for v in seen.values() if v > 1)
        for c, lab, bl, ra in zip(r["conditions"], r["label"], r["blind"], r["rationale"]):
            n_cells += 1
            tot[lab] += 1
            agree[lab] += lab == bl
            if lab == bl and ra:
                kept[lab] += 1
                lens.append(len(ra.split()))
                if lab == "no":
                    kept_no += 1
                    if not (words(ra) - words(c)):
                        bare += 1
    lens.sort()
    return {"instances": len(rows), "cells": n_cells,
            "agree": {k: round(agree[k] / max(1, tot[k]), 3) for k in ("yes", "no", "unclear")},
            "n": dict(tot), "kept_frac": round(sum(kept.values()) / max(1, n_cells), 3),
            "no_kept": f"{kept['no']}/{tot['no']}", "yes_kept": f"{kept['yes']}/{tot['yes']}",
            "reuse_instances": f"{reuse_inst}/{len(rows)}", "reuse_cells": reuse_cells,
            "bare_negation_frac_of_kept_no": round(bare / max(1, kept_no), 3),
            "words_median": lens[len(lens) // 2] if lens else None, "words_p90": lens[int(len(lens) * 0.9)] if lens else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", default=f"{D.TRAIN_ROOT / 'rationales.jsonl'},{D.TRAIN_ROOT / 'rationales_gemini.jsonl'}")
    args = ap.parse_args()
    paths = [Path(p).expanduser() for p in args.files.split(",")]
    per = {}
    for p in paths:
        if p.is_file():
            per[p.name] = stats(p)
            print(p.name, json.dumps(per[p.name]))
    if len(paths) == 2 and all(p.is_file() for p in paths):
        a = {(r["run"], r["image_id"], r["iid"]): r for r in (json.loads(l) for l in open(paths[0], encoding="utf-8") if l.strip())}
        b = {(r["run"], r["image_id"], r["iid"]): r for r in (json.loads(l) for l in open(paths[1], encoding="utf-8") if l.strip())}
        same = n = 0
        both_no = both_no_agree = 0
        for k in a.keys() & b.keys():
            ca = dict(zip(a[k]["conditions"], zip(a[k]["label"], a[k]["blind"])))
            cb = dict(zip(b[k]["conditions"], zip(b[k]["label"], b[k]["blind"])))
            for c in ca.keys() & cb.keys():
                n += 1
                same += ca[c][1] == cb[c][1]
                if ca[c][0] == "no":
                    both_no += 1
                    both_no_agree += (ca[c][1] == "no") and (cb[c][1] == "no")
        print(f"shared instances {len(a.keys() & b.keys())}, cells {n}: blind verdicts identical {same / max(1, n):.3f}; label-no cells where both say no {both_no_agree}/{both_no}")


if __name__ == "__main__":
    main()
