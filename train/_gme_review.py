"""Review dump for data recipe A: descriptions, falsifications and checker verdicts of N scenes (reviewer reads, no API).

Usage: python -m train._gme_review [--n 20] [--seed 0]
"""
import argparse
import json
import random
from collections import Counter

from datagen import common as C
from train.gme_writer import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    written = {(r["run"], r["group"]): r for r in C.read_jsonl(ROOT / "write.jsonl")}
    checks = [r for r in C.read_jsonl(ROOT / "check.jsonl")]
    st = Counter()
    for ck in checks:
        w = written[(ck["run"], ck["group"])]
        tgt = next(r for r in ck["rows"] if r["is_target"])
        n_d = len(w["details"])
        st["scenes"] += 1
        st["details"] += n_d
        st["target_detail_no"] += sum(v == "no" for v in tgt["verdicts"][:n_d])
        st["target_detail_unclear"] += sum(v == "unclear" for v in tgt["verdicts"][:n_d])
        st["unparsed"] += sum(v == "unparsed" for r in ck["rows"] for v in r["verdicts"])
        for vi in range(len(w["versions"])):
            st["flips"] += 1
            st["flip_no_on_all"] += all(r["verdicts"][n_d + vi] == "no" for r in ck["rows"])
            st["flip_target_yes"] += tgt["verdicts"][n_d + vi] == "yes"
        st["seen_empty"] += sum(1 for r in ck["rows"] for s in r["seen"] if not s)
    print("STATS", dict(st))
    random.Random(a.seed).shuffle(checks)
    for ck in checks[: a.n]:
        w = written[(ck["run"], ck["group"])]
        tgt = next(r for r in ck["rows"] if r["is_target"])
        n_d = len(w["details"])
        print("=" * 100)
        print(f"{w['group']} {w['category']} target {w['target']} instances {len(ck['rows'])}")
        print("DESC:", w["description"])
        for j, d in enumerate(w["details"]):
            others = "".join(r["verdicts"][j][0] for r in ck["rows"] if not r["is_target"])
            print(f"  {j + 1:2d}. {d:60s} | target {tgt['verdicts'][j]:7s} seen: {tgt['seen'][j][:50]:50s} | others {others}")
        for vi, v in enumerate(w["versions"]):
            j = n_d + vi
            allv = "".join(r["verdicts"][j][0] for r in ck["rows"])
            print(f"  FLIP {v['kind']} d{v['idx'] + 1}: {v['old']} -> {v['new']} | {v['detail'][:55]:55s} | verdicts {allv} | target seen: {tgt['seen'][j][:60]}")


if __name__ == "__main__":
    main()
