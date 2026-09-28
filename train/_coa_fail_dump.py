"""Dump COA failures with their text: GME rejection items answered with a box, and RefCOCO wrong answers.

Usage: python -m train._coa_fail_dump <tag> [--n 8] [--set gme|refcoco]
"""
import argparse
import json
import random
from pathlib import Path

from train import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--set", default="gme")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path.home() / "vlmg-data/train/eval" / a.tag
    recs = [json.loads(l) for l in open(root / f"{a.set}.jsonl")]
    if a.set == "gme":
        text = {it["id"]: it for it in D.gme_items()}
        fails = [r for r in recs if r["dimension"] == "Rejection" and r["output_type"] == "box"]
    else:
        text = {}
        fails = [r for r in recs if not r["correct"]]
    random.Random(a.seed).shuffle(fails)
    print(f"[{a.tag}/{a.set}] {len(fails)} failures, showing {min(a.n, len(fails))}")
    for r in fails[: a.n]:
        it = text.get(r["id"], {})
        expr = it.get("expr") or it.get("expression") or r.get("raw", "")[:0]
        coa = r["coa"]
        print("=" * 100)
        print(f"id {r['id']} | out {r['output_type']} free {r.get('free_type')} | K {r['n_cand']} k0 {coa.get('k0')} chosen {coa.get('chosen')} derived {coa.get('derived')} fits {coa.get('fits')}")
        if expr:
            print("EXPR:", expr)
        else:  # RefCOCO: the expression lives in the raw prompt dump
            raw = r.get("raw", "")
            print("RAW-HEAD:", raw[:300].replace("\n", " | "))
        for j, au in enumerate(coa["audits"]):
            if isinstance(au, str):  # raw audit text as generated
                print(f"-- audit {j}:")
                for ln in au.strip().splitlines():
                    print("     ", ln.strip())
            else:
                print(f"-- audit {j}: verdict={au.get('verdict')} named={au.get('named')} box={au.get('box')}")
                for ln in au.get("lines", []):
                    print("     ", ln if isinstance(ln, str) else json.dumps(ln, ensure_ascii=False))
        print("ANSWER:", (coa.get("model_answer_raw") or "")[:200])


if __name__ == "__main__":
    main()
