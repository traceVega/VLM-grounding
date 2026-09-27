"""Short referring expressions from the label matrix (RefCOCO-style items, no downloads, no
model calls): a clause that exactly one candidate instance satisfies ("yes") while every other
instance fails ("no") names that instance uniquely -> "the {category} that {clause}".  When no
single clause is unique, a pair of clauses whose intersection is unique is used.

    python -m train.short_items [--max-per-scene 2] -> $VLMG_DATA_ROOT/train/short_items.jsonl (source "short")

Items carry `clause_idx` (matrix columns) and `target_iid` so train.traces renders their traces.
"""

from __future__ import annotations

import argparse
import itertools
import json

from train import data as D
from train import traces as TR

OUT = D.TRAIN_ROOT / "short_items.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-per-scene", type=int, default=2)
    args = ap.parse_args()
    recs = TR.records()
    base = {}
    for it in D.load_items():
        base.setdefault((it["run"], it["group"]), it)
    items = []
    n_scenes = 0
    for key, b in base.items():
        rec = recs.get(key)
        if rec is None or len(rec["instances"]) < 2:
            continue
        cv = rec.get("clause_verdicts") or {}
        clauses = rec["clauses"]
        iids = [i["iid"] for i in rec["instances"]]
        if not clauses or any(str(i) not in cv for i in iids):
            continue
        col = lambda j, iid: (cv[str(iid)] + ["unclear"] * len(clauses))[j]
        found = []
        for j in range(len(clauses)):
            yes = [i for i in iids if col(j, i) == "yes"]
            if len(yes) == 1 and all(col(j, i) == "no" for i in iids if i != yes[0]):
                found.append(([j], yes[0]))
        if not found:
            for j1, j2 in itertools.combinations(range(len(clauses)), 2):
                yes = [i for i in iids if col(j1, i) == "yes" and col(j2, i) == "yes"]
                if len(yes) == 1 and all(col(j1, i) == "no" or col(j2, i) == "no" for i in iids if i != yes[0]):
                    found.append(([j1, j2], yes[0]))
        if not found:
            continue
        n_scenes += 1
        inst = {i["iid"]: i for i in rec["instances"]}
        for n, (idxs, iid) in enumerate(found[: args.max_per_scene]):
            expr = f"the {rec['category']} that " + " and ".join(clauses[j] for j in idxs)
            items.append({"image": b["image"], "image_wh": b["image_wh"], "group": b["group"], "category": b["category"], "label": b["label"],
                          "run": b["run"], "source": "short", "n_candidates": b["n_candidates"], "id": f"{b['run']}:{b['group']}:short{n}",
                          "kind": "positive", "expression": expr, "answer": {"bbox_2d": [round(v, 1) for v in inst[iid]["box"]]},
                          "target_iid": iid, "clause_idx": idxs})
    with open(OUT, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} short items from {n_scenes} scenes -> {OUT}")
    for it in items[:5]:
        print("  ", it["id"], "|", it["expression"])


if __name__ == "__main__":
    main()
