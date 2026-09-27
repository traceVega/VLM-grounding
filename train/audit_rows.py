"""Recomposed audit rows for the observe-then-judge head (zero cost, from the checked clause banks).

    python -m train.audit_rows [--k 6] [--per-instance 2] [--two-error-frac 0.25] [--seed 0]

For every checked instance of every scene: rows of k clauses drawn from the scene's clause bank that all hold on the
instance except exactly one (or, for a fraction, exactly two) whose observed counter-value is known (the checker's
rationale for that cell, or the instance's own checked clause about the same attribute).  These are GME-shaped audit
targets ("the right object, exactly one false detail") which the plain label matrices contain only ~25% of.
Output: $VLMG_DATA_ROOT/train/audit_rows.jsonl — items with source "audit" and a one-row `matrix_pre`; train.sft_lora
turns each into one audit example (no turn-1 or answer example).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from train import coa as COA
from train import data as D
from train import traces as TR
from train.long_items import SOURCES, clause_bank, paraphrases

OUT = D.TRAIN_ROOT / "audit_rows.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--per-instance", type=int, default=2, help="rows per instance")
    ap.add_argument("--two-error-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    items = []
    for f in SOURCES:
        if Path(f).is_file():
            items.extend(D.load_items(Path(f)))
    bank = clause_bank(items)
    rat = TR.rationales()
    out, stats = [], defaultdict(int)
    for (run, group), b in sorted(bank.items()):
        clauses, boxes, base = b["clauses"], b["boxes"], b["base"]
        for iid in sorted(boxes):
            yes = [c for c, vs in clauses.items() if vs.get(iid) == "yes"]
            # 'no' clauses with a usable observed value
            nos = []
            for c, vs in clauses.items():
                if vs.get(iid) != "no":
                    continue
                info = rat.get((run, group, iid), {}).get(c)
                seen = None
                if info and info["blind"] == "no" and info["label"] == "no" and info.get("rationale"):
                    r = TR._clean_reason(info["rationale"])
                    if r and COA.named_value(r, c):
                        seen = r
                if seen is None:
                    seen = COA.seen_from_bank((run, group, iid), c)
                if seen:
                    nos.append((c, seen))
            if len(yes) < 3 or not nos:
                stats["skip_instance"] += 1
                continue
            made = 0
            for _ in range(args.per_instance * 3):
                if made >= args.per_instance:
                    break
                n_err = 2 if (rng.random() < args.two_error_frac and len(nos) >= 2) else 1
                errs = rng.sample(nos, n_err)
                pool = []
                for c in rng.sample(yes, len(yes)):
                    wc = set(COA.norm(c).split()) - COA.NEG - {"wearing", "person", "man", "woman", "car", "dog"}
                    clash = any(paraphrases(c, e[0]) or paraphrases(c, e[1]) or (wc & (set(COA.norm(e[0]).split()) | set(COA.norm(e[1]).split())) - COA.NEG - {"wearing"}) for e in errs)
                    if not clash and not any(paraphrases(c, d) for d in pool):  # nothing about the error's attribute or its counter-value in the true clauses
                        pool.append(c)
                    if len(pool) >= args.k - n_err:
                        break
                if len(pool) < max(2, min(args.k, len(yes)) - n_err):
                    continue
                conds = pool + [e[0] for e in errs]
                rng.shuffle(conds)
                verdicts = ["no" if any(c == e[0] for e in errs) else "yes" for c in conds]
                seen = {conds.index(e[0]): e[1] for e in errs}
                row = {"iid": iid, "box": boxes[iid], "verdicts": verdicts, "seen": {}, "origin": (run, group, iid)}
                m = {"conditions": conds, "rows": [row], "answer_iid": None, "flipped_idx": conds.index(errs[0][0]), "flipped_clause": errs[0][0],
                     "first_iid": iid, "flip_from": errs[0][1], "flip_to": None, "seen_pre": {str(j): v for j, v in seen.items()}}
                out.append({"id": f"audit:{run}:{group}:{iid}:{made}", "kind": "negative", "source": "audit", "run": run, "group": group, "image": base["image"],
                            "image_wh": base["image_wh"], "category": base["category"], "label": base.get("label"), "expression": f"the {base['category']} that " + ", ".join(conds),
                            "answer": {"bbox_2d": None}, "target_iid": None, "n_candidates": 1, "matrix_pre": m})
                made += 1
                stats["two_error" if n_err == 2 else "one_error"] += 1
    with open(OUT, "w", encoding="utf-8") as fh:
        for it in out:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"scenes {len(bank)}, audit rows {len(out)} {dict(stats)} -> {OUT}")


if __name__ == "__main__":
    main()
