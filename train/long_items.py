"""Recomposed long-description items (zero cost: no new images, no writer, no API).

    python -m train.long_items [--k 8] [--per-scene-siblings 1] [--seed 0]

Every scene's label matrices (target, sibling, alternative and short items) are pooled into one
clause bank: clause text -> verdict per instance (checked by the datagen checker + review).  From
the bank, GME-length descriptions are recomposed:
  positive: k clauses that hold on the target and, together, exclude every other instance (each
            other instance fails at least one of them); the target is the scene's original target
            or a sibling with enough verified clauses (so the described object is not always the
            same instance);
  negative: the original target's recomposed clauses with the scene's verified flipped clause
            (holds for no instance) inserted at a random position, replacing its original clause
            when present -> exactly one false clause among k-1 true ones, like GME Rejection.
Output: $VLMG_DATA_ROOT/train/long_items.jsonl, items with `matrix_pre` (source "long"), usable
with --extra like mosaic_items.  Rows carry `origin` so rationale cells are looked up as usual.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from train import data as D
from train import traces as TR

OUT = D.TRAIN_ROOT / "long_items.jsonl"
SOURCES = [D.EXPORT, D.TRAIN_ROOT / "augment" / "cross_v1.jsonl", D.TRAIN_ROOT / "alt_negatives.jsonl", D.TRAIN_ROOT / "short_items.jsonl"]


def clause_bank(items: list[dict]):
    """(run, group) -> {"clauses": {text: {iid: verdict}}, "boxes": {iid: box}, "base": item, "targets": set, "flip": dict|None}"""
    bank: dict[tuple, dict] = {}
    for it in items:
        if it.get("source") in ("mosaic", "long", "cross"):
            continue
        m, why = TR.label_matrix(it)
        if m is None:
            continue
        key = (it["run"], it["group"])
        b = bank.setdefault(key, {"clauses": defaultdict(dict), "boxes": {}, "base": it, "targets": set(), "flip": None})
        for row in m["rows"]:
            b["boxes"].setdefault(row["iid"], row["box"])
            for cond, v in zip(m["conditions"], row["verdicts"]):
                b["clauses"][cond].setdefault(row["iid"], v)
        if it["kind"] == "positive" and it.get("target_iid") is not None:
            b["targets"].add(it["target_iid"])
        if it["kind"] == "negative" and m.get("flipped_clause") and m.get("flip_from") and (b["flip"] is None or (b["flip"].get("alt") and it.get("source") != "alt")):
            fc = m["flipped_clause"]
            col = {row["iid"]: row["verdicts"][m["flipped_idx"]] for row in m["rows"]}
            if all(v == "no" for v in col.values()):  # zero-satisfier verified
                orig = None
                for c, vs in b["clauses"].items():  # the true clause the flip replaced: same target, checker yes, shares the 'from' words
                    if m["flip_from"].lower() in c.lower() and vs.get(m["first_iid"]) == "yes":
                        orig = c
                        break
                b["flip"] = {"clause": fc, "from": m["flip_from"], "to": m.get("flip_to"), "target": m["first_iid"], "orig": orig, "alt": it.get("source") == "alt"}
    return bank


STOP = {"the", "a", "an", "is", "are", "has", "have", "with", "of", "in", "on", "and", "its", "it", "that", "this", "to", "at", "by", "from"}


def words(c: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", c.lower()) if w not in STOP}


def paraphrases(a: str, b: str) -> bool:
    wa, wb = words(a), words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5


def pick(rng: random.Random, yes_clauses: list[str], clauses: dict, target: int, others: list[int], k: int, tries: int = 80):
    """k mutually non-paraphrase clauses true on the target that jointly exclude every other instance;
    prefer sets where the others satisfy many clauses individually (discriminative only in combination)."""
    if len(yes_clauses) < k:
        return None
    best, best_score = None, -1
    for _ in range(tries):
        s = []
        for c in rng.sample(yes_clauses, len(yes_clauses)):
            if not any(paraphrases(c, d) for d in s):
                s.append(c)
            if len(s) == k:
                break
        if len(s) < k:
            return None
        ok = all(any(clauses[c].get(o) == "no" for c in s) for o in others)
        if not ok:
            continue
        score = sum(sum(clauses[c].get(o) == "yes" for c in s) for o in others)
        if score > best_score:
            best, best_score = s, score
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=7, help="clauses per recomposed description (fewer when the bank is short, min 5)")
    ap.add_argument("--per-scene-siblings", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    items = []
    for f in SOURCES:
        if Path(f).is_file():
            items.extend(D.load_items(Path(f)))
    bank = clause_bank(items)
    out, stats = [], defaultdict(int)
    for (run, group), b in sorted(bank.items()):
        base, clauses, boxes = b["base"], b["clauses"], b["boxes"]
        iids = sorted(boxes)
        if len(iids) < 2:
            continue
        cand_targets = list(b["targets"]) + [i for i in iids if i not in b["targets"]]
        made = 0
        for t in cand_targets:
            is_orig = t in b["targets"]
            if not is_orig and made > args.per_scene_siblings:
                break
            yes = [c for c, vs in clauses.items() if vs.get(t) == "yes" and all(vs.get(i) in ("yes", "no", "unclear") for i in iids)]
            k = min(args.k, len(yes))
            if k < 5:
                stats["skip_short_bank"] += 1
                continue
            others = [i for i in iids if i != t]
            sel = pick(rng, yes, clauses, t, others, k)
            if sel is None:
                stats["skip_not_unique"] += 1
                continue
            rng.shuffle(sel)
            rows = [{"iid": i, "box": boxes[i], "verdicts": [clauses[c].get(i, "unclear") for c in sel], "seen": {}, "origin": (run, group, i)} for i in iids]
            expr = f"the {base['category']} that " + ", ".join(sel[:-1]) + (", and " + sel[-1] if len(sel) > 1 else "")
            common = {"image": base["image"], "image_wh": base["image_wh"], "group": group, "category": base["category"], "label": base.get("label"),
                      "run": run, "source": "long", "n_candidates": len(iids)}
            out.append({**common, "id": f"long:{run}:{group}:{t}:pos", "kind": "positive", "expression": expr, "answer": {"bbox_2d": boxes[t]},
                        "target_iid": t, "matrix_pre": {"conditions": sel, "rows": rows, "answer_iid": t, "flipped_idx": None, "flipped_clause": None,
                                                        "first_iid": t, "flip_from": None, "flip_to": None}})
            stats["pos_orig" if is_orig else "pos_sibling"] += 1
            made += 1
            fl = b["flip"]
            if is_orig and fl and fl["target"] == t:
                # drop the clause the flip replaced and anything about the same attribute (shared content words), so the
                # negative never contradicts itself; the false clause must be found by looking, not by reading
                conds = [c for c in sel if c != fl["orig"] and not (words(c) & words(fl["clause"]))]
                if len(conds) < 4:
                    continue
                pos = rng.randrange(0, len(conds) + 1)
                conds.insert(pos, fl["clause"])
                rows_n = [{"iid": i, "box": boxes[i], "verdicts": [("no" if c == fl["clause"] else clauses[c].get(i, "unclear")) for c in conds],
                           "seen": {}, "origin": (run, group, i)} for i in iids]
                expr_n = f"the {base['category']} that " + ", ".join(conds[:-1]) + ", and " + conds[-1]
                out.append({**common, "id": f"long:{run}:{group}:{t}:neg", "kind": "negative", "expression": expr_n, "answer": {"bbox_2d": None},
                            "target_iid": None, "flipped": {"idx": pos, "clause_neg": fl["clause"], "from": fl["from"], "to": fl["to"]},
                            "matrix_pre": {"conditions": conds, "rows": rows_n, "answer_iid": None, "flipped_idx": pos, "flipped_clause": fl["clause"],
                                           "first_iid": t, "flip_from": fl["from"], "flip_to": fl["to"]}})
                stats["neg"] += 1
    with open(OUT, "w", encoding="utf-8") as fh:
        for it in out:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    lens = [len(it["matrix_pre"]["conditions"]) for it in out]
    print(f"scenes {len(bank)}, items {len(out)} {dict(stats)}, clauses per item mean {sum(lens) / max(1, len(lens)):.1f} -> {OUT}")


if __name__ == "__main__":
    main()
