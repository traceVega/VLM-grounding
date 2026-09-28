"""Gates and export for data recipe A (train.gme_writer): GME-style positives, one-error negatives (each falsified detail that
holds for no instance), and two-error negatives composed from pairs of falsifications.

    python -m train.gme_export [--min-details 6] [--two-error]

Gates per scene: the target's details judged "yes" are kept (others dropped; fewer than --min-details -> scene dropped); a
falsification is kept only when its new detail is "no" on every instance (zero satisfier) and its detail survived; the positive
is emitted only when every other instance fails at least one kept detail.  Output: $VLMG_DATA_ROOT/train/gme_style_items.jsonl
(source "gmestyle", matrix_pre with per-row observed values), usable with --extra like long_items.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from datagen import common as C
from train import data as D
from train.gme_writer import ROOT

OUT = D.TRAIN_ROOT / "gme_style_items.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-details", type=int, default=6)
    ap.add_argument("--two-error", action="store_true", help="also emit two-error negatives from pairs of kept falsifications")
    args = ap.parse_args()
    written = {(r["run"], r["group"]): r for r in C.read_jsonl(ROOT / "write.jsonl")}
    checks = {(r["run"], r["group"]): r for r in C.read_jsonl(ROOT / "check.jsonl")}
    out, st = [], Counter()
    for key, ck in sorted(checks.items()):
        w = written[key]
        rows = ck["rows"]
        tgt = next(r for r in rows if r["is_target"])
        n_d = len(w["details"])
        keep = [j for j in range(n_d) if tgt["verdicts"][j] == "yes"]
        st["details_dropped"] += n_d - len(keep)
        if len(keep) < args.min_details:
            st["scene_short"] += 1
            continue
        conds = [w["details"][j] for j in keep]
        common = {"image": w["image"], "image_wh": w["image_wh"], "group": w["group"], "category": w["category"], "label": w.get("label"),
                  "run": w["run"], "source": "gmestyle", "n_candidates": len(rows)}

        def row_of(r, extra_idx=None):
            verdicts = [r["verdicts"][j] for j in keep]
            seen = {str(k): r["seen"][j] for k, j in enumerate(keep) if r["seen"][j]}
            for k, (j_new, j_keep) in (extra_idx or {}).items():
                verdicts[k] = r["verdicts"][j_new]
                if r["seen"][j_new]:
                    seen[str(k)] = r["seen"][j_new]
            return {"iid": r["iid"], "box": r["box"], "verdicts": verdicts, "seen": seen, "origin": (w["run"], w["group"], r["iid"])}

        unique = all(any(r["verdicts"][j] == "no" for j in keep) for r in rows if not r["is_target"])
        if unique:
            out.append({**common, "id": f"gmestyle:{w['run']}:{w['group']}:{w['target']}:pos", "kind": "positive", "expression": w["description"],
                        "answer": {"bbox_2d": tgt["box"]}, "target_iid": w["target"],
                        "matrix_pre": {"conditions": conds, "rows": [row_of(r) for r in rows], "answer_iid": w["target"], "flipped_idx": None,
                                       "flipped_clause": None, "first_iid": w["target"], "flip_from": None, "flip_to": None}})
            st["positive"] += 1
        else:
            st["positive_not_unique"] += 1
        # falsifications: statement index n_d + v in the check
        kept_v = []
        for vi, v in enumerate(w["versions"]):
            j_new = n_d + vi
            if v["idx"] not in keep:
                st["flip_detail_dropped"] += 1
                continue
            if not all(r["verdicts"][j_new] == "no" for r in rows):
                st["flip_has_satisfier"] += 1
                continue
            kept_v.append((vi, v, j_new))
        for vi, v, j_new in kept_v:
            k = keep.index(v["idx"])
            conds_neg = list(conds)
            conds_neg[k] = v["detail"]
            rows_neg = [row_of(r, {k: (j_new, v["idx"])}) for r in rows]
            t_seen = tgt["seen"][j_new] or v["old"]
            out.append({**common, "id": f"gmestyle:{w['run']}:{w['group']}:{w['target']}:neg{vi}", "kind": "negative", "expression": v["description"],
                        "answer": {"bbox_2d": None}, "target_iid": None, "flip_kind": v["kind"],
                        "matrix_pre": {"conditions": conds_neg, "rows": rows_neg, "answer_iid": None, "flipped_idx": k, "flipped_clause": v["detail"],
                                       "first_iid": w["target"], "flip_from": t_seen, "flip_to": v["new"], "seen_pre": {str(k): t_seen}}})
            st[f"negative_{v['kind']}"] += 1
        if args.two_error:
            for a in range(len(kept_v)):
                for b in range(a + 1, len(kept_v)):
                    (va_i, va, ja), (vb_i, vb, jb) = kept_v[a], kept_v[b]
                    if va["idx"] == vb["idx"]:
                        continue
                    ka, kb = keep.index(va["idx"]), keep.index(vb["idx"])
                    conds2 = list(conds)
                    conds2[ka], conds2[kb] = va["detail"], vb["detail"]
                    expr = C.substitute_once(va["description"], vb["old"], vb["new"]) or None
                    if not expr:
                        st["two_error_no_text"] += 1
                        continue
                    rows2 = [row_of(r, {ka: (ja, va["idx"]), kb: (jb, vb["idx"])}) for r in rows]
                    sa, sb = tgt["seen"][ja] or va["old"], tgt["seen"][jb] or vb["old"]
                    out.append({**common, "id": f"gmestyle:{w['run']}:{w['group']}:{w['target']}:neg{va_i}{vb_i}", "kind": "negative", "expression": expr,
                                "answer": {"bbox_2d": None}, "target_iid": None, "flip_kind": va["kind"] + vb["kind"],
                                "matrix_pre": {"conditions": conds2, "rows": rows2, "answer_iid": None, "flipped_idx": ka, "flipped_clause": va["detail"],
                                               "first_iid": w["target"], "flip_from": sa, "flip_to": va["new"], "seen_pre": {str(ka): sa, str(kb): sb}}})
                    st["negative_two_error"] += 1
    with open(OUT, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(out)} items -> {OUT}\n{dict(st)}")


if __name__ == "__main__":
    main()
