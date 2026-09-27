"""A5-1: render every label audit; assert decide(parse(render)) == the label answer; count dropped rows; echo stats."""
import collections, json
from pathlib import Path
from train import coa, data as D, traces as TR
from train.sft_lora import _rationale_of
files = [D.EXPORT, D.TRAIN_ROOT / "augment" / "cross_v1.jsonl", D.TRAIN_ROOT / "alt_negatives.jsonl", D.TRAIN_ROOT / "short_items.jsonl", D.TRAIN_ROOT / "mosaic_items.jsonl", D.TRAIN_ROOT / "long_items.jsonl"]
items = []
for f in files:
    if Path(f).is_file(): items.extend(D.load_items(Path(f)))
n_items = ok_items = 0; n_rows = 0; stats = collections.Counter(); single_err = 0; all_mis = 0; echo_lines = n_lines = 0
bad_examples = []
for it in items:
    m, why = TR.label_matrix(it)
    if m is None: continue
    rows = TR.ordered_rows(m, "commit")
    rof = _rationale_of(m)
    audits = [coa.parse_audit(coa.render_audit(m, r, rof)) for r in rows]
    boxes = [r["box"] for r in rows]
    ans, chosen, fits = coa.decide(audits, boxes)
    label_ans = m.get("answer_iid")
    label_box = next((r["box"] for r in rows if r["iid"] == label_ans), None) if label_ans is not None else None
    consistent = (ans is None and label_box is None) or (ans is not None and label_box is not None and ans == label_box)
    n_items += 1; ok_items += consistent
    if not consistent and len(bad_examples) < 3: bad_examples.append((it["kind"], it.get("source"), fits, label_ans, [r["iid"] for r in rows]))
    stats[(it["kind"], "ok" if consistent else "bad")] += 1
    for a, r in zip(audits, rows):
        n_rows += 1
        mis = [l for l in a["lines"] if l["verdict"] == "mismatch"]
        single_err += len(mis) == 1
        all_mis += len(mis) == len(a["lines"]) and len(a["lines"]) > 1
        for l in a["lines"]:
            n_lines += 1; echo_lines += coa.norm(l["seen"]) == coa.norm(l["claimed"])
print(f"items {n_items}: label-consistent {ok_items} ({ok_items/max(1,n_items):.3f}); by kind {dict(stats)}")
print(f"rows {n_rows}: single-mismatch rows {single_err} ({single_err/max(1,n_rows):.2f}), all-mismatch rows {all_mis} ({all_mis/max(1,n_rows):.2f}); lines {n_lines}: echo (seen==claim) {echo_lines/max(1,n_lines):.2f}")
for b in bad_examples: print("  inconsistent:", b)
