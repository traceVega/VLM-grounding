"""Stage `export`: scene records -> training items (JSONL), one line per item.

Per scene up to three items share `group` = image_id:
  positive           e_T  -> target box
  sibling_positive   e_S  -> sibling box
  negative           e_T⁻ -> null

Which scenes qualify:
  * with a reviewer notes file (`datagen/reviews/<run>.json`): a scene's negative is
    exported when its bucket kind is "ok"; scenes with kind "bad" export nothing; kind
    "unsure" exports nothing; a scene whose only fault is the negative (flip_failed,
    flip_true_of_target, flip_satisfied) exports its positives when the automatic
    description gates pass.  Scenes the reviewer did not annotate fall back to the
    automatic gates.
  * automatic: negative when `negative_usable_sentence`; positive when the writer's
    details are all confirmed and the listener or the base model found the target;
    sibling positive when e_S differs from e_T and the listener found the sibling.

Boxes are [x0, y0, x1, y1] in pixels of the raw OpenImages image (`image` is its path).
"""

from __future__ import annotations

import json
from pathlib import Path

from datagen import common as C
from datagen.report import build_records, load_notes


def _same(a: str | None, b: str | None) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def items_for(run: str, rec: dict, verdict: str | None) -> list[dict]:
    """verdict: 'ok' | 'unsure' | 'bad' from the reviewer, or None for automatic gates."""
    g, e = rec["gates"], rec["expressions"]
    if verdict == "bad":
        return []
    inst = {i["iid"]: i for i in rec["instances"]}
    t, s = inst[rec["target_iid"]], inst.get(rec["sibling_iid"])
    W, H = rec["image_wh"]
    base = {"image": str(C.image_path(rec["image_id"])), "image_wh": [W, H], "group": rec["image_id"],
            "category": rec["category"], "label": rec["label"], "run": run,
            "source": "reviewed" if verdict else "auto", "n_candidates": len(rec["instances"])}
    out = []
    if verdict == "unsure":  # the doubt is usually about the description itself: export nothing
        return []
    auto_pos = bool(g.get("writer_target_all_yes") and (g.get("listener_unique_hit") or g.get("base_correct_on_target")))
    # a scene whose only fault is the negative (flip failed / still true / satisfied) keeps its positives
    # when the automatic description gates pass; not_unique, writer_wrong and scene faults taint them
    pos_ok = (verdict == "ok") or (verdict is None and auto_pos) or (verdict == "flip" and auto_pos)
    if pos_ok and e.get("e_T"):
        out.append({**base, "id": f"{run}:{rec['image_id']}:pos", "kind": "positive", "expression": e["e_T"],
                    "answer": {"bbox_2d": [round(v, 1) for v in t["box"]]}, "target_iid": t["iid"],
                    "base_correct": g.get("base_correct_on_target")})
    if s and e.get("e_S") and not _same(e["e_S"], e["e_T"]) and g.get("sibling_unique_hit"):
        out.append({**base, "id": f"{run}:{rec['image_id']}:sib", "kind": "sibling_positive", "expression": e["e_S"],
                    "answer": {"bbox_2d": [round(v, 1) for v in s["box"]]}, "target_iid": s["iid"]})
    neg_ok = (verdict == "ok") or (verdict is None and g.get("negative_usable_sentence"))
    if verdict == "flip":
        neg_ok = False
    if neg_ok and e.get("e_T_neg") and not _same(e["e_T_neg"], e["e_T"]):
        out.append({**base, "id": f"{run}:{rec['image_id']}:neg", "kind": "negative", "expression": e["e_T_neg"],
                    "answer": {"bbox_2d": None}, "target_iid": None, "flipped": rec["flipped"],
                    "reflipped": rec.get("reflipped"), "blind_judge_found_flip": rec["blind"].get("judge_found_flip"),
                    "base_p_null": (rec["policy"].get("neg") or {}).get("p_null")})
    return out


def run(run: str, primary: str, out_path: str | None = None) -> None:
    root = C.run_root(run)
    records = build_records(root, primary)
    notes = load_notes(run)
    kind = {b["id"]: b["kind"] for b in (notes or {}).get("buckets", [])}
    cases = (notes or {}).get("cases", {})
    out = Path(out_path) if out_path else root.parent / "exports" / f"{run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"positive": 0, "sibling_positive": 0, "negative": 0}
    by_source = {"reviewed": 0, "auto": 0}
    with open(out, "w", encoding="utf-8") as fh:
        for rec in records:
            c = cases.get(rec["image_id"])
            verdict = kind.get(c["bucket"]) if c else None
            if c and c["bucket"] in ("flip_failed", "flip_true_of_target", "flip_satisfied"):
                verdict = "flip"
            for it in items_for(run, rec, verdict):
                counts[it["kind"]] += 1
                by_source[it["source"]] += 1
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"export {run} (primary {primary}): {counts}, by source {by_source} -> {out}")
