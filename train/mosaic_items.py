"""Mosaic augmentation: two same-category scenes side by side, described by scene A's expression,
so the candidate table has A's instances plus B's (more distractors, GME-like).  Labels are
complete without any new call: B's per-instance verdicts on A's clauses are the cross-scene
verification rows (train/augment/verify.jsonl, usable pairs = no B instance satisfies A's
description).

    python -m train.mosaic_items -> $VLMG_DATA_ROOT/train/mosaic_items.jsonl (source "mosaic")

Positives: A's positive/sibling item on the mosaic (answer = A's box, shifted).  Negatives: A's
flipped negative when it exists; B rows keep their verdicts on the unflipped clauses and get
"unclear" on the flipped one, and are kept only if some other clause is "no".  Items carry
`matrix_pre` (rows with boxes in mosaic pixels, verdicts, and `origin` for rationale lookup).
"""

from __future__ import annotations

import argparse
import json

from PIL import Image

from datagen import common as C
from train import data as D
from train import traces as TR

OUT = D.TRAIN_ROOT / "mosaic_items.jsonl"
IMG_DIR = D.TRAIN_ROOT / "mosaic"


def compose(img_a: Image.Image, img_b: Image.Image):
    """A left, B right scaled to A's height; returns (canvas, scale_b, x_offset_b)."""
    wa, ha = img_a.size
    wb, hb = img_b.size
    s = ha / hb
    nb = (max(1, round(wb * s)), ha)
    canvas = Image.new("RGB", (wa + nb[0], ha), (128, 128, 128))
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b.resize(nb, Image.LANCZOS), (wa, 0))
    return canvas, s, wa


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    items_by_id = {it["id"]: it for it in D.load_items()}
    negs_by_group = {it["group"]: it for it in D.load_items() if it["kind"] == "negative"}
    recs = TR.records()
    rows_v = [r for r in TR.cross_rows().values() if r.get("usable")]
    if args.limit:
        rows_v = rows_v[: args.limit]
    out = []
    n_pos = n_neg = 0
    for r in rows_v:
        a = items_by_id.get(r["src_id"])
        rec_b = recs.get((r["run"], r["group"]))
        if a is None or rec_b is None:
            continue
        ma, why = TR.label_matrix(a)
        if ma is None:
            continue
        img_a = Image.open(a["image"]).convert("RGB")
        img_b = Image.open(C.image_path(r["group"])).convert("RGB")
        canvas, s, xo = compose(img_a, img_b)
        stem = f"{a['id'].replace(':', '_')}__{r['group']}"
        path = IMG_DIR / f"{stem}.jpg"
        if not path.is_file():
            canvas.save(path, quality=90)
        W, H = canvas.size
        b_rows = []
        for inst in rec_b["instances"]:
            vs = r["verdicts"].get(str(inst["iid"]))
            if not vs:
                continue
            x0, y0, x1, y1 = inst["box"]
            b_rows.append({"iid": 100 + inst["iid"], "box": [x0 * s + xo, y0 * s, x1 * s + xo, y1 * s],
                           "verdicts": [TR._v(x) for x in vs][: len(ma["conditions"])], "seen": {}, "origin": (r["run"], r["group"], inst["iid"])})
        if not b_rows or any(TR._fit(row["verdicts"]) for row in b_rows):
            continue
        base = {"image": str(path), "image_wh": [W, H], "group": f"mosaic:{a['group']}:{r['group']}", "category": a["category"], "label": a["label"],
                "run": a["run"], "source": "mosaic", "n_candidates": len(ma["rows"]) + len(b_rows)}
        # positive: A's description on the mosaic
        m_pos = {**ma, "rows": [dict(row) for row in ma["rows"]] + b_rows}
        out.append({**base, "id": f"mosaic:{a['id']}:{r['group']}", "kind": "positive", "expression": a["expression"],
                    "answer": {"bbox_2d": a["answer"]["bbox_2d"]}, "target_iid": a["target_iid"], "matrix_pre": m_pos})
        n_pos += 1
        # negative: A's flipped description, if A's scene exported one and the matrix supports it
        neg = negs_by_group.get(a["group"]) if a["kind"] == "positive" else None
        if neg is not None:
            mn, _ = TR.label_matrix(neg)
            if mn is not None and mn.get("flipped_idx") is not None:
                f = mn["flipped_idx"]
                b_neg = []
                for row in b_rows:
                    vs = list(row["verdicts"])
                    if f < len(vs):
                        vs[f] = "unclear"
                    if "no" in vs:
                        b_neg.append({**row, "verdicts": vs})
                if len(b_neg) == len(b_rows):
                    m_neg = {**mn, "rows": [dict(row) for row in mn["rows"]] + b_neg}
                    out.append({**base, "id": f"mosaic:{neg['id']}:{r['group']}", "kind": "negative", "expression": neg["expression"],
                                "answer": {"bbox_2d": None}, "target_iid": None, "flipped": neg.get("flipped"), "matrix_pre": m_neg})
                    n_neg += 1
    with open(OUT, "w", encoding="utf-8") as fh:
        for it in out:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{n_pos} positives + {n_neg} negatives -> {OUT}")


if __name__ == "__main__":
    main()
