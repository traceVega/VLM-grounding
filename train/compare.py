"""Side-by-side of evaluation summaries:  python -m train.compare base4b sft_v1 [more tags]"""

from __future__ import annotations

import json
import sys

from train import data as D


def get(d: dict, path: str):
    for k in path.replace("_0.1", "_0,1").split("."):
        k = k.replace("_0,1", "_0.1")
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


ROWS = [
    ("GroundingME 拒答 201 准确率", "gme.rejection_acc"),
    ("GroundingME 拒答 p(null) 总和", "gme.rejection_p.p_null_sum"),
    ("GroundingME 拒答 p(null)≥0.1 比例", "gme.rejection_p.p_null_frac_ge_0.1"),
    ("GroundingME 非拒答 804 准确率", "gme.positive_acc"),
    ("GroundingME 非拒答 误拒率", "gme.positive_null_rate"),
    ("  Limited", "gme.by_dimension.Limited"),
    ("  Spatial", "gme.by_dimension.Spatial"),
    ("  Discriminative", "gme.by_dimension.Discriminative"),
    ("GroundingME 灰图 拒答项 null 率", "gmegray.rejection_null_rate"),
    ("GroundingME 灰图 正样本 null 率", "gmegray.positive_null_rate"),
    ("自家验证 负样本拒答率", "own.neg_null_rate"),
    ("自家验证 负样本 p(null)≥0.1 比例", "own.neg_p.p_null_frac_ge_0.1"),
    ("自家验证 正样本准确率", "own.pos_acc"),
    ("自家验证 正样本误拒率", "own.pos_null_rate"),
    ("灰图 负样本拒答率", "gray.neg_null_rate"),
    ("灰图 正样本拒答率", "gray.pos_null_rate"),
    ("RefCOCO 准确率", "refcoco.RefCOCO.acc"),
    ("RefCOCO+ 准确率", "refcoco.RefCOCOplus.acc"),
    ("RefCOCOg 准确率", "refcoco.RefCOCOg.acc"),
    ("RefCOCO 系列误拒率（均值）", None),
]


def main() -> None:
    tags = sys.argv[1:]
    sums = {t: json.loads((D.TRAIN_ROOT / "eval" / t / "summary.json").read_text(encoding="utf-8")) for t in tags}
    w = max(len(r[0]) for r in ROWS) + 2
    print("| 指标".ljust(w) + "".join(f"| {t:>12} " for t in tags) + "|")
    print("|" + "-" * (w - 1) + "".join("|" + "-" * 14 for _ in tags) + "|")
    for label, path in ROWS:
        cells = []
        for t in tags:
            if path is None:
                rc = get(sums[t], "refcoco") or {}
                vals = [v.get("null_rate") for v in rc.values() if v.get("null_rate") is not None]
                v = sum(vals) / len(vals) if vals else None
            else:
                v = get(sums[t], path)
            cells.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        print(f"| {label}".ljust(w) + "".join(f"| {c:>12} " for c in cells) + "|")


if __name__ == "__main__":
    main()
