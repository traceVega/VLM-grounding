"""Print a run's unlock curve:  python -m train.curve sft_v1 [more names]"""

from __future__ import annotations

import json
import sys

from train import data as D


def main() -> None:
    for name in sys.argv[1:]:
        f = D.TRAIN_ROOT / name / "curve.jsonl"
        print(f"== {name}")
        if not f.is_file():
            print("   (no curve yet)")
            continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            g = r.get("gray_neg_null_rate")
            print(f"  step {r['step']:3d} | neg null {r['neg_null_rate']:.2f}  p(null) med {r['neg_p_null_median']:.2e}  >=0.1 {r['neg_p_null_frac_ge_0.1']:.2f}"
                  f" | pos acc {r['pos_acc']:.2f}  pos null {r['pos_null_rate']:.2f} | gray neg null {g if g is None else round(g, 2)}")


if __name__ == "__main__":
    main()
