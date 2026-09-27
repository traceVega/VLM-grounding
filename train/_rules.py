"""Offline re-scoring of saved two-turn outputs under alternative table-to-answer rules (GME screening subset)."""
import json, sys
from shared.harness import parsers as P
from train import eval_suite as E, traces as TR

tags = sys.argv[1:] or ["scr_sft_v2_b400", "scr_sft_v2_half_b400", "scr_sft_v2h_rand"]
gt = {it["id"]: it for it in E.gme_subset(100, 0, 100)}
whs = {}

def counts(c):
    v = list(c["verdicts"].values())
    return sum(x == "yes" for x in v), sum(x == "no" for x in v), sum(x == "unclear" for x in v)

def decide(cands, rule):
    """rule = (k_exclude, w_no, need_margin): a candidate is excluded when it has >= k_exclude 'no';
    among the rest pick max(yes - w_no*no); with need_margin, null unless the best is unique."""
    k, w, margin = rule
    alive = [(c, counts(c)) for c in cands if counts(c)[1] < k]
    if not alive:
        return None
    scored = sorted(alive, key=lambda t: -(t[1][0] - w * t[1][1]))
    if margin and len(scored) > 1 and (scored[0][1][0] - w * scored[0][1][1]) == (scored[1][1][0] - w * scored[1][1][1]):
        return None
    return scored[0][0].get("box")

rules = {"current(any no->out, most yes)": (1, 0.0, False), "k=2 (out only with 2+ no)": (2, 0.0, False),
         "k=2, w=1": (2, 1.0, False), "k=2, w=1, margin": (2, 1.0, True), "k=3, w=1": (3, 1.0, False), "k=1, margin": (1, 0.0, True)}
for tag in tags:
    rows = [json.loads(l) for l in open(f"/home/jiaqi/vlmg-data/train/eval/{tag}/gme.jsonl") if l.strip()]
    parsed = []
    for r in rows:
        g = gt.get(r["id"])
        if g is None:
            continue
        wh = whs.setdefault(r["id"], E._open(g).size)
        pr = TR.parse(r["raw"], tuple(wh))
        parsed.append((r, g, pr))
    print(f"== {tag}")
    for name, rule in rules.items():
        rej_ok = rej_n = pos_ok = pos_null = pos_n = 0
        for r, g, pr in parsed:
            ans = decide(pr["candidates"], rule) if pr["format_ok"] else None
            if r["dimension"] == "Rejection":
                rej_n += 1; rej_ok += ans is None
            else:
                pos_n += 1
                if ans is None:
                    pos_null += 1
                elif any(P.iou(tuple(ans), tuple(b)) >= 0.5 for b in g["gt_boxes"]):
                    pos_ok += 1
        print(f"  {name:32s} rej {100*rej_ok/rej_n:5.1f}  pos {100*pos_ok/pos_n:5.1f}  null {100*pos_null/pos_n:5.1f}  net {100*(rej_ok/rej_n - pos_null/pos_n):5.1f}")
