"""Probe diagnostics from the synthetic <candidates> tables (per candidate: fits / excluded with a named mismatch)."""
import json, sys
from shared.harness import parsers as P
from train import eval_suite as E, traces as TR
tag = sys.argv[1]
gt = {it["id"]: it for it in E.gme_subset(100, 0, 100)}
rows = [json.loads(l) for l in open(f"/home/jiaqi/vlmg-data/train/eval/{tag}/gme.jsonl") if l.strip()]
gt_acc = gt_n = oth_acc = oth_n = 0; rank = {1: [0, 0], 2: [0, 0]}
rej_top1 = rej_any = rej_n = 0; k0 = 0
kcurve = {K: [0, 0, 0, 0, 0] for K in range(1, 7)}
free_vs = [0, 0]
for r in rows:
    g = gt.get(r["id"])
    if g is None: continue
    raw = r["raw"]; i = raw.find("<coa>"); raw = raw[:i] if i >= 0 else raw
    pr = TR.parse(raw, E._open(g).size)
    cands = [c for c in pr["candidates"] if c.get("box")]
    if not cands: k0 += 1
    excl = [any(v == "no" for v in c["verdicts"].values()) for c in cands]
    free_vs[0] += 1; free_vs[1] += pr["free_type"] == pr["derived_type"]
    if r["dimension"] == "Rejection":
        rej_n += 1; rej_top1 += bool(excl) and excl[0]; rej_any += any(excl)
        for K in kcurve:
            kcurve[K][1] += 1; kcurve[K][0] += (not cands) or all(excl[:K])
    elif g.get("gt_boxes"):
        for k, (c, e) in enumerate(zip(cands, excl)):
            hit = max(P.iou(tuple(c["box"]), tuple(b)) for b in g["gt_boxes"]) >= 0.5
            if hit:
                gt_n += 1; gt_acc += e; key = 1 if k == 0 else 2; rank[key][0] += e; rank[key][1] += 1
            else:
                oth_n += 1; oth_acc += e
        for K in kcurve:
            kcurve[K][4] += 1
            ans = next((c["box"] for c, e in list(zip(cands, excl))[:K] if not e), None)
            if ans is None: kcurve[K][3] += 1
            elif any(P.iou(tuple(ans), tuple(b)) >= 0.5 for b in g["gt_boxes"]): kcurve[K][2] += 1
print(f"[{tag}] positives: GT candidate falsely accused {gt_acc}/{gt_n} = {gt_acc/max(1,gt_n):.2f} (rank1 {rank[1][0]}/{rank[1][1]}, rank>=2 {rank[2][0]}/{rank[2][1]}); other candidates accused {oth_acc}/{oth_n} = {oth_acc/max(1,oth_n):.2f}")
print(f"   rejection: top-1 candidate accused {rej_top1}/{rej_n} = {rej_top1/max(1,rej_n):.2f}; any candidate accused {rej_any}/{rej_n}; K=0 items {k0}; model answer == derived {free_vs[1]/max(1,free_vs[0]):.2f}")
for K, (ro, rn, po, pn, pnn) in kcurve.items():
    print(f"   K<={K}: derived rej {100*ro/rn:4.0f} pos {100*po/pnn:4.0f} null {100*pn/pnn:4.0f} net {100*(ro/rn - pn/pnn):5.1f}")
