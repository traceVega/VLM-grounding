"""Diagnostics of a COA run (GME subset): audit format compliance, echo rate, named-mismatch rates on the GT-matching
candidate vs other candidates, rank split, model-answer vs rule-derived agreement, and the K curve (rule-derived answer
restricted to the first K proposals)."""
import json, sys
from shared.harness import parsers as P
from train import coa, eval_suite as E, traces as TR

tag = sys.argv[1]
gt = {it["id"]: it for it in E.gme_subset(100, 0, 100)}
rows = [json.loads(l) for l in open(f"/home/jiaqi/vlmg-data/train/eval/{tag}/gme.jsonl") if l.strip()]
n_aud = n_ok = n_lines = n_echo = n_unsure = n_mis = 0
gt_named = gt_n = other_named = other_n = 0
rank_named = {1: [0, 0], 2: [0, 0]}  # rank 1 vs rank>=2 of the GT candidate: [named, n]
rej_named_any = rej_n = rej_top1_named = 0
k0 = unparsable_items = 0
agree = n_items = 0
kcurve = {K: [0, 0, 0, 0, 0] for K in range(1, 7)}  # rej_ok, rej_n, pos_ok, pos_null, pos_n
for r in rows:
    raw = r["raw"]; i = raw.find("<coa>")
    extra = r.get("coa")
    if extra is None:
        continue
    if i < 0:
        i = len(raw)
    g = gt.get(r["id"])
    if g is None:
        continue
    wh = E._open(g).size
    audits = [coa.parse_audit(t) for t in extra["audits"]]
    k0 += bool(extra.get("k0")); unparsable_items += bool(extra.get("unparsable"))
    for a in audits:
        n_aud += 1; n_ok += a["format_ok"]
        for ln in a["lines"]:
            n_lines += 1; n_echo += coa.norm(ln["seen"]) == coa.norm(ln["claimed"]); n_unsure += ln["verdict"] == "unsure"; n_mis += ln["verdict"] == "mismatch"
    pr = TR.parse(raw[:i], wh)
    n_items += 1; agree += pr["free_type"] == pr["derived_type"]
    cands = [c for c in pr["candidates"] if c.get("box")]
    boxes = [c["box"] for c in cands]
    if r["dimension"] == "Rejection":
        rej_n += 1
        rej_named_any += any(coa.named_mismatches(a["lines"]) for a in audits if a["format_ok"])
        rej_top1_named += bool(audits) and audits[0]["format_ok"] and bool(coa.named_mismatches(audits[0]["lines"]))
        for K in kcurve:
            ans, _, _ = coa.decide(audits[:K], boxes[:K]) if boxes else (None, None, [])
            kcurve[K][1] += 1; kcurve[K][0] += ans is None
    elif g.get("gt_boxes"):
        for k, c in enumerate(cands):
            if k >= len(audits) or not audits[k]["format_ok"]:
                continue
            hit = max(P.iou(tuple(c["box"]), tuple(b)) for b in g["gt_boxes"]) >= 0.5
            nm = bool(coa.named_mismatches(audits[k]["lines"]))
            if hit:
                gt_n += 1; gt_named += nm
                key = 1 if k == 0 else 2
                rank_named[key][0] += nm; rank_named[key][1] += 1
            else:
                other_n += 1; other_named += nm
        for K in kcurve:
            ans, _, _ = coa.decide(audits[:K], boxes[:K]) if boxes else (None, None, [])
            kcurve[K][4] += 1
            if ans is None:
                kcurve[K][3] += 1
            elif any(P.iou(tuple(ans), tuple(b)) >= 0.5 for b in g["gt_boxes"]):
                kcurve[K][2] += 1
print(f"[{tag}] items {n_items}: audits {n_aud} parseable {n_ok/max(1,n_aud):.2f}; lines {n_lines}: echo {n_echo/max(1,n_lines):.2f}, unsure {n_unsure/max(1,n_lines):.2f}, mismatch {n_mis/max(1,n_lines):.2f}; K=0 items {k0}; items with an unparsable audit {unparsable_items}")
print(f"   positives: GT candidate falsely accused {gt_named}/{gt_n} = {gt_named/max(1,gt_n):.2f} (rank1 {rank_named[1][0]}/{rank_named[1][1]}, rank>=2 {rank_named[2][0]}/{rank_named[2][1]}); other candidates accused {other_named}/{other_n} = {other_named/max(1,other_n):.2f}")
print(f"   rejection: any candidate accused {rej_named_any}/{rej_n}; top-1 candidate accused {rej_top1_named}/{rej_n}")
print(f"   model answer == rule-derived answer: {agree/max(1,n_items):.2f}")
for K, (ro, rn, po, pn, pnn) in kcurve.items():
    if rn and pnn:
        print(f"   K<={K}: rule-derived rej {100*ro/rn:4.0f} pos {100*po/pnn:4.0f} null {100*pn/pnn:4.0f} net {100*(ro/rn - pn/pnn):5.1f}")
