"""Offline: derived answer with the 'named no' rule (a 'no' excludes only when its reason names an observed value / contrast)."""
import json, re, sys
from shared.harness import parsers as P
from train import eval_suite as E, traces as TR
CONTRAST = re.compile(r"\bnot\b|\binstead\b|\brather\b|\bno\b\s+\w+|\bwithout\b|\bmissing\b|\babsent\b", re.I)
tags = sys.argv[1:] or ["scr_sft_hint_v3_s", "scr_sft_v2_b400", "scr_sft_v2_half_b400", "scr_sft_v2h_rand", "scr_sft_v2_ov", "scr_grpo_v2"]
gt = {it["id"]: it for it in E.gme_subset(100, 0, 100)}
whs = {}
def decide(pr, rule):
    J = len(pr["conditions"]); cands = [c for c in pr["candidates"] if c.get("box")]
    if not cands or J == 0: return None
    def eff_no(c, j):
        return c["verdicts"].get(j) == "no" and (rule == "current" or bool(CONTRAST.search(c["seen"].get(j) or "")))
    fits = [c for c in cands if not any(eff_no(c, j) for j in range(1, J + 1))]
    if not fits: return None
    return max(fits, key=lambda c: sum(c["verdicts"].get(j) == "yes" for j in range(1, J + 1)))["box"]
for tag in tags:
    rows = [json.loads(l) for l in open(f"/home/jiaqi/vlmg-data/train/eval/{tag}/gme.jsonl") if l.strip()]
    parsed = []
    for r in rows:
        g = gt.get(r["id"])
        if g is None: continue
        wh = whs.setdefault(r["id"], E._open(g).size)
        parsed.append((r, g, TR.parse(r["raw"], tuple(wh))))
    bare = named = 0
    for r, g, pr in parsed:
        for c in pr["candidates"]:
            for j, v in c["verdicts"].items():
                if v == "no":
                    if CONTRAST.search(c["seen"].get(j) or ""): named += 1
                    else: bare += 1
    line = f"{tag:22s} no-cells named {named} bare {bare} |"
    for rule in ("current", "named"):
        rej_ok = rej_n = pos_ok = pos_null = pos_n = 0
        for r, g, pr in parsed:
            ans = decide(pr, rule) if pr["format_ok"] else None
            if r["dimension"] == "Rejection":
                rej_n += 1; rej_ok += ans is None
            else:
                pos_n += 1
                if ans is None: pos_null += 1
                elif any(P.iou(tuple(ans), tuple(b)) >= 0.5 for b in g["gt_boxes"]): pos_ok += 1
        line += f" {rule}: rej {100*rej_ok/rej_n:4.0f} pos {100*pos_ok/pos_n:4.0f} null {100*pos_null/pos_n:4.0f} net {100*(rej_ok/rej_n - pos_null/pos_n):5.1f} |"
    print(line)
