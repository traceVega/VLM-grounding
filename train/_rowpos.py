"""Where does the chosen row sit vs the row holding the object? (confirmation-bias check on two-turn screenings)"""
import collections, json, sys
from shared.harness import parsers as P
from train import eval_suite as E, traces as TR

tag = sys.argv[1] if len(sys.argv) > 1 else "scr_sft_v2_b400"
gt = {it["id"]: it for it in E.gme_subset(100, 0, 100)}
rows = [json.loads(l) for l in open(f"/home/jiaqi/vlmg-data/train/eval/{tag}/gme.jsonl") if l.strip()]
pos = [r for r in rows if r["dimension"] != "Rejection" and r["id"] in gt]
print("positives matched to GT:", len(pos), "of", sum(r["dimension"] != "Rejection" for r in rows))
chosen_pos, obj_pos, wrong_first, wrong_total, null_objrow = collections.Counter(), collections.Counter(), 0, 0, collections.Counter()
for r in pos:
    g = gt[r["id"]]
    boxes = g.get("gt_boxes") or []
    if not boxes:
        continue
    pr = TR.parse(r["raw"], tuple(E._open(g).size))
    cands = pr["candidates"]
    if not cands:
        continue
    ious = [max(P.iou(tuple(c["box"]), tuple(b)) for b in boxes) if c.get("box") else 0.0 for c in cands]
    obj_idx = max(range(len(cands)), key=lambda i: ious[i]) if max(ious) >= 0.5 else None
    if obj_idx is not None:
        obj_pos[obj_idx + 1] += 1
    if pr["derived_type"] == "box" and pr.get("derived_cand") is not None:
        ci = cands.index(pr["derived_cand"]) + 1 if pr["derived_cand"] in cands else None
        if ci:
            chosen_pos[ci] += 1
            if obj_idx is not None and ci != obj_idx + 1:
                wrong_total += 1
                wrong_first += ci == 1
    elif pr["derived_type"] == "null" and obj_idx is not None:
        null_objrow[obj_idx + 1] += 1
        if sum(1 for i in range(len(cands)) if not any(v == "no" for v in cands[i]["verdicts"].values())) == 0:
            pass
print("object sits in row:", dict(sorted(obj_pos.items())))
print("chosen row:", dict(sorted(chosen_pos.items())))
print(f"wrong choices with the object among candidates: {wrong_total}, of which row 1 chosen: {wrong_first}")
print("nulled positives, object row:", dict(sorted(null_objrow.items())))
# how often does the object's row carry an explicit 'no' (false accusation) when the item is nulled or wrong?
fa = 0; n = 0
for r in pos:
    g = gt[r["id"]]; boxes = g.get("gt_boxes") or []
    pr = TR.parse(r["raw"], tuple(E._open(g).size))
    for c in pr["candidates"]:
        if c.get("box") and boxes and max(P.iou(tuple(c["box"]), tuple(b)) for b in boxes) >= 0.5:
            n += 1; fa += any(v == "no" for v in c["verdicts"].values())
print(f"object rows with an explicit no: {fa}/{n}")
