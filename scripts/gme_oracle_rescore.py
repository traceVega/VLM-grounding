"""Re-score tables/gme_original.jsonl under the official GroundingME per-item
oracle vs a pinned relative_1000 convention. Zero GPU; CPU only."""
import json, math, collections
import pyarrow.parquet as pq

REPO = "/mnt/d/Dev/ArcNova/auto-research/VLM-grounding"
t = pq.read_table("/home/jiaqi/vlmg-data/prepared/groundingme/items.parquet")
col = {n: t.column(n).to_pylist() for n in t.column_names}
gt  = {col["item_id"][i]: col["gt_boxes_xyxy_px"][i] for i in range(t.num_rows)}
dim = {col["item_id"][i]: col["dimension"][i] for i in range(t.num_rows)}
sb  = {col["item_id"][i]: col["size_bin"][i] for i in range(t.num_rows)}
ngt = {col["item_id"][i]: col["n_gt"][i] for i in range(t.num_rows)}


def iou(a, b):
    xl = max(a[0], b[0]); yt = max(a[1], b[1])
    xr = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0.0, xr - xl) * max(0.0, yb - yt)
    a1 = (a[2] - a[0]) * (a[3] - a[1]); a2 = (b[2] - b[0]) * (b[3] - b[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def normalize_bbox(bbox, width, height):   # verbatim from official evaluate.py
    if all(c <= 1 for c in bbox):
        return [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
    return [bbox[0] / 999 * width, bbox[1] / 999 * height,
            bbox[2] / 999 * width, bbox[3] / 999 * height]


rows = [json.loads(l) for l in open(REPO + "/tables/gme_original.jsonl", encoding="utf-8") if l.strip()]
print("records: %d" % len(rows))

recs = []
int_ok = 0
raw_over = 0
nbox = 0
for r in rows:
    iid = r["item_id"]; w, h = r["image_wh"]
    g = gt[iid]; box = r["box"]
    if box is None:
        raw = None
    else:
        nbox += 1
        raw = [box[0] * 1000.0 / w, box[1] * 1000.0 / h,
               box[2] * 1000.0 / w, box[3] * 1000.0 / h]
        if all(abs(c - round(c)) < 1e-6 for c in raw):
            int_ok += 1
        if any(c > 1000.0 + 1e-6 for c in raw):
            raw_over += 1

    if ngt[iid] == 0:
        pred_is_null = (box is None)
        i_pin = 1.0 if pred_is_null else 0.0
        i_abs = i_nor = i_orc = i_pin
    elif box is None:
        i_pin = i_abs = i_nor = i_orc = 0.0
    else:
        cand_abs = list(raw)
        cand_nor = normalize_bbox(raw, w, h)
        i_pin = max(iou(box, x) for x in g)
        i_abs = max(iou(cand_abs, x) for x in g)
        i_nor = max(iou(cand_nor, x) for x in g)
        i_orc = max(i_abs, i_nor)
    recs.append(dict(id=iid, dim=dim[iid], size=sb[iid], ngt=ngt[iid],
                     out=r["output_type"], pin=i_pin, abs=i_abs, nor=i_nor,
                     orc=i_orc, team=r["iou_gt"]))

print("raw recovers to integers: %d/%d   raw>1000: %d" % (int_ok, nbox, raw_over))
mx = max(abs(r["pin"] - (r["team"] or 0.0)) for r in recs if r["ngt"] == 1)
print("max |pinned - stored iou_gt| on positives: %.3e" % mx)

THR = [0.5 + 0.05 * i for i in range(9)]


def acc(rs, key, t):
    return 100.0 * sum(1 for r in rs if r[key] >= t) / len(rs) if rs else float("nan")


def macc(rs, key):
    return sum(acc(rs, key, t) for t in THR) / len(THR)


def table(label, groups):
    print("\n=== %s ===" % label)
    print("%-16s%5s  %-30s | %-30s | %s" % ("group", "n",
          "  PINNED  50/75/90/mAcc", "  ORACLE  50/75/90/mAcc", " d50   dmAcc"))
    for name, rs in groups:
        if not rs:
            continue
        p = [acc(rs, "pin", 0.5), acc(rs, "pin", 0.75), acc(rs, "pin", 0.9), macc(rs, "pin")]
        o = [acc(rs, "orc", 0.5), acc(rs, "orc", 0.75), acc(rs, "orc", 0.9), macc(rs, "orc")]
        print("%-16s%5d  %-30s | %-30s | %+5.2f %+6.2f" % (
            name, len(rs),
            "".join("%7.2f" % x for x in p),
            "".join("%7.2f" % x for x in o),
            o[0] - p[0], o[3] - p[3]))


order = ["Discriminative", "Spatial", "Limited", "Rejection"]
by = collections.defaultdict(list)
for r in recs:
    by[r["dim"]].append(r)
pos = [r for r in recs if r["ngt"] == 1]
table("Overall and per dimension",
      [("ALL (1005)", recs)] + [(d, by[d]) for d in order if d in by] +
      [("positives 804", pos)])

bys = collections.defaultdict(list)
for r in pos:
    bys[r["size"]].append(r)
table("Positives by size bin", [(k, bys[k]) for k in ["small", "medium", "large", "xl"] if k in bys])

gift = [r for r in pos if r["orc"] > r["pin"] + 1e-9]
flip = {t: [r for r in pos if r["orc"] >= t and r["pin"] < t] for t in (0.5, 0.75, 0.9)}
abs_wins = [r for r in pos if r["abs"] > r["nor"] + 1e-9]
print("\noracle strictly raises IoU on %d/%d positives" % (len(gift), len(pos)))
print("  absolute-pixel candidate beats the relative one on %d positives; %d of those clear 0.5"
      % (len(abs_wins), sum(1 for r in abs_wins if r["abs"] >= 0.5)))
print("  flipped to correct by the oracle: @0.5 %d  @0.75 %d  @0.9 %d"
      % (len(flip[0.5]), len(flip[0.75]), len(flip[0.9])))
print("  mean IoU positives: pinned %.4f  oracle %.4f"
      % (sum(r["pin"] for r in pos) / len(pos), sum(r["orc"] for r in pos) / len(pos)))

print("\n/999 (official relative) vs /1000 (pinned), positives only")
for t_ in (0.5, 0.75, 0.9):
    print("   @%.2f: pinned %6.2f  /999 %6.2f  abs-as-px %6.2f  oracle %6.2f"
          % (t_, acc(pos, "pin", t_), acc(pos, "nor", t_), acc(pos, "abs", t_), acc(pos, "orc", t_)))
print("   mAcc:  pinned %6.2f  /999 %6.2f  abs-as-px %6.2f  oracle %6.2f"
      % (macc(pos, "pin"), macc(pos, "nor"), macc(pos, "abs"), macc(pos, "orc")))
n999 = sum(1 for r in pos if (r["nor"] >= 0.5) != (r["pin"] >= 0.5))
print("   /999-vs-/1000 alone flips %d of 804 positives at IoU 0.5" % n999)

hits = [r for r in recs if r["pin"] >= 0.5]
hi = {t: [r for r in hits if r["pin"] >= t] for t in (0.75, 0.9)}
print("\nheadline pinned, all 1005: Acc@0.5 %.2f (%d hits); of those IoU>=0.75 %d (%.1f%%), "
      ">=0.9 %d (%.1f%%)" % (100 * len(hits) / len(recs), len(hits), len(hi[0.75]),
                             100 * len(hi[0.75]) / len(hits), len(hi[0.9]),
                             100 * len(hi[0.9]) / len(hits)))
print("all 1005: Acc@0.5 pinned %.2f oracle %.2f | Acc@0.75 %.2f/%.2f | Acc@0.9 %.2f/%.2f | mAcc %.2f/%.2f"
      % (acc(recs, "pin", 0.5), acc(recs, "orc", 0.5), acc(recs, "pin", 0.75), acc(recs, "orc", 0.75),
         acc(recs, "pin", 0.9), acc(recs, "orc", 0.9), macc(recs, "pin"), macc(recs, "orc")))
p = len(hits) / len(recs)
print("95%% CI (Wald) on Acc@0.5 over 1005: +/-%.2f" % (1.96 * math.sqrt(p * (1 - p) / len(recs)) * 100))
p2 = acc(recs, "pin", 0.5) / 100
mp = macc(recs, "pin") / 100
print("95%% CI on mAcc-as-a-rate over 1005: +/-%.2f" % (1.96 * math.sqrt(mp * (1 - mp) / len(recs)) * 100))

print("\noutput_type all 1005:", dict(collections.Counter(r["out"] for r in recs)))
rej = [r for r in recs if r["ngt"] == 0]
print("Rejection n=%d output_type:" % len(rej), dict(collections.Counter(r["out"] for r in rej)))
posn = [r for r in pos if r["out"] == "none"]
print("abstentions on 804 positives: %d (%.1f%%);" % (len(posn), 100 * len(posn) / len(pos)),
      dict(collections.Counter(r["dim"] for r in posn)))
inval = [r for r in recs if r["out"] == "invalid"]
print("unparseable rows anywhere: %d" % len(inval))

print("\nper-dimension Acc@0.5 pinned -> oracle, and mAcc")
for d in order:
    rs = by[d]
    print("  %-16s n=%4d  Acc@0.5 %6.2f -> %6.2f (%+.2f)   mAcc %6.2f -> %6.2f (%+.2f)"
          % (d, len(rs), acc(rs, "pin", 0.5), acc(rs, "orc", 0.5),
             acc(rs, "orc", 0.5) - acc(rs, "pin", 0.5),
             macc(rs, "pin"), macc(rs, "orc"), macc(rs, "orc") - macc(rs, "pin")))

json.dump(recs, open("/tmp/gme_rescored.json", "w"))
print("\nwrote /tmp/gme_rescored.json")
