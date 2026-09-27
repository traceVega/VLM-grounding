"""Line up our re-scored run against the paper's Table 3 / Table 8 rows for
Qwen3-VL-8B, at L1 and L2, under both mAcc definitions."""
import csv, json, collections, math

recs = json.load(open("/tmp/gme_rescored.json"))
l2 = {r["item_id"]: (r["subtask_l1"], r["subtask_l2"]) for r in
      csv.DictReader(open("/home/jiaqi/vlmg-data/prepared/groundingme/dimensions.csv"))}
for r in recs:
    r["l1"], r["l2"] = l2[r["id"]]

T9  = [0.5 + 0.05 * i for i in range(9)]    # [0.5, 0.9]  (the follow-up's definition)
T10 = [0.5 + 0.05 * i for i in range(10)]   # [0.5, 0.95] (the paper's definition, A.2)


def acc(rs, k, t):
    return 100.0 * sum(1 for r in rs if r[k] >= t) / len(rs) if rs else float("nan")


def macc(rs, k, T):
    return sum(acc(rs, k, t) for t in T) / len(T)


PAPER_L1 = {  # Table 3 (Acc@0.5) and Table 8 (Acc0.75, Acc0.9, mAcc[0.5,0.95])
    "Discriminative": (61.3, 57.4, 47.1, 55.0),
    "Spatial":        (26.3, 23.7, 21.0, 23.3),
    "Limited":        (36.0, 27.3, 16.7, 26.5),
    "Rejection":      (0.0, 0.0, 0.0, 0.0),
    "TOTAL":          (31.0, 26.9, 20.8, 26.0),
}
PAPER_L2 = {  # Table 3, Acc@0.5
    ("Discriminative", "Appearance"): 55.8, ("Discriminative", "Component"): 68.0,
    ("Discriminative", "Text"): 80.0, ("Discriminative", "State"): 42.3,
    ("Spatial", "Relationship"): 32.7, ("Spatial", "Counting"): 20.0,
    ("Limited", "Occlusion"): 56.0, ("Limited", "Small"): 16.0,
    ("Rejection", "Appearance"): 0.0, ("Rejection", "Component"): 0.0,
    ("Rejection", "Text"): 0.0, ("Rejection", "State"): 0.0,
}

by1 = collections.defaultdict(list)
by2 = collections.defaultdict(list)
for r in recs:
    by1[r["l1"]].append(r)
    by2[(r["l1"], r["l2"])].append(r)
by1["TOTAL"] = recs

print("L1: ours (pinned relative_1000) vs paper Qwen3-VL-8B, arXiv 2512.17495 T3/T8")
print("%-16s%5s | %-27s | %-27s | %s" % ("", "n", "        OURS", "       PAPER", "  delta (A50/A75/A90/mAcc)"))
print("%-16s%5s | %6s%7s%7s%7s | %6s%7s%7s%7s |" %
      ("", "", "A@.5", "A@.75", "A@.9", "mAcc", "A@.5", "A@.75", "A@.9", "mAcc"))
for k in ["Discriminative", "Spatial", "Limited", "Rejection", "TOTAL"]:
    rs = by1[k]
    o = (acc(rs, "pin", 0.5), acc(rs, "pin", 0.75), acc(rs, "pin", 0.9), macc(rs, "pin", T10))
    p = PAPER_L1[k]
    print("%-16s%5d | %6.2f%7.2f%7.2f%7.2f | %6.1f%7.1f%7.1f%7.1f | %+6.2f%+7.2f%+7.2f%+7.2f" %
          ((k, len(rs)) + o + p + tuple(o[i] - p[i] for i in range(4))))

print("\nL2 Acc@0.5: ours vs paper Table 3")
print("%-16s%-14s%5s%9s%9s%9s" % ("L1", "L2", "n", "ours", "paper", "delta"))
for (a, b), pv in PAPER_L2.items():
    rs = by2[(a, b)]
    ov = acc(rs, "pin", 0.5)
    print("%-16s%-14s%5d%9.2f%9.1f%+9.2f" % (a, b, len(rs), ov, pv, ov - pv))

print("\nmAcc definition matters:")
for k in ["Discriminative", "Spatial", "Limited", "TOTAL"]:
    rs = by1[k]
    print("  %-16s mAcc[0.5,0.90] (9 thr) %6.2f   mAcc[0.5,0.95] (10 thr, paper) %6.2f"
          % (k, macc(rs, "pin", T9), macc(rs, "pin", T10)))

print("\nheadline choice, all 1005 items, ours:")
for k, T in (("Acc@0.5", None), ("mAcc[0.5,0.90]", T9), ("mAcc[0.5,0.95]", T10)):
    v = acc(recs, "pin", 0.5) if T is None else macc(recs, "pin", T)
    print("   %-16s %6.2f" % (k, v))
print("   Acc@0.9         %6.2f" % acc(recs, "pin", 0.9))

# how much the metric choice compresses the dimension spread
for name, f in (("Acc@0.5", lambda rs: acc(rs, "pin", 0.5)),
                ("mAcc[.5,.95]", lambda rs: macc(rs, "pin", T10))):
    d, s, l = (f(by1["Discriminative"]), f(by1["Spatial"]), f(by1["Limited"]))
    print("   %-14s Dis %.2f  Spa %.2f  Lim %.2f   Lim-Spa gap %+.2f" % (name, d, s, l, l - s))

# paired-comparison arithmetic for the 2.1-point reconciliation
n = 1005
d = 33.13 - 31.0
print("\nreconciliation arithmetic")
print("  2.13 points on n=1005 is %.1f items" % (d / 100 * n))
print("  unpaired Wald CI on one run's Acc@0.5: +/-%.2f"
      % (1.96 * math.sqrt(0.3313 * 0.6687 / n) * 100))
for disc in (60, 100, 150, 200, 300):
    se = math.sqrt(disc) / n * 100
    print("    if the two harnesses disagree on %3d of 1005 items, McNemar SE on the "
          "difference is %.2f pts -> 2.13 pts is %.1f SE" % (disc, se, d / se))
print("  paper's own 5-prompt spread for this model: 31.36 +/- 2.02 (arXiv 2512.17495 C.2)")
