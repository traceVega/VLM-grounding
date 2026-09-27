"""Per-item diagnostics of a two-turn screening run against the single-crop baseline (GME subset)."""
import collections, json, sys

tag = sys.argv[1] if len(sys.argv) > 1 else "scr_sft_v2"
base_tag = sys.argv[2] if len(sys.argv) > 2 else "scr_sft_hint_v3_s"


def load(t):
    return {r["id"]: r for r in (json.loads(l) for l in open(f"/home/jiaqi/vlmg-data/train/eval/{t}/gme.jsonl") if l.strip())}


v2, base = load(tag), load(base_tag)
pos = [r for r in v2.values() if r["dimension"] != "Rejection"]
rej = [r for r in v2.values() if r["dimension"] == "Rejection"]


def hit(r):
    return r.get("cand_recall_iou") is not None and r["cand_recall_iou"] >= 0.5


print(f"[{tag}] positives {len(pos)}: candidate recall>=0.5 {sum(hit(r) for r in pos) / len(pos):.2f}, "
      f"mean tool boxes {sum(r['n_tool_boxes'] for r in pos) / len(pos):.2f}, no boxes {sum(r['n_tool_boxes'] == 0 for r in pos)}")
print(f"   baseline hint IoU>=0.5 on the same positives: {sum((base[r['id']].get('hint_iou') or 0) >= 0.5 for r in pos if r['id'] in base) / max(1, sum(r['id'] in base for r in pos)):.2f}")
out = collections.Counter()
for r in pos:
    k = ("correct" if r["correct"] else ("null" if r["output_type"] in ("null", "none", "reject", "null_json") else "wrong-box")) + ("/recall-hit" if hit(r) else "/recall-miss")
    out[k] += 1
print("   outcomes:", dict(out))
for dim in ("Discriminative", "Limited", "Spatial"):
    rs = [r for r in pos if r["dimension"] == dim]
    if rs:
        print(f"   {dim:14s} n={len(rs):2d} acc {sum(r['correct'] for r in rs) / len(rs):.2f} null {sum(not r['correct'] and r['output_type'] not in ('box',) for r in rs) / len(rs):.2f} recall {sum(hit(r) for r in rs) / len(rs):.2f} cands {sum(r['n_tool_boxes'] for r in rs) / len(rs):.1f}")
for sb in sorted({r.get("size_bin") for r in pos}):
    rs = [r for r in pos if r.get("size_bin") == sb]
    print(f"   size {str(sb):8s} n={len(rs):2d} acc {sum(r['correct'] for r in rs) / len(rs):.2f} recall {sum(hit(r) for r in rs) / len(rs):.2f}")
print(f"[{tag}] rejection {len(rej)}: null rate {sum(r['correct'] for r in rej) / len(rej):.2f}; mean tool boxes {sum(r['n_tool_boxes'] for r in rej) / len(rej):.2f}; "
      f"null with 0 boxes {sum(r['correct'] and r['n_tool_boxes'] == 0 for r in rej)}, null with boxes {sum(r['correct'] and r['n_tool_boxes'] > 0 for r in rej)}")
both = sum(r["correct"] and base[r["id"]]["correct"] for r in rej if r["id"] in base)
only_v2 = sum(r["correct"] and not base[r["id"]]["correct"] for r in rej if r["id"] in base)
only_b = sum((not r["correct"]) and base[r["id"]]["correct"] for r in rej if r["id"] in base)
print(f"   paired vs baseline: both {both}, only {tag} {only_v2}, only baseline {only_b}")
bothp = sum(r["correct"] and base[r["id"]]["correct"] for r in pos if r["id"] in base)
only_v2p = sum(r["correct"] and not base[r["id"]]["correct"] for r in pos if r["id"] in base)
only_bp = sum((not r["correct"]) and base[r["id"]]["correct"] for r in pos if r["id"] in base)
print(f"   positives paired: both {bothp}, only {tag} {only_v2p}, only baseline {only_bp}")
ex = next((r for r in pos if not r["correct"] and r["output_type"] != "box" and hit(r)), None)
if ex:
    print("\n--- a positive nulled although the object was among the candidates ---")
    print(ex["raw"][:1600])
ex = next((r for r in pos if r["dimension"] == "Spatial" and not r["correct"]), None)
if ex:
    print("\n--- a Spatial failure ---")
    print(ex["raw"][:1400])
