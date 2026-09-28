"""Verification traces (notes/DESIGN-FINAL-LIGHT-2026-09-24.md section 2): rendered
deterministically from the datagen labels, parsed back from model output, and the
derived-answer rule.

    <conditions>
    1. is black
    2. has "RANGE ROVER" written on its hood
    </conditions>
    <candidates>
    <object id="1" bbox="[79, 339, 475, 797]">1: yes; 2: yes -> fits</object>
    <object id="2" bbox="[495, 320, 970, 740]">1: no; 2: yes -> excluded</object>
    </candidates>
    <answer>{"bbox_2d": [79, 339, 475, 797]}</answer>

Labels: candidate boxes and per-candidate per-clause verdicts come from the datagen records
(Gemini batched checker, reviewed scenes only); the negative's flipped clause carries the
original span as the observed value; sibling descriptions use the local 8B matrix
(train.sibling_check); cross-scene negatives use train/augment/verify.jsonl.

Derived answer: candidates with no explicit "no" fit; exactly one fits -> its box; several ->
the one with most "yes"; none -> null.  "unclear" never counts as "no".
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from datagen import common as C
from datagen.report import build_records
from shared.harness import parsers as P
from train import data as D

RUNS = ("test50c", "poc155")
MAX_CAND = 6
SIB_MATRIX = D.TRAIN_ROOT / "sibling_matrix.jsonl"
VERIFY = D.TRAIN_ROOT / "augment" / "verify.jsonl"
STYLES = ("yn", "obs", "rat")

_CACHE: dict = {}


def records() -> dict[tuple[str, str], dict]:
    if "rec" not in _CACHE:
        out = {}
        for run in RUNS:
            root = C.run_root(run)
            if (root / "scenes.jsonl").is_file():
                for r in build_records(root, "gemini-b"):
                    out[(run, r["image_id"])] = r
        _CACHE["rec"] = out
    return _CACHE["rec"]


def sibling_clauses() -> dict[tuple[str, str], list[str]]:
    if "sibcl" not in _CACHE:
        out = {}
        for run in RUNS:
            f = C.run_root(run) / "sibling.jsonl"
            if f.is_file():
                for r in C.read_jsonl(f):
                    out[(run, r["image_id"])] = r.get("clauses_sibling") or []
        _CACHE["sibcl"] = out
    return _CACHE["sibcl"]


def sibling_matrix() -> dict[tuple[str, str], dict[int, list[str]]]:
    if "sibmat" not in _CACHE:
        out = {}
        if SIB_MATRIX.is_file():
            for l in open(SIB_MATRIX, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    out[(r["run"], r["image_id"])] = {int(k): v for k, v in r["verdicts"].items()}
        _CACHE["sibmat"] = out
    return _CACHE["sibmat"]


def cross_rows() -> dict[str, dict]:
    if "cross" not in _CACHE:
        out = {}
        if VERIFY.is_file():
            for l in open(VERIFY, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    out[r["pair_id"]] = r
        _CACHE["cross"] = out
    return _CACHE["cross"]


def _v(x) -> str:
    return x if x in ("yes", "no", "unclear") else "unclear"


def _fit(verdicts: list[str]) -> bool:
    return "no" not in verdicts


def label_matrix(item: dict) -> tuple[dict | None, str]:
    """The label table of an exported item: conditions, candidate rows, answer iid, flipped
    clause.  Returns (matrix, reason) with matrix None when the labels cannot support a
    consistent trace (the reason says why)."""
    if item.get("source") in ("mosaic", "long", "audit", "gmestyle"):  # precomputed by train.mosaic_items / long_items / audit_rows / gme_export
        return item["matrix_pre"], "ok"
    rec = records().get((item["run"], item["group"]))
    if rec is None:
        return None, "no_record"
    insts = rec["instances"]
    kind = item["kind"]
    seen: dict[int, dict[int, str]] = {}
    fidx = None
    fclause = None
    flip_from = flip_to = None
    if item.get("source") == "cross":
        r = cross_rows().get(item["id"][len("cross:"):])
        if not r:
            return None, "no_cross_row"
        conds = list(r["clauses"])
        rows = [{"iid": i["iid"], "box": i["box"], "verdicts": [_v(x) for x in (r["verdicts"].get(str(i["iid"])) or [])]} for i in insts]
        ans = None
    elif item.get("source") == "alt":  # alternative flip of the primary negative (train.alt_negatives)
        conds = list(rec["clauses"])
        fidx = item["flipped"]["idx"]
        fclause = item["flipped"]["clause_neg"]
        if fidx is None or fidx >= len(conds):
            return None, "no_flip"
        conds[fidx] = fclause
        cv = rec.get("clause_verdicts") or {}
        av = item.get("alt_verdicts") or {}
        rows = []
        for i in insts:
            base = [_v(x) for x in (cv.get(str(i["iid"])) or [])]
            base = (base + ["unclear"] * len(conds))[: len(conds)]
            base[fidx] = _v(av.get(str(i["iid"])))
            rows.append({"iid": i["iid"], "box": i["box"], "verdicts": base})
        t = rec["target_iid"]
        if item["flipped"].get("from"):
            seen[t] = {fidx: item["flipped"]["from"]}
        flip_from, flip_to = item["flipped"].get("from"), item["flipped"].get("to")
        for row in rows:
            if row["iid"] == t:
                row["verdicts"][fidx] = "no"
        ans = None
    elif item.get("source") == "short":  # one or two matrix columns that name one instance (train.short_items)
        idxs = item["clause_idx"]
        conds = [rec["clauses"][j] for j in idxs]
        cv = rec.get("clause_verdicts") or {}
        rows = [{"iid": i["iid"], "box": i["box"], "verdicts": [_v((cv.get(str(i["iid"])) or [])[j] if j < len(cv.get(str(i["iid"])) or []) else None) for j in idxs]} for i in insts]
        ans = item["target_iid"]
    elif kind == "positive":
        conds = list(rec["clauses"])
        cv = rec.get("clause_verdicts") or {}
        rows = [{"iid": i["iid"], "box": i["box"], "verdicts": [_v(x) for x in (cv.get(str(i["iid"])) or [])]} for i in insts]
        ans = item["target_iid"]
    elif kind == "sibling_positive":
        conds = list(sibling_clauses().get((item["run"], item["group"]), []))
        m = sibling_matrix().get((item["run"], item["group"]))
        if not conds or not m:
            return None, "no_sibling_matrix"
        rows = [{"iid": i["iid"], "box": i["box"], "verdicts": [_v(x) for x in (m.get(i["iid"]) or [])]} for i in insts]
        ans = item["target_iid"]
    else:  # negative: one clause flipped
        conds = list(rec["clauses_neg"] or [])
        fidx = rec["flipped"]["idx"]
        fclause = rec["flipped"].get("clause_neg")
        cv = rec.get("clause_verdicts") or {}
        nv = rec.get("neg_clause_verdicts") or {}
        if not conds and fidx is not None and fclause and rec.get("clauses") and fidx < len(rec["clauses"]):
            conds = list(rec["clauses"])  # re-flipped scene: the second flip replaced one clause of the original list
            conds[fidx] = fclause
        if fidx is None or not conds or fidx >= len(conds):
            return None, "no_flip"
        rows = []
        for i in insts:
            base = [_v(x) for x in (cv.get(str(i["iid"])) or [])]
            base = (base + ["unclear"] * len(conds))[: len(conds)]
            base[fidx] = _v(nv.get(str(i["iid"])))
            rows.append({"iid": i["iid"], "box": i["box"], "verdicts": base})
        t = rec["target_iid"]
        if rec["flipped"].get("from"):
            seen[t] = {fidx: rec["flipped"]["from"]}
        flip_from, flip_to = rec["flipped"].get("from"), rec["flipped"].get("to")
        for row in rows:  # the flipped detail is false of the intended object by construction
            if row["iid"] == t:
                row["verdicts"][fidx] = "no"
        ans = None
    if not conds:
        return None, "no_clauses"
    for row in rows:
        row["verdicts"] = (row["verdicts"] + ["unclear"] * len(conds))[: len(conds)]
        row["seen"] = seen.get(row["iid"], {})
    if ans is not None:
        for row in rows:
            if row["iid"] == ans:
                row["verdicts"] = ["yes"] * len(conds)
    # cap the candidate count: keep the answer row, then the largest instances
    area = {i["iid"]: i.get("area", 0.0) for i in insts}
    if len(rows) > MAX_CAND:
        keep = sorted(rows, key=lambda r: (r["iid"] != ans, -area.get(r["iid"], 0.0)))[:MAX_CAND]
        rows = sorted(keep, key=lambda r: r["iid"])
    fits = [r for r in rows if _fit(r["verdicts"])]
    if ans is not None:  # the derived rule (most "yes" among the fitting rows) must pick the label
        top = max((r["verdicts"].count("yes") for r in fits), default=-1)
        winners = [r["iid"] for r in fits if r["verdicts"].count("yes") == top]
        if winners != [ans]:
            return None, "positive_not_unique"
    if ans is None and fits:
        return None, "negative_has_satisfier"
    # the "commit" candidate: the object the description is about (the answer, or the intended
    # object of a flipped negative); cross-scene negatives have none
    first = ans if ans is not None else (None if item.get("source") == "cross" else rec["target_iid"])
    for row in rows:
        row.setdefault("origin", (item["run"], item["group"], row["iid"]))  # where this row's rationales live
    return {"conditions": conds, "rows": rows, "answer_iid": ans, "flipped_idx": fidx, "flipped_clause": fclause, "first_iid": first,
            "flip_from": flip_from, "flip_to": flip_to}, "ok"


ORDERS = ("iid", "commit", "random", "coin")  # random: seeded shuffle; coin: target first with p=0.5, else a distractor first (seed = scene, so twins share the order)
ORDER_SALT = 0  # bumped per SFT epoch so seeded orders vary across epochs


def render(m: dict, wh: list[int], style: str = "yn", order: str = "iid") -> str:
    """order="commit" lists the described object first (commit-then-audit, single turn)."""
    assert style in STYLES, style
    assert order in ORDERS, order
    lines = ["<conditions>"] + [f"{j + 1}. {c}" for j, c in enumerate(m["conditions"])] + ["</conditions>", "<candidates>"]
    ans_box = None
    rows = m["rows"]
    if order == "commit" and m.get("first_iid") is not None:
        rows = sorted(rows, key=lambda r: (r["iid"] != m["first_iid"], r["iid"]))
    for k, row in enumerate(rows, 1):
        cells = []
        for j, v in enumerate(row["verdicts"]):
            s = f"{j + 1}: {v}"
            if style == "obs" and v == "no" and row.get("seen", {}).get(j):
                s += f" (seen: {row['seen'][j]})"
            cells.append(s)
        rel = D.relative_1000(row["box"], wh)
        lines.append(f'<object id="{k}" bbox="{json.dumps(rel)}">' + "; ".join(cells) + (" -> fits" if _fit(row["verdicts"]) else " -> excluded") + "</object>")
        if row["iid"] == m["answer_iid"]:
            ans_box = rel
    lines.append("</candidates>")
    lines.append("<answer>" + json.dumps({"bbox_2d": ans_box}) + "</answer>")
    return "\n".join(lines)


# --- parsing -----------------------------------------------------------------

_COND = re.compile(r"<conditions>(.*?)</conditions>", re.S | re.I)
_COND_LINE = re.compile(r"^\s*(\d+)\s*[.):]\s*(.+?)\s*$")
_OBJ = re.compile(r'<object\s+id="?(\d+)"?\s+bbox="?\s*\[?\s*([-\d.,\s]+?)\s*\]?\s*"?\s*>(.*?)</object>', re.S | re.I)
_CELL = re.compile(r"(\d+)\s*:\s*(yes|no|unclear)\b\s*(?:\(\s*seen\s*:\s*([^)]*)\)|(?:[—–]|-(?!>))+\s*([^;\n<]*?)(?=\s*(?:;|->|\n|</object>|$)))?", re.I)
_ANS = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _px(rel: list[float], wh) -> list[float]:
    W, H = wh
    x0, y0, x1, y1 = rel
    return [x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000]


def parse(text: str, wh) -> dict:
    """Trace -> conditions, candidates (box px, verdicts per condition), free answer, derived
    answer, format validity.  Never raises."""
    text = text or ""
    conds: list[str] = []
    m = _COND.search(text)
    if m:
        for line in m.group(1).splitlines():
            lm = _COND_LINE.match(line)
            if lm:
                conds.append(lm.group(2))
    cands = []
    for om in _OBJ.finditer(text):
        nums = [float(x) for x in _NUM.findall(om.group(2))]
        box_ok = len(nums) == 4 and all(0 <= v <= 1000 for v in nums) and nums[0] < nums[2] and nums[1] < nums[3]
        verdicts: dict[int, str] = {}
        seen: dict[int, str] = {}
        for cm in _CELL.finditer(om.group(3)):
            j = int(cm.group(1))
            if j not in verdicts:
                verdicts[j] = cm.group(2).lower()
                why = cm.group(3) or cm.group(4)
                if why and why.strip():
                    seen[j] = why.strip()
        cands.append({"id": int(om.group(1)), "box_rel": nums if box_ok else None, "box": _px(nums, wh) if box_ok else None,
                      "verdicts": verdicts, "seen": seen})
    J = len(conds)
    am = _ANS.search(text)
    free = P.parse_qwen3vl(am.group(1) if am else text, convention=P.RELATIVE_1000, original_wh=tuple(wh), sent_wh=tuple(wh))
    free_type = "box" if (free.output_type == "box" and free.box_xyxy_px) else ("null" if free.output_type in D.NULL_TYPES else "invalid")
    free_box = list(free.box_xyxy_px) if free_type == "box" else None
    format_ok = (J >= 1 and 1 <= len(cands) <= MAX_CAND and all(c["box"] is not None for c in cands)
                 and all(all(j in c["verdicts"] for j in range(1, J + 1)) for c in cands) and free_type != "invalid")
    # derived answer
    fits = [c for c in cands if c["box"] is not None and not any(c["verdicts"].get(j) == "no" for j in range(1, J + 1))]
    if not cands or J == 0:
        derived_type, derived_box, derived_cand = "invalid", None, None
    elif len(fits) == 0:
        derived_type, derived_box, derived_cand = "null", None, None
    else:
        best = max(fits, key=lambda c: sum(c["verdicts"].get(j) == "yes" for j in range(1, J + 1)))
        derived_type, derived_box, derived_cand = "box", best["box"], best
    return {"conditions": conds, "candidates": cands, "n_cand": len(cands), "format_ok": bool(format_ok),
            "free_type": free_type, "free_box": free_box, "derived_type": derived_type, "derived_box": derived_box,
            "derived_cand": derived_cand}


def consistent(pr: dict) -> bool:
    if pr["free_type"] != pr["derived_type"]:
        return False
    if pr["free_type"] == "box":
        return P.iou(tuple(pr["free_box"]), tuple(pr["derived_box"])) >= 0.5
    return True


_STOP = {"the", "a", "an", "is", "has", "of", "and", "with", "that", "its", "it"}


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower())) - _STOP


def align(conditions: list[str], clause: str | None) -> int | None:
    """1-based index of the model's condition closest to a label clause (word Jaccard >= 0.3)."""
    if not clause or not conditions:
        return None
    w = _words(clause)
    best, score = None, 0.0
    for j, c in enumerate(conditions, 1):
        cw = _words(c)
        s = len(w & cw) / max(1, len(w | cw))
        if s > score:
            best, score = j, s
    return best if score >= 0.3 else None


def evidence(pr: dict, clause: str | None, need_seen: bool = False) -> bool | None:
    """Does some candidate mark the (aligned) flipped clause 'no'?  None if unalignable.
    With need_seen the accusing cell must also name what was seen ("no (seen: ...)")."""
    j = align(pr["conditions"], clause)
    if j is None:
        return None
    return any(c["verdicts"].get(j) == "no" and (not need_seen or bool(c["seen"].get(j))) for c in pr["candidates"])


def intended_box(m: dict | None):
    """Pixel box of the object the description is about (answer, or the flipped negative's
    intended object); None for cross-scene negatives."""
    if not m or m.get("first_iid") is None:
        return None
    return next((r["box"] for r in m["rows"] if r["iid"] == m["first_iid"]), None)


def gt_candidate(pr: dict, gt_box) -> dict | None:
    best, biou = None, 0.0
    for c in pr["candidates"]:
        if c["box"] is None:
            continue
        v = P.iou(tuple(c["box"]), tuple(gt_box))
        if v > biou:
            best, biou = c, v
    return best if biou >= 0.5 else None


def false_no_cells(pr: dict, gt_box) -> tuple[int, int] | None:
    """(number of 'no' cells, number of cells) on the candidate matching the ground-truth box."""
    c = gt_candidate(pr, gt_box)
    if c is None:
        return None
    J = len(pr["conditions"])
    return sum(c["verdicts"].get(j) == "no" for j in range(1, J + 1)), J


def scale_augment(image, m: dict, wh, target_frac: float, rng):
    """Shrink the scene so the described object covers `target_frac` of the frame (GME-like
    scale) and paste it at a random offset onto a gray canvas of the original size; returns
    (new image, matrix with transformed boxes).  No-op when the object is already smaller."""
    from PIL import Image as _Image

    W, H = wh
    box = intended_box(m)
    if box is None:
        return image, m, (1.0, 0, 0)
    x0, y0, x1, y1 = box
    f0 = max(1e-6, (x1 - x0) * (y1 - y0) / float(W * H))
    if f0 <= target_frac:
        return image, m, (1.0, 0, 0)
    s = (target_frac / f0) ** 0.5
    nw, nh = max(8, round(W * s)), max(8, round(H * s))
    small = image.resize((nw, nh), _Image.LANCZOS)
    ox, oy = rng.randint(0, W - nw), rng.randint(0, H - nh)
    canvas = _Image.new("RGB", (W, H), (128, 128, 128))
    canvas.paste(small, (ox, oy))
    rows = [{**r, "box": [r["box"][0] * s + ox, r["box"][1] * s + oy, r["box"][2] * s + ox, r["box"][3] * s + oy]} for r in m["rows"]]
    return canvas, {**m, "rows": rows}, (s, ox, oy)


def transform_box(box, t):
    s, ox, oy = t
    return [box[0] * s + ox, box[1] * s + oy, box[2] * s + ox, box[3] * s + oy]


# --- self-hint crop: a close-up of the most likely candidate as a second image ---------------


def crop_view(image, box):
    """Close-up of a box with 25% context, short side >= 512 px (the checker's view)."""
    return C.closeup(image, box)


def hint_text(box, wh) -> str:
    return json.dumps(D.relative_1000(box, wh))


def train_hint_box(item: dict, lookup: dict | None):
    """The hint box of a training item: the base model's own box for that expression when
    datagen recorded one (the inference distribution), else the described object, else the
    largest instance."""
    m = item.get("matrix")
    src = item.get("source")
    if lookup and src not in ("short", "alt", "cross") and item["kind"] in ("positive", "sibling_positive", "negative"):
        try:
            v = D.base4b_view(item, lookup)
        except KeyError:
            v = None
        if v and v.get("box"):
            return list(v["box"])
    b = intended_box(m)
    if b is not None:
        return b
    if m and m["rows"]:
        return max(m["rows"], key=lambda r: (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]))["box"]
    if (item.get("answer") or {}).get("bbox_2d"):  # answer-only items (RefCOCO train): the ground-truth box
        return list(item["answer"]["bbox_2d"])
    return None


def train_hint_boxes(item: dict, lookup: dict | None, k: int, rng=None, noise: float = 0.0) -> list:
    """Up to k hint boxes for a training item: the single hint first (base box / described
    object), then the largest other instances.  With `noise` the first box is replaced by
    another instance's box with that probability (the trace must not trust the crops blindly)."""
    m = item.get("matrix")
    first = train_hint_box(item, lookup)
    if first is None:
        return []
    rows = list(m["rows"]) if m else []
    if rng is not None and noise > 0 and len(rows) > 1 and rng.random() < noise:
        others = [r["box"] for r in rows if r["iid"] != m.get("first_iid")]
        if others:
            first = rng.choice(others)
    out = [first]
    for r in sorted(rows, key=lambda r: -(r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1])):
        if len(out) >= k:
            break
        if all(P.iou(tuple(r["box"]), tuple(b)) < 0.5 for b in out):
            out.append(r["box"])
    return out


def hints_text(boxes, wh) -> str:
    return ", ".join(hint_text(b, wh) for b in boxes)


def build_all(items: list[dict], style: str = "yn", order: str = "iid") -> tuple[list[dict], dict[str, int]]:
    """Attach `trace` to every item whose labels support one; returns (kept items, drop reasons)."""
    kept, reasons = [], {}
    for it in items:
        m, why = label_matrix(it)
        if m is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        kept.append({**it, "trace": render(m, it["image_wh"], style, order), "matrix": m})
    return kept, reasons


if __name__ == "__main__":
    import argparse
    import collections

    ap = argparse.ArgumentParser(description="Render and self-check the label traces.")
    ap.add_argument("--style", default="yn", choices=STYLES)
    ap.add_argument("--order", default="iid", choices=ORDERS)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--extra", default=str(D.TRAIN_ROOT / "augment" / "cross_v1.jsonl"))
    args = ap.parse_args()
    items = D.load_items() + (D.load_items(Path(args.extra)) if args.extra and Path(args.extra).is_file() else [])
    kept, reasons = build_all(items, args.style, args.order)
    print(f"{len(kept)} traces of {len(items)} items; dropped {reasons}")
    print("by kind:", dict(collections.Counter(it["kind"] + ("/cross" if it.get("source") == "cross" else "") for it in kept)))
    bad = 0
    for it in kept:  # round trip: parse(render) must derive the label answer
        pr = parse(it["trace"], it["image_wh"])
        gt = it["answer"]["bbox_2d"]
        ok = pr["format_ok"] and consistent(pr) and ((gt is None and pr["derived_type"] == "null") or
                                                    (gt is not None and pr["derived_type"] == "box" and P.iou(tuple(pr["derived_box"]), tuple(gt)) >= 0.5))
        bad += not ok
    print(f"round-trip failures: {bad}")
    lens = [len(it["trace"]) for it in kept]
    print(f"trace chars: median {sorted(lens)[len(lens) // 2]}, max {max(lens)}")
    shown = collections.Counter()
    for it in kept:
        key = it["kind"] + ("/cross" if it.get("source") == "cross" else "")
        if shown[key] < args.show:
            shown[key] += 1
            print(f"\n=== {it['id']} ({key})\n{it['expression']}\n{it['trace']}")


# --- rationales ("rat" style) and the two-turn tool protocol -------------------------------------

RATIONALES = Path(os.environ["VLMG_RATIONALES"]).expanduser() if os.environ.get("VLMG_RATIONALES") else D.TRAIN_ROOT / "rationales.jsonl"  # VLMG_RATIONALES selects another rationale file (e.g. the Gemini-written one)
TOOL_CALL = '<tool_call>{"name": "image_zoom_in", "arguments": {"boxes": %s}}</tool_call>'
_TOOL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S | re.I)
_BOX4 = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def rationales() -> dict[tuple, dict[str, dict]]:
    """(run, image_id, iid) -> {condition text: {label, blind, rationale}} from train.rationales."""
    if "rat" not in _CACHE:
        out: dict[tuple, dict[str, dict]] = {}
        if RATIONALES.is_file():
            for l in open(RATIONALES, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    d = out.setdefault((r["run"], r["image_id"], r["iid"]), {})
                    for c, lab, bl, ra in zip(r["conditions"], r["label"], r["blind"], r["rationale"]):
                        d[c] = {"label": lab, "blind": bl, "rationale": ra}
        _CACHE["rat"] = out
    return _CACHE["rat"]


def _clean_reason(text: str | None) -> str | None:
    if not text:
        return None
    t = re.sub(r"[;\n<>]", ",", text).replace("->", "to").strip(" .,")
    words = t.split()
    return " ".join(words[:12]) if words else None


def cell_text(m: dict, row: dict, j: int, style: str) -> str:
    """One verdict cell.  'rat': verdict — reason, where the reason is the 8B rationale kept only
    when its blind verdict agrees with the label; the flipped cell of the intended object gets the
    label-exact reason '<from>, not <to>'."""
    v = row["verdicts"][j]
    s = f"{j + 1}: {v}"
    if style == "obs":
        if v == "no" and row.get("seen", {}).get(j):
            s += f" (seen: {row['seen'][j]})"
        return s
    if style != "rat":
        return s
    reason = None
    if m.get("flipped_idx") == j and row["iid"] == m.get("first_iid") and m.get("flip_from") and v == "no":
        reason = f"{m['flip_from']}, not {m['flip_to']}" if m.get("flip_to") else m["flip_from"]
    else:
        info = rationales().get(tuple(row.get("origin", ())), {}).get(m["conditions"][j])
        if info and info["blind"] == v and info["label"] == v:
            reason = info["rationale"]
    reason = _clean_reason(reason)
    return s + (f" — {reason}" if reason else "")


_ORDER_RNG = random.Random(20260926)


def ordered_rows(m: dict, order: str) -> list[dict]:
    """Rows in trace order (commit: the described object first), capped at MAX_CAND keeping the
    described object and then the largest instances (mosaic matrices can exceed the cap)."""
    rows = list(m["rows"])
    keep_iid = m.get("first_iid") if m.get("first_iid") is not None else m.get("answer_iid")
    if len(rows) > MAX_CAND:
        area = lambda r: (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1])
        rows = sorted(rows, key=lambda r: (r["iid"] != keep_iid, -area(r)))[:MAX_CAND]
        rows = sorted(rows, key=lambda r: r["iid"])
    if order == "commit" and m.get("first_iid") is not None:
        rows = sorted(rows, key=lambda r: (r["iid"] != m["first_iid"], r["iid"]))
    elif order in ("random", "coin"):
        # one permutation per (scene, salt): the tool-call boxes and the table rows agree, and a scene's positive and
        # negative twins render the same order (they differ only in the flipped cell)
        origin = (m["rows"][0].get("origin") or ())[:2] if m.get("rows") else ()
        rng = random.Random(f"{origin}|{sorted(r['iid'] for r in rows)}|{ORDER_SALT}")  # twins: same scene, same rows -> same order
        rows = sorted(rows, key=lambda r: r["iid"])
        first = m.get("first_iid")
        tgt = next((r for r in rows if r["iid"] == first), None) if first is not None else None
        if order == "random" or tgt is None:
            rng.shuffle(rows)
        else:
            others = [r for r in rows if r["iid"] != first]
            rng.shuffle(others)
            if not others or rng.random() < 0.5:
                rows = [tgt] + others  # target first (the model's top-1 is a real prior)
            else:
                k = rng.randrange(1, len(others) + 1)  # a distractor first, the target anywhere after it
                rows = others[:k] + [tgt] + others[k:]
    return rows


CONTRAST = re.compile(r"\bnot\b|\binstead\b|\brather\b|\bno\b\s+\w+|\bwithout\b|\bmissing\b|\babsent\b", re.I)


def named_no(c: dict, j: int) -> bool:
    """A 'no' cell whose reason names an observed value or contrast (bare 'no' is treated as unclear)."""
    return c["verdicts"].get(j) == "no" and bool(CONTRAST.search(c["seen"].get(j) or ""))


def rederive(pr: dict, rule: str = "named") -> tuple[str, list | None, dict | None]:
    """Derived answer under a different exclusion rule: 'named' excludes a candidate only through named 'no' cells."""
    J = len(pr["conditions"])
    cands = [c for c in pr["candidates"] if c.get("box")]
    if not cands or J == 0:
        return "invalid", None, None
    is_no = named_no if rule == "named" else (lambda c, j: c["verdicts"].get(j) == "no")
    fits = [c for c in cands if not any(is_no(c, j) for j in range(1, J + 1))]
    if not fits:
        return "null", None, None
    best = max(fits, key=lambda c: sum(c["verdicts"].get(j) == "yes" for j in range(1, J + 1)))
    return "box", best["box"], best


def render_table(m: dict, wh, style: str, order: str) -> tuple[list[str], list]:
    """The <candidates> lines and the answer box (relative coords), rows in `order`."""
    lines = ["<candidates>"]
    ans_box = None
    for k, row in enumerate(ordered_rows(m, order), 1):
        cells = [cell_text(m, row, j, style) for j in range(len(row["verdicts"]))]
        rel = D.relative_1000(row["box"], wh)
        lines.append(f'<object id="{k}" bbox="{json.dumps(rel)}">' + "; ".join(cells) + (" -> fits" if _fit(row["verdicts"]) else " -> excluded") + "</object>")
        if row["iid"] == m["answer_iid"]:
            ans_box = rel
    lines.append("</candidates>")
    return lines, ans_box


def render_turns(m: dict, wh, style: str = "rat", order: str = "commit") -> tuple[str, list, str]:
    """Two-turn label trace: (turn-1 text = conditions + zoom tool call, the boxes in pixels in
    that order, turn-2 text = candidate table + answer)."""
    rows = ordered_rows(m, order)
    boxes_px = [r["box"] for r in rows]
    boxes_rel = [D.relative_1000(b, wh) for b in boxes_px]
    t1 = "<conditions>\n" + "\n".join(f"{j + 1}. {c}" for j, c in enumerate(m["conditions"])) + "\n</conditions>\n" + TOOL_CALL % json.dumps(boxes_rel)
    table, ans_box = render_table(m, wh, style, order)
    t2 = "\n".join(table) + "\n<answer>" + json.dumps({"bbox_2d": ans_box}) + "</answer>"
    return t1, boxes_px, t2


def parse_tool_boxes(text: str, wh, max_boxes: int = MAX_CAND) -> list:
    """Boxes (pixels) named in the first <tool_call> of a turn-1 text; malformed -> []."""
    m = _TOOL.search(text or "")
    if not m:
        return []
    out = []
    for g in _BOX4.findall(m.group(1)):
        b = [float(x) for x in g]
        if 0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000:
            out.append(_px(b, wh))
        if len(out) >= max_boxes:
            break
    return out


OVERVIEW = os.environ.get("VLMG_OVERVIEW", "") not in ("", "0")  # turn 2 also gets the full image with the candidate boxes drawn and numbered


def turn2_user_text(n: int) -> str:
    if OVERVIEW:
        return f"An overview of the image with your {n} candidate boxes drawn and numbered, then close-ups of the {n} boxes in the same order. Continue with step 2."
    return f"Close-ups of your {n} candidate boxes, in the same order. Continue with step 2."


def turn2_text_for(n_views: int) -> str:
    """User text of turn 2 given the number of images shown (overview + crops, or crops)."""
    return turn2_user_text(n_views - 1 if (OVERVIEW and n_views > 1) else n_views)


def mark_overview(image, boxes_px: list) -> "Image.Image":
    """A copy of the image with the candidate boxes outlined and numbered (set-of-mark style) so
    relation clauses can be judged with the candidate located in the whole scene."""
    from PIL import ImageDraw

    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    lw = max(2, round(min(w, h) / 250))
    for k, b in enumerate(boxes_px, 1):
        x0, y0, x1, y1 = [float(v) for v in b]
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=lw)
        label = str(k)
        fs = max(12, round(min(w, h) / 30))
        try:
            from PIL import ImageFont

            font = ImageFont.truetype("DejaVuSans-Bold.ttf", fs)
        except Exception:  # noqa: BLE001
            font = None
        tw, th = (fs * 0.65 * len(label), fs) if font is None else (d.textlength(label, font=font), fs)
        tx, ty = min(max(0, x0), w - tw - 4), max(0, y0 - th - 4) if y0 - th - 4 >= 0 else min(y0 + 2, h - th - 4)
        d.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill=(255, 0, 0))
        d.text((tx + 2, ty + 2), label, fill=(255, 255, 255), font=font)
    return im


def turn2_views(image, boxes_px: list) -> list:
    """Images of the turn-2 user message: crops of the candidate boxes, preceded by the marked
    overview under VLMG_OVERVIEW; the full image alone when no valid box was named."""
    if not boxes_px:
        return [image]
    crops = [crop_view(image, b) for b in boxes_px]
    return ([mark_overview(image, boxes_px)] if OVERVIEW else []) + crops
