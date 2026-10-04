"""Decision rules recomputed offline from stored COA audits (answer-first / veto probe).

    python -m train.veto_rules <tag> [<tag> ...]

Rules over the candidate list (candidate 1 = the injected answer under --first-box-from, else the model's first proposal):
  first_fit   the first candidate without a named mismatch (the current harness rule)
  veto_most   candidate 1 if it has no mismatch, else the passing alternative with the most "match" lines (ties: order)
  c1_only     candidate 1 if it has no mismatch, else no object
Reports GME rejection / positives / positives answered null, RefCOCO / + / g and the PR-Bench dev split per rule.
"""

from __future__ import annotations

import io
import json
import sys
from collections import defaultdict

from PIL import Image

from shared.harness import parsers as P
from train import coa as COA
from train import data as D
from train import traces as TR


def choose(audits, boxes, rule):
    fits = [k < len(audits) and audits[k]["format_ok"] and not COA.named_mismatches(audits[k]["lines"]) for k in range(len(boxes))]
    if not boxes:
        return None
    if rule == "first_fit":
        return next((boxes[k] for k, f in enumerate(fits) if f), None)
    if fits[0]:
        return boxes[0]
    if rule == "c1_only":
        return None
    alts = [k for k in range(1, len(boxes)) if fits[k]]
    if not alts:
        return None
    n_match = lambda k: sum(ln["verdict"] == "match" for ln in audits[k]["lines"])
    return boxes[max(alts, key=lambda k: (n_match(k), -k))]


def score(tag: str) -> None:
    E = D.TRAIN_ROOT / "eval" / tag
    gt = {}
    for it in D.gme_items():
        gt[it["id"]] = it
    for it in D.refcoco_items(300, 0):
        gt[it["id"]] = it
    from train import ood_items as OOD
    for it in OOD.items("prbench_dev"):
        gt[it["id"]] = it
    res = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for name in ("gme", "refcoco", "prbench_dev"):
        f = E / f"{name}.jsonl"
        if not f.is_file():
            continue
        for l in open(f):
            r = json.loads(l)
            it = gt.get(r["id"])
            if it is None or r.get("coa") is None:
                continue
            if "image_bytes" in it:
                wh = Image.open(io.BytesIO(it["image_bytes"])).size
            else:
                wh = D.open_image(it).size if "image" in it else None
            raw = r["raw"]
            i = raw.find("<coa>")
            pr = TR.parse(raw[: len(raw) if i < 0 else i], wh)
            boxes = [c["box"] for c in pr["candidates"] if c.get("box")]
            audits = [COA.parse_audit(t) for t in r["coa"]["audits"]]
            if name == "gme":
                group = "gme_rej" if r.get("dimension") == "Rejection" else "gme_pos"
            elif name == "refcoco":
                group = it["set"]
            else:
                group = "pr_reject" if not it["gt_boxes"] else "pr_pos"
            for rule in ("first_fit", "veto_most", "c1_only"):
                ans = choose(audits, boxes, rule)
                if not it.get("gt_boxes"):
                    ok = ans is None
                else:
                    ok = ans is not None and max(P.iou(tuple(ans), tuple(g)) for g in it["gt_boxes"]) >= 0.5
                res[rule][group][0] += 1
                res[rule][group][1] += ok
                if it.get("gt_boxes"):
                    res[rule][group + "_null"][0] += 1
                    res[rule][group + "_null"][1] += ans is None
    order = ["gme_rej", "gme_pos", "gme_pos_null", "RefCOCO", "RefCOCOplus", "RefCOCOg", "pr_reject", "pr_pos", "pr_pos_null"]
    print(f"[{tag}] " + " | ".join(order))
    for rule, d in res.items():
        print(f"  {rule:10s} " + " | ".join(f"{100 * d[g][1] / d[g][0]:5.1f}" if d[g][0] else "  -  " for g in order))


if __name__ == "__main__":
    for t in sys.argv[1:]:
        score(t)
