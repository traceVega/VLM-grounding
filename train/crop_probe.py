"""Crop-audit probe on the GME dev subset (zero-shot, no training): take a trace model's own
decomposition and its committed (first) candidate from an existing evaluation, and re-verify
the conditions on that candidate with the base 4B in isolation, either from the global view
(candidate outlined) or from the global view plus a native-resolution close-up.

Two decision rules are scored: the hard rule (null iff some condition is judged "no") and the
soft rule (null iff the largest per-condition probability of "no" exceeds a threshold), the
latter as an operating curve and an AUROC between rejection items and positives.  This
measures, before any training, how much of the GME rejection ceiling is the verification view
and how much is the calibration of the decision.

    python -m train.crop_probe --tag scr_grpo_v3b_pair [--views global,crop] [--model 4b]

Output: $VLMG_DATA_ROOT/train/eval/crop_probe_<tag>/{rows.jsonl, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import re
import time

import torch
from PIL import Image

from datagen import common as C
from datagen.checker_api import _shrink, parse_batched
from scripts.pilot_abstain_signal import _first_ids
from shared.harness import parsers as P
from shared.harness import prompts
from shared.harness import tokens as T
from train import data as D
from train import traces as TR
from train.eval_suite import gme_subset

_NUMCOLON = re.compile(r"(\d+)\s*:\s*$")


def verdict_probs(seq: list[int], scores, tok, groups: dict[str, set[int]]) -> dict[int, dict[str, float]]:
    """At every generated position that follows 'k:' return the probability mass of yes / no /
    unclear (normalised over the three), keyed by k."""
    out = {}
    text = ""
    for i, tid in enumerate(seq):
        m = _NUMCOLON.search(text)
        if m and int(m.group(1)) not in out:
            probs = torch.softmax(scores[i][0].float(), dim=-1)
            p = {g: float(probs[list(ids)].sum()) for g, ids in groups.items()}
            z = sum(p.values()) or 1.0
            out[int(m.group(1))] = {g: v / z for g, v in p.items()}
        text += tok.decode([tid])
    return out


def auroc(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    """Probability that a rejection item scores above a positive item (ties half)."""
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    for a in pos_scores:
        for b in neg_scores:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return round(wins / (len(pos_scores) * len(neg_scores)), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="evaluation tag whose gme.jsonl holds the traces")
    ap.add_argument("--model", default="4b", choices=list(D.MODELS))
    ap.add_argument("--views", default="global,crop")
    ap.add_argument("--gme-pos-n", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="output directory name (default crop_probe_<tag>)")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText

    src = D.TRAIN_ROOT / "eval" / args.tag / "gme.jsonl"
    rows_in = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    items = {it["id"]: it for it in gme_subset(args.gme_pos_n, 0)}
    out_dir = D.TRAIN_ROOT / "eval" / (args.out or f"crop_probe_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tm = {"global": prompts.load("verifier_clauses_batched_global", non_kill=True), "crop": prompts.load("verifier_clauses_batched", non_kill=True)}
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    hf, rev = D.MODELS[args.model]
    processor = T.load_capped_processor(hf, rev)
    tok = processor.tokenizer
    groups = {"no": _first_ids(tok, [" no", "no", " No", "No"]), "yes": _first_ids(tok, [" yes", "yes", " Yes", "Yes"]),
              "unclear": _first_ids(tok, [" unclear", "unclear", " Unclear", "Unclear"])}
    model = AutoModelForImageTextToText.from_pretrained(hf, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    out_rows = []
    skipped = 0
    t0 = time.time()
    todo = rows_in[: args.limit] if args.limit else rows_in
    for n, r in enumerate(todo, 1):
        it = items.get(r["id"])
        if it is None:
            skipped += 1
            continue
        full = Image.open(it["image"]).convert("RGB")
        pr = TR.parse(r["raw"], full.size)
        if not pr["conditions"] or not pr["candidates"] or pr["candidates"][0]["box"] is None:
            skipped += 1
            continue
        conds = pr["conditions"]
        commit = pr["candidates"][0]
        box = commit["box"]
        small, s = _shrink(full)
        outlined = C.outline(small, [v * s for v in box], "red")
        text_items = "\n".join(f"{k + 1}. The object {c}." for k, c in enumerate(conds))
        rec = {"id": r["id"], "n_gt": it["n_gt"], "dimension": it.get("dimension"), "n_cond": len(conds), "commit": box,
               "commit_iou": max((P.iou(tuple(box), tuple(g)) for g in it["gt_boxes"]), default=0.0),
               "trace_verdicts": [commit["verdicts"].get(j, "missing") for j in range(1, len(conds) + 1)],
               "trace_derived": r.get("output_type")}
        for view in views:
            imgs = [outlined] if view == "global" else [outlined, C.closeup(full, box)]
            text = tm[view].render(category="object", items=text_items)
            msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": text}]}]
            chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=imgs, text=chat, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False, output_scores=True, return_dict_in_generate=True)
            seq = gen.sequences[0][inputs["input_ids"].shape[1]:].tolist()
            raw = processor.decode(seq, skip_special_tokens=True)
            rec[f"{view}_verdicts"] = parse_batched(raw, len(conds))
            vp = verdict_probs(seq, gen.scores, tok, groups)
            rec[f"{view}_pno"] = [round(vp.get(k, {}).get("no", 0.0), 4) for k in range(1, len(conds) + 1)]
        out_rows.append(rec)
        if n % 40 == 0:
            print(f"  [{n}/{len(todo)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
        for rec in out_rows:
            fh.write(json.dumps(rec) + "\n")

    rej = [x for x in out_rows if x["n_gt"] == 0]
    pos = [x for x in out_rows if x["n_gt"] > 0]

    def hard(key):
        null = lambda x: "no" in x[key]
        rej_acc = sum(null(x) for x in rej) / max(1, len(rej))
        pos_null = sum(null(x) for x in pos) / max(1, len(pos))
        pos_acc = sum((not null(x)) and x["commit_iou"] >= 0.5 for x in pos) / max(1, len(pos))
        no_cells = sum(v == "no" for x in pos if x["commit_iou"] >= 0.5 for v in x[key])
        cells = sum(len(x[key]) for x in pos if x["commit_iou"] >= 0.5)
        return {"rejection_acc": round(rej_acc, 4), "positive_null_rate": round(pos_null, 4), "positive_acc": round(pos_acc, 4),
                "net_rejection": round(rej_acc - pos_null, 4), "false_no_cell_rate_on_correct_commit": round(no_cells / max(1, cells), 4)}

    def soft(key):
        s = lambda x: max(x[key]) if x[key] else 0.0
        curve = []
        for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            rej_acc = sum(s(x) > tau for x in rej) / max(1, len(rej))
            pos_null = sum(s(x) > tau for x in pos) / max(1, len(pos))
            pos_acc = sum(s(x) <= tau and x["commit_iou"] >= 0.5 for x in pos) / max(1, len(pos))
            curve.append({"tau": tau, "rejection_acc": round(rej_acc, 4), "positive_null_rate": round(pos_null, 4),
                          "positive_acc": round(pos_acc, 4), "net_rejection": round(rej_acc - pos_null, 4)})
        return {"auroc_rej_vs_pos": auroc([s(x) for x in rej], [s(x) for x in pos]),
                "auroc_rej_vs_pos_correct_commit": auroc([s(x) for x in rej], [s(x) for x in pos if x["commit_iou"] >= 0.5]),
                "curve": curve}

    summary = {"n": len(out_rows), "skipped": skipped,
               "commit_recall_pos": round(sum(x["commit_iou"] >= 0.5 for x in pos) / max(1, len(pos)), 4),
               "in_trace_hard": hard("trace_verdicts")}
    for view in views:
        summary[f"{view}_hard"] = hard(f"{view}_verdicts")
        summary[f"{view}_soft"] = soft(f"{view}_pno")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
