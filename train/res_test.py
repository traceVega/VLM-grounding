"""H5, the resolution test (notes/DESIGN-FINAL-LIGHT-2026-09-24.md): can the 4B judge the
clauses of a description on the *global* view (full image at the pixel cap, target outlined
in red) as well as on a native-resolution close-up?  Zero-shot, batched clause prompt, the
held-out scenes' target instance, positives (all clauses true) and negatives (one flipped).

    python -m train.res_test [--model 4b] [--scales 1.0,0.5]

Per configuration (view in global/crop x scale): false-"no" rate on true clauses, detection
rate ("no") on flipped clauses, unclear rate.  Output $VLMG_DATA_ROOT/train/eval/res_test/
{rows.jsonl, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from PIL import Image

from datagen import common as C
from datagen.checker_api import _shrink, parse_batched
from shared.harness import prompts
from shared.harness import tokens as T
from train import data as D
from train import traces as TR


def scaled(image: Image.Image, box, s: float):
    if s >= 1.0:
        return image, box
    w, h = image.size
    im = image.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return im, [v * s for v in box]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="4b", choices=list(D.MODELS))
    ap.add_argument("--scales", default="1.0,0.5")
    ap.add_argument("--n-val-scenes", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText

    out_dir = D.TRAIN_ROOT / "eval" / "res_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    _, val = D.split(D.load_items(), args.n_val_scenes, args.seed)
    items, drops = TR.build_all([it for it in val if it["kind"] in ("positive", "negative")])
    print(f"{len(items)} held-out items with label matrices (drops {drops})", flush=True)
    recs = TR.records()
    tm = {"global": prompts.load("verifier_clauses_batched_global", non_kill=True), "crop": prompts.load("verifier_clauses_batched", non_kill=True)}
    hf, rev = D.MODELS[args.model]
    processor = T.load_capped_processor(hf, rev)
    model = AutoModelForImageTextToText.from_pretrained(hf, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    scales = [float(s) for s in args.scales.split(",")]
    rows = []
    t0 = time.time()
    for n, it in enumerate(items, 1):
        rec = recs[(it["run"], it["group"])]
        m = it["matrix"]
        tiid = rec["target_iid"]
        inst = next(i for i in rec["instances"] if i["iid"] == tiid)
        full = Image.open(it["image"]).convert("RGB")
        clauses = m["conditions"]
        text_items = "\n".join(f"{k + 1}. The {rec['category']} {c}." for k, c in enumerate(clauses))
        for s in scales:
            im, box = scaled(full, inst["box"], s)
            small, sh = _shrink(im)
            outlined = C.outline(small, [v * sh for v in box], "red")
            for view in ("global", "crop"):
                views = [outlined] if view == "global" else [outlined, C.closeup(im, box)]
                text = tm[view].render(category=rec["category"], items=text_items)
                msgs = [{"role": "user", "content": [{"type": "image"} for _ in views] + [{"type": "text", "text": text}]}]
                chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(images=views, text=chat, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
                raw = processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                verdicts = parse_batched(raw, len(clauses))
                truth = ["yes"] * len(clauses)
                if it["kind"] == "negative":
                    truth[m["flipped_idx"]] = "no"
                rows.append({"id": it["id"], "kind": it["kind"], "view": view, "scale": s, "verdicts": verdicts, "truth": truth,
                             "box_frac": inst.get("area"), "raw": raw[:200]})
        if n % 8 == 0:
            print(f"  [{n}/{len(items)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = {}
    for view in ("global", "crop"):
        for s in scales:
            sub = [r for r in rows if r["view"] == view and r["scale"] == s]
            tp = fp = unc = n_true = n_false = det = 0
            for r in sub:
                for v, t in zip(r["verdicts"], r["truth"]):
                    if t == "yes":
                        n_true += 1
                        fp += v == "no"
                        unc += v not in ("yes", "no")
                    else:
                        n_false += 1
                        det += v == "no"
            summary[f"{view}@{s}"] = {"items": len(sub), "true_clauses": n_true, "false_no_rate": round(fp / max(1, n_true), 4),
                                      "unclear_rate_on_true": round(unc / max(1, n_true), 4), "flipped_clauses": n_false,
                                      "flip_detect_rate": round(det / max(1, n_false), 4)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
