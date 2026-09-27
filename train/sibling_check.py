"""Sibling-description label matrix, checked locally with Qwen3-VL-8B (no API): for every
scene that exports a sibling positive, each candidate instance is shown outlined in the
full image plus as a close-up, and asked the sibling's clauses in one batched prompt
(configs/prompts/verifier_clauses_batched.txt, the same protocol as the Gemini checker).

    python -m train.sibling_check [--limit N]

Output: $VLMG_DATA_ROOT/train/sibling_matrix.jsonl, one row per scene
  {"run", "image_id", "verdicts": {iid: [yes|no|unclear|unparsed, ...]}, "checker"}
Resumable by (run, image_id).  train.traces reads it for the sibling traces.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="8b", choices=list(D.MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText

    out = TR.SIB_MATRIX
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        for l in open(out, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add((r["run"], r["image_id"]))
    scenes = sorted({(it["run"], it["group"]) for it in D.load_items() if it["kind"] == "sibling_positive"})
    todo = [s for s in scenes if s not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"sibling matrix: {len(scenes)} scenes, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    recs = TR.records()
    sib = TR.sibling_clauses()
    tmpl = prompts.load("verifier_clauses_batched", non_kill=True)
    hf, rev = D.MODELS[args.model]
    processor = T.load_capped_processor(hf, rev)
    model = AutoModelForImageTextToText.from_pretrained(hf, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    t0 = time.time()
    n_inst = 0
    for n, (run, image_id) in enumerate(todo, 1):
        rec = recs[(run, image_id)]
        clauses = sib.get((run, image_id)) or []
        if not clauses:
            continue
        full = Image.open(C.image_path(image_id)).convert("RGB")
        small, s = _shrink(full)
        items = "\n".join(f"{k + 1}. The {rec['category']} {c}." for k, c in enumerate(clauses))
        text = tmpl.render(category=rec["category"], items=items)
        verdicts = {}
        for inst in rec["instances"]:
            views = [C.outline(small, [v * s for v in inst["box"]], "red"), C.closeup(full, inst["box"])]
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]
            chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=views, text=chat, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
            raw = processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            verdicts[str(inst["iid"])] = parse_batched(raw, len(clauses))
            n_inst += 1
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run": run, "image_id": image_id, "verdicts": verdicts, "checker": f"{hf}@{rev[:8]}"}) + "\n")
        if n % 10 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {n_inst} instances, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
