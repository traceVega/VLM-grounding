"""Stage `policy`: the base grounder under the byte-identical GroundingME prompt.

Records the box, the decision-token probabilities (p(null), p(box)) and which
candidate instance the box landed on.  This is the difficulty gate (gate 3) and
the source of the sibling S for the pairwise data: S is the instance the base
boxed on the target expression when it was wrong.

Model: Qwen3-VL-8B-Instruct at the pinned revision (the 4B training target is not
downloaded yet; the gate must be re-run on the 4B before training).
"""

from __future__ import annotations

import time

import torch
from PIL import Image

from datagen import common as C
from scripts.pilot_abstain_signal import _decision, _first_ids
from shared.harness import parsers as P
from shared.harness import prompts
from shared.harness import tokens as T

HF = "Qwen/Qwen3-VL-8B-Instruct"
REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODELS = {
    "8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "policy.jsonl"),
    # the PoC training target (downloaded 2026-09-23); its own baseline file
    "4b": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17", "policy_4b.jsonl"),
}

EXPR_KEY = {"target": ("write.jsonl", "expr_target"),
            "neg": ("write.jsonl", "expr_neg"),
            "neg2": ("reflip.jsonl", "expr_neg"),  # accepted second-chance negatives
            "sibling": ("sibling.jsonl", "expr_sibling")}


def run(run: str, which: list[str], limit: int | None, model_key: str = "8b") -> None:
    from transformers import AutoModelForImageTextToText

    HF, REV, fname = MODELS[model_key]
    root = C.run_root(run)
    scenes = C.by_image(C.read_jsonl(root / "scenes.jsonl"))
    out = root / fname
    done = C.done_keys(out, ("image_id", "which"))
    jobs = []
    for w in which:
        fname, key = EXPR_KEY[w]
        for r in C.read_jsonl(root / fname):
            if r.get(key) and (r["image_id"], w) not in done:
                jobs.append((r["image_id"], w, r[key]))
    if limit:
        jobs = jobs[:limit]
    print(f"policy: {len(done)} done, {len(jobs)} to run ({','.join(which)})", flush=True)
    if not jobs:
        return

    template = prompts.load("grounding_qwen3vl_primary")
    processor = T.load_capped_processor(HF, REV)
    T.assert_cap_is_in_force(processor)
    tok = processor.tokenizer
    null_ids, box_ids = _first_ids(tok, [" null", "null", " Null"]), _first_ids(tok, [" [", "["])
    model = AutoModelForImageTextToText.from_pretrained(
        HF, revision=REV, dtype=torch.bfloat16, device_map="cuda").eval()
    t0 = time.time()
    for n, (image_id, w, expr) in enumerate(jobs, 1):
        sc = scenes[image_id]
        image = Image.open(C.image_path(image_id)).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": template.render(expr=expr)}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        seq = gen.sequences[0][inputs["input_ids"].shape[1]:].tolist()
        raw = processor.decode(seq, skip_special_tokens=True)
        parsed = P.parse_qwen3vl(raw, convention=P.RELATIVE_1000,
                                 original_wh=image.size, sent_wh=image.size)
        box = list(parsed.box_xyxy_px) if parsed.box_xyxy_px else None
        boxed_iid, boxed_iou = C.assign_box(box, sc["instances"])
        target = next(i for i in sc["instances"] if i["is_target"])
        rec = {
            "image_id": image_id, "which": w, "expr": expr, "raw": raw,
            "output_type": parsed.output_type, "box": box,
            "boxed_iid": boxed_iid, "boxed_iou": round(boxed_iou, 3),
            "iou_target": round(C.box_iou(box, target["box"]), 3) if box else None,
            **_decision(seq, gen.scores, tok, null_ids, box_ids),
            "policy": f"{HF}@{REV[:8]}",
        }
        C.append_jsonl(out, rec)
        if n % 10 == 0 or n == len(jobs):
            print(f"  [{n}/{len(jobs)}] {(time.time() - t0) / n:.1f}s/item", flush=True)
