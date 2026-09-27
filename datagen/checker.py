"""Stage `check`: Gemma4-12B (judge_b) answers, for every candidate instance
outlined in red on the full image, whether each detail holds.

Two questions per scene family:
  * the flipped detail of the negative on every instance -> zero-satisfier gate
    (every answer must be "no"; "unclear" and unparsed count as not-no)
  * every detail of the target expression on every instance -> the verdict
    matrix (writer sanity on the target row; sibling distinctness on the others)

Same loading path as scripts/pilot_verify_removals.py; answers parsed on the
first word.
"""

from __future__ import annotations

import time

import torch
from PIL import Image

from datagen import common as C
from shared.harness import prompts

CHECKERS = {
    "gemma4": ("google/gemma-4-12b-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", "checker.jsonl"),
    # second opinion from a different model (same family as the writer, so it bounds
    # checker disagreement rather than writer error)
    "qwen3vl": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "checker2.jsonl"),
}
HF, REV, _ = CHECKERS["gemma4"]
CLOSEUP = True  # v2 prompt: outlined scene plus a close-up crop of the outlined instance


def run(run: str, limit: int | None, checker: str = "gemma4", target_only: bool = False) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    HF, REV, fname = CHECKERS[checker]
    root = C.run_root(run)
    scenes = C.by_image(C.read_jsonl(root / "scenes.jsonl"))
    written = C.by_image(C.read_jsonl(root / "write.jsonl"))
    out = root / fname
    done = C.done_keys(out)
    todo = [i for i in written if (i,) not in done]
    if limit:
        todo = todo[:limit]
    print(f"check: {len(written)} written, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    tmpl = prompts.load("verifier_clause_instance_v2" if CLOSEUP else "verifier_clause_instance", non_kill=True)
    t0 = time.time()
    if checker == "qwen3vl":
        from shared.harness import tokens as T
        processor = T.load_capped_processor(HF, REV)
    else:
        processor = AutoProcessor.from_pretrained(HF, revision=REV)
    model = AutoModelForImageTextToText.from_pretrained(
        HF, revision=REV, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"{HF} loaded in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 1e9:.1f} GB",
          flush=True)

    def ask(views, category: str, clause: str) -> str:
        images = list(views) if isinstance(views, (list, tuple)) else [views]
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in images]
                                            + [{"type": "text", "text": tmpl.render(category=category, clause=clause)}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=images, text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        return processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    t0, calls = time.time(), 0
    for n, image_id in enumerate(todo, 1):
        sc, w = scenes[image_id], written[image_id]
        image = Image.open(C.image_path(image_id)).convert("RGB")
        cat = sc["category"]
        per_inst = []
        for inst in sc["instances"]:
            if target_only and not inst["is_target"]:
                continue
            view = C.outline(image, inst["box"], "red")
            if CLOSEUP:
                view = [view, C.closeup(image, inst["box"])]
            row = {"iid": inst["iid"], "is_target": inst["is_target"], "verdicts": [], "raw": []}
            for clause in w["clauses"]:
                a = ask(view, cat, clause)
                calls += 1
                row["verdicts"].append(C.yes_no_unclear(a))
                row["raw"].append(a.strip())
            if w.get("clause_neg"):
                a = ask(view, cat, w["clause_neg"])
                calls += 1
                row["neg_verdict"], row["neg_raw"] = C.yes_no_unclear(a), a.strip()
            else:
                row["neg_verdict"], row["neg_raw"] = None, None
            per_inst.append(row)
        target_row = next(r for r in per_inst if r["is_target"])
        rec = {
            "image_id": image_id,
            "instances": per_inst,
            "target_all_yes": all(v == "yes" for v in target_row["verdicts"]) if target_row["verdicts"] else None,
            "zero_satisfier": all(r["neg_verdict"] == "no" for r in per_inst) if w.get("clause_neg") else None,
            "siblings_distinct": (all(any(v == "no" for v in r["verdicts"]) for r in per_inst if not r["is_target"])
                                  if not target_only else None),
            "checker": f"{HF}@{REV[:8]}", "prompt": tmpl.name,
        }
        C.append_jsonl(out, rec)
        if n % 5 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {calls} calls, {(time.time() - t0) / max(calls, 1):.2f}s/call "
                  f"target_ok={rec['target_all_yes']} zero={rec['zero_satisfier']} "
                  f"distinct={rec['siblings_distinct']}", flush=True)
