"""Cross-family recheck of data recipe A (reviewer's reservation: writer and checker were both Gemini): Gemma4-12B, the
project's local checker, re-judges every kept falsified detail on every instance (zero-satisfier gate) and every description
detail on the target.  Prints the agreement with Gemini per cell and the falsifications Gemma would reject.

    python -m train.gme_recheck [--limit N]

Output: $VLMG_DATA_ROOT/train/gme_style/recheck.jsonl (per scene), resumable.  GPU only (no API).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import torch
from PIL import Image

from datagen import checker as CK
from datagen import common as C
from shared.harness import prompts
from train.gme_writer import ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    written = {(r["run"], r["group"]): r for r in C.read_jsonl(ROOT / "write.jsonl")}
    checks = [r for r in C.read_jsonl(ROOT / "check.jsonl")]
    out = ROOT / "recheck.jsonl"
    done = {(r["run"], r["group"]) for r in C.read_jsonl(out)} if out.is_file() else set()
    todo = [c for c in checks if (c["run"], c["group"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"recheck: {len(checks)} checked scenes, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    HF, REV, _ = CK.CHECKERS["gemma4"]
    tmpl = prompts.load("verifier_clause_instance_v2", non_kill=True)
    processor = AutoProcessor.from_pretrained(HF, revision=REV)
    model = AutoModelForImageTextToText.from_pretrained(HF, revision=REV, dtype=torch.bfloat16, device_map="cuda").eval()

    def ask(views, category: str, clause: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in views] + [{"type": "text", "text": tmpl.render(category=category, clause=clause)}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=list(views), text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        return processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    t0, calls, agree = time.time(), 0, Counter()
    for n, ck in enumerate(todo, 1):
        w = written[(ck["run"], ck["group"])]
        image = Image.open(w["image"]).convert("RGB")
        cat, n_d = w["category"], len(w["details"])
        rows = []
        for r in ck["rows"]:
            views = [C.outline(image, r["box"], "red"), C.closeup(image, r["box"])]
            rec = {"iid": r["iid"], "is_target": r["is_target"], "flip_verdicts": [], "detail_verdicts": []}
            for vi, v in enumerate(w["versions"]):  # every falsification on every instance
                a = C.yes_no_unclear(ask(views, cat, v["detail"]))
                calls += 1
                rec["flip_verdicts"].append(a)
                agree["flip_" + ("agree" if a == r["verdicts"][n_d + vi] else "differ")] += 1
            if r["is_target"]:  # the description's details on the target only
                for j, d in enumerate(w["details"]):
                    a = C.yes_no_unclear(ask(views, cat, d))
                    calls += 1
                    rec["detail_verdicts"].append(a)
                    agree["detail_" + ("agree" if a == r["verdicts"][j] else "differ")] += 1
            rows.append(rec)
        C.append_jsonl(out, {"run": ck["run"], "group": ck["group"], "rows": rows})
        if n % 10 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} scenes, {calls} calls, {(time.time() - t0) / 60:.1f} min, {dict(agree)}", flush=True)
    # summary: falsifications Gemma would not accept as zero-satisfier
    rej = tot = 0
    for rec in C.read_jsonl(out):
        w = written[(rec["run"], rec["group"])]
        for vi in range(len(w["versions"])):
            tot += 1
            if any(r["flip_verdicts"][vi] != "no" for r in rec["rows"] if vi < len(r["flip_verdicts"])):
                rej += 1
    print(f"falsifications with a non-'no' from Gemma on some instance: {rej}/{tot}; cell agreement {dict(agree)}", flush=True)


if __name__ == "__main__":
    main()
