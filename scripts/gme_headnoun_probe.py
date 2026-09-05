"""Same pixels, shorter words: is the absence signal lost to the scene or to the sentence?

    python scripts/gme_headnoun_probe.py

On class labels, removing the referent moves p(null) by sixteen orders of
magnitude and the model abstains half the time. On GroundingME's 39-word
expressions the same kind of removal moves it by two orders and the model
abstains 4% of the time. Two things changed between those runs -- the images
got harder and the prompts got longer -- and they have opposite implications.

This holds the images fixed. The human-clean GroundingME removals are asked
again with just the head noun ("the lamp") in the same prompt template, so the
only difference from the paired run already on disk is the expression. If the
signal comes back, the long description was suppressing it and the failure is
on the language side; if it stays flat, the scene is the reason and the
class-label result does not transfer.

Exploratory, `--non-kill`. The head noun is not the benchmark's expression, so
no number here is a GroundingME score.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.pilot_abstain_signal import _decision, _first_ids, auroc, iou

Image.MAX_IMAGE_PIXELS = None

HF = "Qwen/Qwen3-VL-8B-Instruct"
REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
OUT = Path("tables/gme_headnoun.jsonl")


def clean_items() -> list[dict]:
    from shared import paths

    with open(paths.DATA_ROOT / "gme_review" / "gme_labels.csv", encoding="utf-8") as fh:
        clean = {r["item_id"] for r in csv.DictReader(fh) if r["label"].strip() == "clean"}
    with open(paths.DATA_ROOT / "prepared/groundingme/dimensions.csv", encoding="utf-8") as fh:
        head = {r["item_id"]: r["head_noun"].strip() for r in csv.DictReader(fh)}
    rows = json.loads(Path("tables/gme_removals_index.json").read_text(encoding="utf-8"))
    out = [dict(r, head_noun=head.get(r["item_id"], "")) for r in rows
           if r["item_id"] in clean and head.get(r["item_id"])]
    out.sort(key=lambda r: r["item_id"])
    return out


def done_ids() -> set[str]:
    if not OUT.is_file():
        return set()
    return {json.loads(x)["item_id"]
            for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()}


def run() -> None:
    from transformers import AutoModelForImageTextToText

    from idea91.edits.build import load_edited
    from scripts.build_gme_removals import edits_root
    from shared import paths
    from shared.harness import parsers as P
    from shared.harness import prompts
    from shared.harness import tokens as T

    rows = clean_items()
    done = done_ids()
    todo = [r for r in rows if r["item_id"] not in done]
    print(f"{len(rows)} human-clean removals; {len(done)} done, {len(todo)} to run\n")
    if not todo:
        return

    template = prompts.load("grounding_qwen3vl_primary")
    processor = T.load_capped_processor(HF, REV)
    T.assert_cap_is_in_force(processor)
    tokenizer = processor.tokenizer
    null_ids = _first_ids(tokenizer, [" null", "null", " Null"])
    box_ids = _first_ids(tokenizer, [" [", "["])
    model = AutoModelForImageTextToText.from_pretrained(
        HF, revision=REV, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def ask(image: Image.Image, expr: str) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": template.render(expr=expr)}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0][inputs["input_ids"].shape[1]:].tolist()
        raw = processor.decode(seq, skip_special_tokens=True)
        parsed = P.parse_qwen3vl(raw, convention=P.RELATIVE_1000,
                                 original_wh=image.size, sent_wh=image.size)
        return {"output_type": parsed.output_type,
                "box": list(parsed.box_xyxy_px) if parsed.box_xyxy_px else None,
                **_decision(seq, out.scores, tokenizer, null_ids, box_ids)}

    started = time.time()
    for n, item in enumerate(todo, 1):
        original = np.asarray(Image.open(paths.DATA_ROOT / item["image_path"]).convert("RGB"))
        edited = Image.fromarray(load_edited(original, item, edits_root()))
        expr = f"the {item['head_noun']}"
        rec = {"item_id": item["item_id"], "dimension": item["dimension"],
               "head_noun": item["head_noun"], "expr_short": expr,
               "gt_box": item["gt_box"],
               "ORIGINAL": ask(Image.fromarray(original), expr),
               "REMOVE": ask(edited, expr)}
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        if n % 10 == 0:
            rate = (time.time() - started) / n
            print(f"  [{n}/{len(todo)}]  {rate:.1f}s/item", flush=True)
    print(f"\n-> {OUT}")


def report() -> None:
    short = {json.loads(x)["item_id"]: json.loads(x)
             for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()}
    long_o = {json.loads(x)["item_id"]: json.loads(x)
              for x in Path("tables/gme_original.jsonl").read_text(encoding="utf-8").splitlines()
              if x.strip()}
    long_r = {json.loads(x)["item_id"]: json.loads(x)
              for x in Path("tables/gme_remove.jsonl").read_text(encoding="utf-8").splitlines()
              if x.strip()}
    ids = [i for i in short if i in long_o and i in long_r]
    print(f"\n{len(ids)} human-clean removals, same images, two expressions each\n")

    # Only items the model grounded correctly under the SHORT prompt too can
    # show it stopped looking; report both frames.
    short_ok = [i for i in ids if short[i]["ORIGINAL"]["box"]
                and iou(short[i]["ORIGINAL"]["box"], short[i]["gt_box"]) >= 0.5]
    print(f"under the head noun, ORIGINAL box correct on {len(short_ok)}/{len(ids)}"
          f" (the 39-word expression was correct on all {len(ids)} by construction)\n")

    def cell(name, rows_o, rows_r):
        po = [r["p_null_norm"] for r in rows_o if r["p_null_norm"] is not None]
        pr = [r["p_null_norm"] for r in rows_r if r["p_null_norm"] is not None]
        dec = sum(1 for r in rows_r if r["output_type"] == "none")
        rose = sum(1 for a, b in zip(rows_o, rows_r)
                   if a["p_null_norm"] is not None and b["p_null_norm"] is not None
                   and b["p_null_norm"] > a["p_null_norm"])
        a = auroc(pr, po)
        print(f"  {name:<28} n={len(rows_r):3d}  declines {dec / max(len(rows_r), 1):4.0%}"
              f"   p(null) present {statistics.median(po):.1e} -> removed {statistics.median(pr):.1e}"
              f"   rose {rose / max(len(rows_r), 1):4.0%}   AUROC {a if a is not None else float('nan'):.3f}")

    for frame_name, frame in (("all clean items", ids), ("short-prompt-correct only", short_ok)):
        print(f"[{frame_name}]")
        cell("39-word expression", [long_o[i] for i in frame], [long_r[i] for i in frame])
        cell("head noun only", [short[i]["ORIGINAL"] for i in frame],
             [short[i]["REMOVE"] for i in frame])
        print()

    print("reading:")
    print("  head-noun AUROC back near 0.9+ and declines up  -> the sentence suppressed the signal (language side)")
    print("  head-noun AUROC still ~0.7 and declines still low -> the scene did (vision side); class-label result does not transfer")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        run()
    report()


if __name__ == "__main__":
    main()
