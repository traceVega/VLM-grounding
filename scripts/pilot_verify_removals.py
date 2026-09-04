"""P10's verifier over the pilot: was the object actually removed?

    python scripts/pilot_verify_removals.py

The pilot measured what the model does on an edited image without ever checking
that the edit worked.  If the inpainting left the object visible, then a box on
the "removed" image is correct and counting it as a necessity violation is
simply wrong.  So the denominator has to be the removals that a verifier
confirms are clean.

That is P10, run here on the pilot rather than on K2: Gemma4-12B, a different
lineage from the policy under test, shown the **edit window** at up to 1,536 px
and asked one yes/no question:

    Is there a {class label} in this image? Answer with one word, yes or no.

"no" means the object is gone and the item counts; "yes" means the removal
failed and it is dropped.  Unparseable counts as not clean, which is the
conservative direction -- it drops items rather than admitting them.

This is a cheap layer, not a truth oracle. P11 exists to calibrate it against
three human annotators, and if Cohen's kappa comes in under 0.6 the human column
becomes decisive instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

HF = "google/gemma-4-12b-it"
REV = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"


def edit_rows() -> dict[str, dict]:
    from shared import paths

    shards = sorted((paths.EDITS_ROOT / "index").glob("shard-*.parquet"))
    table = pa.concat_tables([pq.read_table(s) for s in shards])
    cols = {n: table.column(n).to_pylist() for n in table.column_names}
    return {
        cols["image_id"][i]: {
            "window_xyxy_px": cols["window_xyxy_px"][i],
            "window_path": cols["window_path"][i],
        }
        for i in range(table.num_rows)
        if cols["operator"][i] == "REMOVE"
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", type=Path, default=Path("tables/pilot_does_it_look.json"))
    ap.add_argument("--out", type=Path, default=Path("tables/pilot_verified.json"))
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    from idea91.edits.build import load_edited, read_image
    from idea91.gate.removal_success import window_view
    from idea91.relations.verifier_rules import parse_yes_no
    from shared import paths
    from shared.harness import prompts

    records = json.loads(args.pilot.read_text())["records"]
    correct = [r for r in records if (r["original_iou_gt"] or 0) >= 0.5]
    print(f"{len(correct)} ORIGINAL-correct items to verify\n")

    edits = edit_rows()
    prompt = prompts.load("verifier_classlabel")
    processor = AutoProcessor.from_pretrained(HF, revision=REV)
    model = AutoModelForImageTextToText.from_pretrained(
        HF, revision=REV, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    image_dir = paths.RAW / "openimages" / "images"

    verified = []
    for n, rec in enumerate(correct, 1):
        image_id = rec["image_id"]
        label = rec["expr"].removeprefix("the ").strip()
        edit = edits.get(image_id)
        if edit is None:
            continue
        try:
            original = read_image(image_dir / f"{image_id}.jpg")
            removed = load_edited(original, edit, paths.EDITS_ROOT)
            view = window_view(removed, edit["window_xyxy_px"])
        except Exception as exc:
            print(f"  {image_id}: {type(exc).__name__}: {exc}")
            continue

        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt.render(class_label=label)}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[Image.fromarray(view)], text=text,
                           return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        answer = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        still_there = parse_yes_no(answer)
        verified.append({
            **rec,
            "verifier_answer": answer.strip(),
            # P10: unparseable counts as not clean, so it drops rather than admits
            "removal_clean": still_there is False,
        })
        if n % 25 == 0:
            clean = sum(v["removal_clean"] for v in verified)
            print(f"  [{n}/{len(correct)}] clean so far {clean}/{len(verified)}", flush=True)

    report(verified, args.out)


def report(verified: list[dict], out: Path) -> None:
    n = len(verified)
    clean = [v for v in verified if v["removal_clean"]]
    dirty = n - len(clean)
    print(f"\n{'='*68}")
    print(f"{n} items verified")
    print(f"  removal CLEAN (verifier says the object is gone): {len(clean)} ({len(clean)/max(n,1):.0%})")
    print(f"  removal FAILED (object still visible), dropped:   {dirty} ({dirty/max(n,1):.0%})")

    if not clean:
        print("\nnothing survived verification")
        return

    boxed = sum(v["removed_type"] == "box" for v in clean)
    declined = sum(v["removed_type"] == "none" for v in clean)
    same = [v for v in clean if (v["same_box"] or 0) >= 0.5]
    ious = [v["same_box"] for v in clean if v["same_box"] is not None]

    print(f"\nOn the {len(clean)} VERIFIED-CLEAN removals:")
    print(f"  still returns a box   {boxed}/{len(clean)} ({boxed/len(clean):.0%})")
    print(f"  declines (null)       {declined}/{len(clean)} ({declined/len(clean):.0%})")
    print(f"  SAME BOX (IoU>=0.5)   {len(same)}/{len(clean)} ({len(same)/len(clean):.0%})   <- headline")
    if ious:
        print(f"  median IoU(removed, original) = {np.median(ious):.3f}")

    print(f"\nfor comparison, before verification (n={n}):")
    b0 = sum(v["removed_type"] == "box" for v in verified)
    s0 = sum(1 for v in verified if (v["same_box"] or 0) >= 0.5)
    print(f"  still returns a box   {b0/max(n,1):.0%}   same box {s0/max(n,1):.0%}")
    print(f"{'='*68}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": n, "records": verified}, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
