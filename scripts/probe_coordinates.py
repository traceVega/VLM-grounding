"""Q-3: which coordinate convention does Qwen3-VL emit? (SPEC Section 1, P21)

    python scripts/probe_coordinates.py

SPEC allows ``relative_1000``, ``absolute_resized``, ``percent_float`` and
``loc_tokens``.  Qwen2-VL used 0-1000 relative; Qwen2.5-VL moved to absolute
pixels of the resized input.  Which applies to Qwen3-VL-8B-Instruct decides
every box in K2, and neither the model card nor either config file states it.

**The test does not use ground truth.**  It would be easy to try each reading
and keep whichever best matches the annotated box -- and that is exactly the
oracle-assisted decode this project refuses, because the REMOVE condition has no
ground-truth box and P14 compares REMOVE against ORIGINAL.  A convention picked
that way would also be circular: it would encode the model's accuracy into the
definition of its output.

Instead the probe sends the **same image twice at two different caps**, so the
resized frame differs while the depicted scene does not:

* coordinates that stay put are in a frame independent of the resize, so the
  convention is normalised -- ``relative_1000`` or ``percent_float``, told apart
  by magnitude;
* coordinates that scale with the resize are ``absolute_resized``.

Agreement with the annotated box is computed too, but only as corroboration in
the report -- never as the thing that decides.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

HF = "Qwen/Qwen3-VL-8B-Instruct"
REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
BIG = 2_457_600  # P21's cap
SMALL = 614_400  # a quarter of the area, so the resized frame halves per side


def load_items(n: int) -> list[dict]:
    """Positive GroundingME items with their annotated box, largest first.

    Large images are the informative ones: both caps then bite, so the two
    resized frames genuinely differ.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    files = sorted(
        glob.glob(
            "/home/jiaqi/.cache/huggingface/hub/datasets--lirang04--GroundingME"
            "/snapshots/*/data/*.parquet"
        )
    )
    frames = []
    for path in files:
        table = pq.read_table(path, columns=["id", "description", "bbox", "width", "height",
                                             "image"])
        frames.append(table.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df = df[df.bbox.notna()].copy()
    df["px"] = df.width * df.height
    df = df.sort_values("px", ascending=False).head(n)
    return df.to_dict("records")


def ask(model, processor, image, prompt: str, max_new_tokens: int = 64) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text",
                                                                "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)
    import torch

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


def numbers_in(text: str) -> list[float] | None:
    """The four numbers of a bbox_2d answer, whatever frame they are in."""
    import re

    match = re.search(r'"bbox_2d"\s*:\s*\[([^\]]*)\]', text)
    if not match:
        match = re.search(
            r"(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+"
            r"(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)",
            text,
        )
        return [float(v) for v in match.groups()] if match else None
    parts = [p.strip() for p in match.group(1).split(",")]
    try:
        return [float(p) for p in parts] if len(parts) == 4 else None
    except ValueError:
        return None


#: How far the observed ratio may sit from 1.0 and still count as "did not move".
STAYS_PUT = 0.08
#: ...and from the frame ratio to count as "scaled with the frame".
SCALES = 0.12
#: Above this, coordinates are 0-1000 rather than 0-100.
RELATIVE_1000_FLOOR = 100.0


def decide(ratio: float, frame_ratio: float, max_coordinate: float) -> tuple[str, str]:
    """Which convention the two-cap ratio implies, and why.

    ``UNDECIDED`` rather than a nearest-match, because a convention chosen by
    "which is least unlike the data" is a guess wearing a measurement's clothes,
    and every box in K2 depends on it.
    """
    if abs(ratio - 1.0) < STAYS_PUT:
        if max_coordinate > RELATIVE_1000_FLOOR:
            return "relative_1000", "coordinates did not move when the resized frame changed"
        return "percent_float", "coordinates did not move, and are on a 0-100 scale"
    if abs(ratio - frame_ratio) < SCALES:
        return "absolute_resized", "coordinates scaled with the resized frame"
    return (
        "UNDECIDED",
        f"ratio {ratio:.3f} matches neither 1.0 nor the frame ratio {frame_ratio:.3f}",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="items to probe")
    ap.add_argument("--out", type=Path, default=Path("tables/q3_coordinate_probe.json"))
    args = ap.parse_args()

    import io

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from shared.harness import prompts

    template = prompts.load("grounding_qwen3vl_primary")
    print(f"loading {HF} @ {REV[:12]} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        HF, revision=REV, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    processors = {
        "big": AutoProcessor.from_pretrained(HF, revision=REV, max_pixels=BIG),
        "small": AutoProcessor.from_pretrained(HF, revision=REV, max_pixels=SMALL),
    }

    records = []
    for item in load_items(args.n):
        image = Image.open(io.BytesIO(item["image"]["bytes"])).convert("RGB")
        prompt = template.render(expr=item["description"])
        answers, frames = {}, {}
        for name, processor in processors.items():
            answers[name] = numbers_in(ask(model, processor, image, prompt))
            # what the processor actually produced, for absolute_resized
            probe = processor(images=[image], text="<|image_pad|>", return_tensors="pt")
            grid = probe["image_grid_thw"][0]
            frames[name] = (int(grid[2]) * 16, int(grid[1]) * 16)
        if not answers["big"] or not answers["small"]:
            continue
        records.append(
            {
                "id": int(item["id"]),
                "original_wh": [int(item["width"]), int(item["height"])],
                "frame_big": frames["big"],
                "frame_small": frames["small"],
                "answer_big": answers["big"],
                "answer_small": answers["small"],
                "gt_bbox": [float(v) for v in item["bbox"]],
            }
        )
        print(f"  id {item['id']}: big {answers['big']}  small {answers['small']}")

    if not records:
        raise SystemExit("no parseable answers; cannot decide Q-3")

    # The decision: do the numbers move with the resized frame, or not?
    ratios, expected = [], []
    for r in records:
        big, small = np.array(r["answer_big"]), np.array(r["answer_small"])
        nonzero = np.abs(big) > 1e-6
        if not nonzero.any():
            continue
        ratios.append(float(np.median(small[nonzero] / big[nonzero])))
        expected.append(r["frame_small"][0] / r["frame_big"][0])

    ratio = float(np.median(ratios))
    frame_ratio = float(np.median(expected))
    magnitude = float(np.max([np.max(np.abs(r["answer_big"])) for r in records]))

    print(f"\nmedian small/big coordinate ratio: {ratio:.3f}")
    print(f"median small/big resized-frame ratio: {frame_ratio:.3f}")
    print(f"largest coordinate seen: {magnitude:.1f}")

    convention, why = decide(ratio, frame_ratio, magnitude)
    print(f"\nQ-3: {convention}  ({why})")
    if convention == "UNDECIDED":
        print("Do not guess. Widen --n, or read the answers in the JSON below.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": HF, "revision": REV,
                "max_pixels_big": BIG, "max_pixels_small": SMALL,
                "coordinate_ratio": ratio, "frame_ratio": frame_ratio,
                "max_coordinate": magnitude,
                "convention": convention, "reason": why,
                "method": "same image at two caps; ground truth never used to choose",
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"-> {args.out}")
    if convention != "UNDECIDED":
        print(f"\nSet in configs/models/qwen3vl-8b-instruct.yaml:\n"
              f"  coordinate_convention: {convention}")


if __name__ == "__main__":
    main()
