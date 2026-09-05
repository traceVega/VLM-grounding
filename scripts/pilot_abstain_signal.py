"""Does the model know the referent is gone, or only sometimes say so?

    python scripts/pilot_abstain_signal.py            # run, then report
    python scripts/pilot_abstain_signal.py --report   # re-read the saved run

Half the verified-clean removals get an abstention and half get a box, and the
rate alone cannot say which of two very different things is wrong:

    the model cannot tell the object left      -> a representation problem
    the model can tell but answers anyway      -> an operating point

Those have opposite costs to fix, so the measurement is the model's *decision*,
not its output.  The grounding prompt ends in a two-way choice, and the token
after `{"bbox_2d":` is either ` null` or ` [`; the probability it puts on each,
at that step, is the abstention it was willing to make.

Ground truth comes from the pair, not from a judge.  Every image is run in both
edited conditions:

    REMOVE       the referent is inpainted away   -> `null` is the only answer
    CONTROL_OBJ  a *different* object is removed  -> the referent is untouched

Same editor, same hole size, same artifact.  The only difference between the two
is whether the referent is still there, so "was this edited" carries nothing and
AUROC over `p(null)` is a clean read on whether the model can tell.  That is
P3's control doing the job it was designed for.

The pool is the 339 removals the author confirmed clean by eye, less the ones
the box review later dropped.  The verifier is not used: it agreed with the
author on 28% of a 32-item check.

CONTROL_OBJ also gives invariance, `V`, for the first time -- removing a
non-referent must not move the box -- so one run covers two relations.

Exploratory, `--non-kill`: class labels rather than referring expressions, one
model, and no number here may enter a kill table or change P8 to P21.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HF = "Qwen/Qwen3-VL-8B-Instruct"
REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

#: Results are appended a line at a time rather than written once at the end.
#: This host throws a transient `cudaErrorUnknown` often enough that a run of a
#: few hundred items will not finish in one attempt -- the first try died in a
#: layernorm at item 76 and took the other 75 with it -- and a poisoned CUDA
#: context cannot be recovered inside the process, so the only cheap protection
#: is to lose nothing and resume.  `run_all.sh` supervises on progress.
OUT = Path("tables/pilot_abstain.jsonl")
#: Items begun but never finished, so a crash that repeats on the same image
#: does not stall the supervisor forever.
ATTEMPTS = Path("tables/pilot_abstain.attempts")
MAX_ATTEMPTS = 2

#: `V` passes when a control edit leaves the box where it was.  The idea file's
#: default is 0.7; the complement at 0.5 is reported too, since P15 measures box
#: shift on controls at 0.5.
V_HIGH = 0.7


def pool(label: str = "clean") -> list[dict]:
    """Removals carrying one human label, with their matched control.

    `clean` is the measurement.  `not_clean` is the cell that decides what the
    measurement means: the inpainting ran on the referent and the object
    survived it anyway, so the artifact is present where REMOVE puts it while
    the object is present as in CONTROL_OBJ.  If p(null) is high there, the
    model is reading the artifact rather than the absence.
    """
    import csv

    import pyarrow as pa
    import pyarrow.parquet as pq

    from shared import paths

    root = Path(__file__).resolve().parents[1]
    with open(root / "data/human/openimages_removal_labels.csv", encoding="utf-8") as fh:
        clean = {r["image_id"] for r in csv.DictReader(fh) if r["label"].strip() == label}

    dropped: set[str] = set()
    box_csv = paths.DATA_ROOT / "box_review" / "box_labels.csv"
    if label == "clean" and box_csv.is_file():
        with open(box_csv, encoding="utf-8") as fh:
            dropped = {r["image_id"] for r in csv.DictReader(fh)
                       if r["label"].strip() in ("drop", "unsure")}

    table = pa.concat_tables([pq.read_table(s) for s in
                              sorted((paths.EDITS_ROOT / "index").glob("shard-*.parquet"))])
    col = {n: table.column(n).to_pylist() for n in table.column_names}
    keep = ("window_xyxy_px", "window_path", "mask_rle")
    edits: dict[tuple[str, str], dict] = {}
    for i in range(table.num_rows):
        if col["operator"][i] in ("REMOVE", "CONTROL_OBJ"):
            edits[(col["image_id"][i], col["operator"][i])] = {k: col[k][i] for k in keep}

    import sys

    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        from scripts.pilot_does_it_look import unambiguous_referents
        referents = unambiguous_referents()
    finally:
        sys.argv = argv

    out = []
    for image_id in sorted(clean - dropped):
        rem = edits.get((image_id, "REMOVE"))
        ctl = edits.get((image_id, "CONTROL_OBJ"))
        ref = referents.get(image_id)
        if rem is None or ctl is None or ref is None:
            continue
        out.append({"image_id": image_id, "expr": f"the {ref['label']}",
                    "gt_box": [float(v) for v in ref["bbox_xyxy_px"]],
                    "REMOVE": rem, "CONTROL_OBJ": ctl})
    return out


def _first_ids(tokenizer, strings) -> list[int]:
    ids = set()
    for s in strings:
        encoded = tokenizer.encode(s, add_special_tokens=False)
        if encoded:
            ids.add(encoded[0])
    return sorted(ids)


def _decision(seq: list[int], scores, tokenizer, null_ids, box_ids) -> dict:
    """The two-way choice at the token after `{"bbox_2d":`.

    Found by walking the generated tokens rather than by assuming a position,
    because the model sometimes prefixes the JSON with a word or a fence.
    """
    seen = ""
    for i, tid in enumerate(seq):
        piece = tokenizer.decode([tid])
        stripped = piece.strip()
        if "bbox_2d" in seen and (stripped.startswith("null") or stripped.startswith("[")):
            probs = torch.softmax(scores[i][0].float(), dim=-1)
            p_null = float(probs[null_ids].max())
            p_box = float(probs[box_ids].max())
            total = p_null + p_box
            return {"p_null": p_null, "p_box": p_box,
                    "p_null_norm": p_null / total if total > 0 else None,
                    "step": i}
        seen += piece
    return {"p_null": None, "p_box": None, "p_null_norm": None, "step": None}


def done_ids() -> set[str]:
    if not OUT.is_file():
        return set()
    ids = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["image_id"])
    return ids


def poisoned_ids(done: set[str]) -> set[str]:
    """Items begun `MAX_ATTEMPTS` times and never finished: skip them."""
    if not ATTEMPTS.is_file():
        return set()
    counts: dict[str, int] = {}
    for line in ATTEMPTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[line.strip()] = counts.get(line.strip(), 0) + 1
    return {k for k, v in counts.items() if v >= MAX_ATTEMPTS and k not in done}


def run(limit: int | None, label: str = "clean") -> None:
    from transformers import AutoModelForImageTextToText

    from idea91.edits.build import load_edited, read_image
    from shared import paths
    from shared.harness import parsers as P
    from shared.harness import prompts
    from shared.harness import tokens as T

    rows = pool(label)
    if limit:
        rows = rows[:limit]
    total = len(rows)
    done, skip = done_ids(), poisoned_ids(done_ids())
    rows = [r for r in rows if r["image_id"] not in done and r["image_id"] not in skip]
    print(f"{total} human-clean items; {len(done)} already done, "
          f"{len(skip)} skipped as repeatedly fatal, {len(rows)} to run\n")
    if not rows:
        return

    template = prompts.load("grounding_qwen3vl_primary")
    processor = T.load_capped_processor(HF, REV)
    T.assert_cap_is_in_force(processor)
    tokenizer = processor.tokenizer
    null_ids = _first_ids(tokenizer, [" null", "null", " Null"])
    box_ids = _first_ids(tokenizer, [" [", "["])
    print(f"decision tokens: null {null_ids}   box {box_ids}")

    model = AutoModelForImageTextToText.from_pretrained(
        HF, revision=REV, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    image_dir = paths.RAW / "openimages" / "images"

    def ask(image: np.ndarray, expr: str) -> tuple[str, dict]:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": template.render(expr=expr)}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[Image.fromarray(image)], text=text,
                           return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0][inputs["input_ids"].shape[1]:].tolist()
        raw = processor.decode(seq, skip_special_tokens=True)
        return raw, _decision(seq, out.scores, tokenizer, null_ids, box_ids)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for n, item in enumerate(rows, 1):
        with open(ATTEMPTS, "a", encoding="utf-8") as fh:
            fh.write(item["image_id"] + "\n")
        try:
            original = read_image(image_dir / f"{item['image_id']}.jpg")
            images = {
                "ORIGINAL": original,
                "REMOVE": load_edited(original, item["REMOVE"], paths.EDITS_ROOT),
                "CONTROL_OBJ": load_edited(original, item["CONTROL_OBJ"], paths.EDITS_ROOT),
            }
        except Exception as exc:
            print(f"  {item['image_id']}: {type(exc).__name__}: {exc}")
            continue

        wh = (original.shape[1], original.shape[0])
        rec = {"image_id": item["image_id"], "expr": item["expr"], "gt_box": item["gt_box"],
               "model": HF, "revision": REV}
        for condition, image in images.items():
            raw, decision = ask(image, item["expr"])
            parsed = P.parse_qwen3vl(raw, convention=P.RELATIVE_1000,
                                     original_wh=wh, sent_wh=wh)
            rec[condition] = {
                "output_type": parsed.output_type,
                "box": list(parsed.box_xyxy_px) if parsed.box_xyxy_px else None,
                **decision,
            }
        # Written and flushed per item: the next fault costs this one only.
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

        if n % 25 == 0:
            torch.cuda.empty_cache()
            rate = (time.time() - started) / n
            print(f"  [{n}/{len(rows)}]  {rate:.1f}s/item, "
                  f"{(len(rows) - n) * rate / 60:.0f} min left", flush=True)

    print(f"\n-> {OUT}  ({len(done_ids())} records)")


# --- analysis ---------------------------------------------------------------

def auroc(positive: list[float], negative: list[float]) -> float | None:
    """Rank-based AUROC, ties at half credit."""
    if not positive or not negative:
        return None
    values = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    ranks, i = {}, 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1][0] == values[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, y) in enumerate(values) if y == 1)
    n1, n0 = len(positive), len(negative)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(positive, negative, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.array(positive), np.array(negative)
    draws = [auroc(list(rng.choice(pos, len(pos))), list(rng.choice(neg, len(neg))))
             for _ in range(n_boot)]
    draws = [d for d in draws if d is not None]
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def report() -> None:
    import csv

    from shared import paths

    records = [json.loads(line) for line in
               OUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [r for r in records
          if r["ORIGINAL"]["box"] and iou(r["ORIGINAL"]["box"], r["gt_box"]) >= 0.5]
    print(f"{len(records)} items run, {len(ok)} with the ORIGINAL box correct\n")

    print("abstention rate, by condition (the model output a literal null bbox)")
    for condition in ("ORIGINAL", "REMOVE", "CONTROL_OBJ"):
        none = sum(1 for r in ok if r[condition]["output_type"] == "none")
        print(f"  {condition:12s} {none:4d}/{len(ok)}  {none / max(len(ok), 1):5.0%}")

    pos = [r["REMOVE"]["p_null_norm"] for r in ok if r["REMOVE"]["p_null_norm"] is not None]
    neg = [r["CONTROL_OBJ"]["p_null_norm"] for r in ok
           if r["CONTROL_OBJ"]["p_null_norm"] is not None]
    area = auroc(pos, neg)
    print(f"\n>>> AUROC  p(null) separating REMOVE from CONTROL_OBJ")
    print(f"    {area:.3f}   n={len(pos)} vs {len(neg)}" if area else "    not computable")
    if area:
        lo, hi = boot_ci(pos, neg)
        print(f"    95% CI  {lo:.3f} .. {hi:.3f}")
        print(f"    median p(null)   REMOVE {np.median(pos):.3f}   "
              f"CONTROL_OBJ {np.median(neg):.3f}")
        verdict = ("the information is in the model; abstention is an operating point"
                   if area >= 0.85 else
                   "partial signal; abstention needs more than a threshold" if area >= 0.7
                   else "the model largely cannot tell")
        print(f"    reading: {verdict}")

    print("\n>>> V (invariance): does a non-referent removal move the box?")
    both = [r for r in ok if r["CONTROL_OBJ"]["box"]]
    held = sum(1 for r in both if iou(r["CONTROL_OBJ"]["box"], r["ORIGINAL"]["box"]) >= V_HIGH)
    at50 = sum(1 for r in both if iou(r["CONTROL_OBJ"]["box"], r["ORIGINAL"]["box"]) >= 0.5)
    abstained = sum(1 for r in ok if r["CONTROL_OBJ"]["output_type"] == "none")
    print(f"    box held at IoU>=0.70   {held:4d}/{len(ok)}  {held / max(len(ok), 1):5.0%}"
          f"   <- V pass")
    print(f"    box held at IoU>=0.50   {at50:4d}/{len(ok)}  {at50 / max(len(ok), 1):5.0%}")
    print(f"    abstained on the control {abstained:3d}/{len(ok)}"
          f"  {abstained / max(len(ok), 1):5.0%}   <- the edit confused it")

    box_csv = paths.DATA_ROOT / "box_review" / "box_labels.csv"
    if not box_csv.is_file():
        return
    with open(box_csv, encoding="utf-8") as fh:
        human = {r["image_id"]: r["label"].strip() for r in csv.DictReader(fh)}
    right = [r["REMOVE"]["p_null_norm"] for r in ok
             if human.get(r["image_id"]) == "correct" and r["REMOVE"]["box"]
             and r["REMOVE"]["p_null_norm"] is not None]
    wrong = [r["REMOVE"]["p_null_norm"] for r in ok
             if human.get(r["image_id"]) == "nothing"
             and r["REMOVE"]["p_null_norm"] is not None]
    if right and wrong:
        area2 = auroc(wrong, right)
        print("\n>>> when it does answer, does p(null) know the answer is wrong?")
        print(f"    AUROC {area2:.3f}   n={len(wrong)} wrong vs {len(right)} right")
        print(f"    median p(null)  wrong {np.median(wrong):.3f}  right {np.median(right):.3f}")
        print("    (ground truth is the author's box review)")


def report_artifact_cell() -> None:
    """Is p(null) reading the object's absence, or the inpainting over it?

    `not_clean` fills the missing cell: the editor ran on the referent, so the
    artifact is where REMOVE puts it, and the object survived, so the referent
    is present as in CONTROL_OBJ.  Whichever of the two this condition matches
    is the thing p(null) was tracking all along.
    """
    def load(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    clean = [r for r in load(Path("tables/pilot_abstain.jsonl"))
             if r["ORIGINAL"]["box"] and iou(r["ORIGINAL"]["box"], r["gt_box"]) >= 0.5]
    dirty = [r for r in load(Path("tables/pilot_abstain_not_clean.jsonl"))
             if r["ORIGINAL"]["box"] and iou(r["ORIGINAL"]["box"], r["gt_box"]) >= 0.5]
    if not dirty:
        print("no not_clean run found")
        return

    def col(rows, cond):
        return [r[cond]["p_null_norm"] for r in rows if r[cond]["p_null_norm"] is not None]

    rem_clean, ctl_clean = col(clean, "REMOVE"), col(clean, "CONTROL_OBJ")
    rem_dirty = col(dirty, "REMOVE")

    print(f"ORIGINAL-correct items:  clean {len(clean)}   not_clean {len(dirty)}\n")
    print(f"{'condition':<44} {'n':>4}  {'median p(null)':>16}  {'log10':>7}  abstains")
    for name, rows, cond, values in (
            ("REMOVE, object GONE           (clean)", clean, "REMOVE", rem_clean),
            ("REMOVE, object STILL THERE    (not_clean)", dirty, "REMOVE", rem_dirty),
            ("CONTROL_OBJ, object untouched (clean)", clean, "CONTROL_OBJ", ctl_clean)):
        med = float(np.median(values))
        rate = sum(1 for r in rows if r[cond]["output_type"] == "none") / max(len(rows), 1)
        print(f"  {name:<42} {len(values):4d}  {med:16.3e}  "
              f"{np.log10(med) if med > 0 else float('-inf'):7.1f}  {rate:6.0%}")

    a_absence = auroc(rem_clean, rem_dirty)
    a_artifact = auroc(rem_dirty, ctl_clean)
    print(f"\n  AUROC(gone vs still-there, both with the artifact ON the referent)"
          f"  {a_absence:.3f}")
    print(f"      high  -> p(null) tracks the OBJECT'S ABSENCE")
    print(f"  AUROC(still-there-with-artifact vs untouched-control)             "
          f"  {a_artifact:.3f}")
    print(f"      high  -> p(null) also picks up the ARTIFACT itself")
    if a_absence is not None and a_artifact is not None:
        verdict = ("absence, cleanly -- the artifact adds little"
                   if a_absence >= 0.85 and a_artifact <= 0.65 else
                   "mostly the artifact -- the headline result needs restating"
                   if a_absence < 0.7 else "both, and they need separating")
        print(f"\n  reading: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="skip inference, read the saved run")
    ap.add_argument("--no-report", action="store_true", help="run only; the supervisor reports")
    ap.add_argument("--label", default="clean", choices=["clean", "not_clean"],
                    help="which human label defines the pool")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    global OUT, ATTEMPTS
    if args.label != "clean":
        OUT = Path(f"tables/pilot_abstain_{args.label}.jsonl")
        ATTEMPTS = Path(f"tables/pilot_abstain_{args.label}.attempts")

    if not args.report:
        run(args.limit, args.label)
    if not args.no_report:
        report() if args.label == "clean" else report_artifact_cell()


if __name__ == "__main__":
    main()
