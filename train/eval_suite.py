"""The PoC evaluation suite, run identically before and after training.

    python -m train.eval_suite --tag base4b [--model 4b] [--adapter <dir>] [--sets gme,own,gray,refcoco]
                               [--trace yn|obs] [--batch 8] [--gme-pos-n 150] [--refcoco-n 100]

Sets:
  gme      GroundingME, all 1,005 items: Rejection accuracy (201; parse failures count as
           correct, as the benchmark scores it), positive accuracy IoU>=0.5 (804) by dimension,
           decision-token p(null) mass on both (coordinate mode only).
  own      the held-out scenes of the PoC split (same split as train.sft_lora).
  gray     the same held-out items with a gray image of the same size (text-shortcut control).
  gmegray  GroundingME text-shortcut control: all 201 Rejection items plus a stratified sample of
           positives (--gme-gray-pos-n), each with a gray image of the original size.  A model that
           refuses these as often as the real images is reading the words, not the picture.
  refcoco  --refcoco-n sampled boxes from each of RefCOCO / RefCOCO+ / RefCOCOg val (short expressions).

--gme-pos-n N keeps every Rejection item and a stratified (by dimension) sample of N positives:
the screening configuration used between training variants.

--trace yn|obs switches to the verification-trace protocol (train.traces): the prompt asks for
conditions, per-candidate verdicts and an answer; the scored answer is the one *derived* from
the verdict table (the free answer is recorded too), and the trace metrics are added: format
validity, free/derived consistency, candidate count, false "no" cells on the ground-truth
candidate, the accusation on own negatives, and same-scene pair accuracy on `own`.

--batch B generates B items at a time (left padding).  Per-item records are appended to
$VLMG_DATA_ROOT/train/eval/<tag>/<set>.jsonl and the run resumes by id.  `summary.json` holds
the numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from scripts.pilot_abstain_signal import _decision, _first_ids
from shared.harness import parsers as P
from shared.harness import prompts
from shared.harness import tokens as T
from train import data as D
from train import traces as TR

TRACE_PROMPTS = {"yn": "grounding_verify_trace", "obs": "grounding_verify_trace_obs", "rat": "grounding_verify_trace_tool"}  # rat = rationale cells, used with --turns 2


def load_model(model_key: str, adapter: str | None, max_pixels: int | None = None):
    from transformers import AutoModelForImageTextToText

    hf, rev = D.MODELS[model_key]
    processor = T.load_capped_processor(hf, rev, max_pixels=max_pixels or T.MAX_PIXELS)
    if not max_pixels:
        T.assert_cap_is_in_force(processor)
    model = AutoModelForImageTextToText.from_pretrained(hf, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


def gme_subset(pos_n: int | None, seed: int, rej_n: int | None = None) -> list[dict]:
    """All Rejection items (or a seeded sample of rej_n) plus a seeded, dimension-stratified sample of pos_n positives."""
    items = D.gme_items()
    if not pos_n and not rej_n:
        return items
    rej = [it for it in items if it["dimension"] == "Rejection"]
    if rej_n:
        rej = sorted(rej, key=lambda it: it["id"])
        random.Random(seed).shuffle(rej)
        rej = rej[:rej_n]
    if not pos_n:
        return rej
    pos = [it for it in items if it["dimension"] != "Rejection"]
    dims = sorted({it["dimension"] for it in pos})
    rng = random.Random(seed)
    out = list(rej)
    for d in dims:
        sub = [it for it in pos if it["dimension"] == d]
        rng.shuffle(sub)
        out.extend(sub[: round(pos_n * len(sub) / len(pos))])
    return out


def with_gray(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        with Image.open(it["image"]) as im:
            wh = list(im.size)
        out.append({**it, "image_wh": wh, "gray": True})
    return out


def own_items(args) -> list[dict]:
    _, val = D.split(D.load_items(), args.n_val_scenes, args.split_seed)
    return [{"id": it["id"], "expr": it["expression"], "image": it["image"], "image_wh": it["image_wh"],
             "gt_boxes": [it["answer"]["bbox_2d"]] if it["answer"]["bbox_2d"] else [], "n_gt": int(it["answer"]["bbox_2d"] is not None),
             "kind": it["kind"], "group": it["group"], "flipped_clause": (it.get("flipped") or {}).get("clause_neg")} for it in val]


def items_for(set_name: str, args) -> list[dict]:
    if set_name == "gme":
        return gme_subset(args.gme_pos_n, args.split_seed, args.gme_rej_n)
    if set_name == "gmegray":
        return with_gray(gme_subset(args.gme_gray_pos_n, args.split_seed))
    if set_name == "own":
        return own_items(args)
    if set_name == "gray":
        return [{**it, "gray": True} for it in own_items(args)]
    if set_name == "refcoco":
        return D.refcoco_items(args.refcoco_n, args.split_seed)
    raise ValueError(set_name)


def done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}


def _trim(seq: list[int], eos: set[int]) -> list[int]:
    out = []
    for t in seq:
        if t in eos:
            break
        out.append(t)
    return out


def _open(it: dict):
    image = D.gray_image(it["image_wh"]) if it.get("gray") else D.open_image(it)
    it["_wh"] = image.size
    return image


@torch.no_grad()
def generate_batch(model, processor, template, items: list[dict], max_new: int, want_scores: bool, views_of=None, text_of=None, stop: str | None = None):
    """Greedy generation for a list of items (left padding); returns (raw texts, token ids, scores per item).
    views_of(item, image) -> list of prompt images (default [image]); text_of(item) -> prompt text."""
    images, texts = [], []
    for it in items:
        image = _open(it)
        views = views_of(it, image) if views_of else [image]
        images.extend(views)
        text = text_of(it) if text_of else template.render(expr=it["expr"])
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in views] + [{"type": "text", "text": text}]}]
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    processor.tokenizer.padding_side = "left"
    inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(model.device)
    extra = {"stop_strings": [stop], "tokenizer": processor.tokenizer} if stop else {}
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, output_scores=want_scores, return_dict_in_generate=True, **extra)
    n_prompt = inputs["input_ids"].shape[1]
    tok = processor.tokenizer
    eos = {t for t in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None}
    raws, seqs, scores = [], [], []
    for b in range(len(items)):
        seq = _trim(out.sequences[b][n_prompt:].tolist(), eos)
        seqs.append(seq)
        raws.append(processor.decode(seq, skip_special_tokens=True))
        scores.append([s[b: b + 1] for s in out.scores] if want_scores else None)
    return raws, seqs, scores


MAX_NEW1 = 400  # turn-1 budget: conditions + tool call; 200 truncated 91/199 GME tool calls (long descriptions)


@torch.no_grad()
def generate_two_turn(model, processor, template, items: list[dict], max_new: int = 384, max_new1: int | None = None):
    """The v2 protocol at inference: turn 1 (conditions + zoom tool call) is generated until
    </tool_call>, the named boxes are cropped, turn 2 is generated with the crops appended.
    Returns (raw texts = turn1 + newline + turn2, token ids of turn 2, None scores); sets
    `_tool_boxes` (pixels) and `_turn1` on every item."""
    tok = processor.tokenizer
    raws1, _, _ = generate_batch(model, processor, template, items, max_new1 or MAX_NEW1, want_scores=False, stop="</tool_call>")
    for it, r1 in zip(items, raws1):
        it["_turn1"] = r1
        it["_tool_boxes"] = TR.parse_tool_boxes(r1, it["_wh"])
    if AUDIT == "coa":
        return _coa_turn2(model, processor, template, items, max_new)
    if ISOLATE:
        return _isolated_turn2(model, processor, template, items, max_new)
    convs, images = [], []
    for it, r1 in zip(items, raws1):
        image = _open(it)
        boxes = it["_tool_boxes"]
        crops = TR.turn2_views(image, boxes)
        u1 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": template.render(expr=it["expr"])}]}
        a1 = {"role": "assistant", "content": [{"type": "text", "text": r1}]}
        u2 = {"role": "user", "content": [{"type": "image"} for _ in crops] + [{"type": "text", "text": TR.turn2_user_text(len(boxes)) if boxes else "No valid boxes were given; answer from the full image. Continue with step 2."}]}
        convs.append(processor.apply_chat_template([u1, a1, u2], tokenize=False, add_generation_prompt=True))
        images.extend([image] + crops)
    processor.tokenizer.padding_side = "left"
    inputs = processor(images=images, text=convs, padding=True, return_tensors="pt").to(model.device)
    eos = {t for t in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None}
    n_prompt = inputs["input_ids"].shape[1]
    if SAMPLES > 1:
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=SAMPLE_TEMP, top_p=1.0, top_k=0, num_return_sequences=SAMPLES)
        raws, seqs = [], []
        for b, it in enumerate(items):
            wh = it["_wh"]
            texts = [processor.decode(_trim(out[b * SAMPLES + s_][n_prompt:].tolist(), eos), skip_special_tokens=True) for s_ in range(SAMPLES)]
            parsed = [TR.parse(it["_turn1"] + "\n" + t, tuple(wh)) for t in texts]
            table = _vote_table(parsed, it["_tool_boxes"], wh)
            pr = TR.parse(it["_turn1"] + "\n" + table + '<answer>{"bbox_2d": null}</answer>', tuple(wh))
            ans = json.dumps({"bbox_2d": _rel(pr["derived_box"], wh)}) if pr.get("derived_type") == "box" and pr.get("derived_box") else '{"bbox_2d": null}'
            raws.append(it["_turn1"] + "\n" + table + f"<answer>{ans}</answer>")
            seqs.append([])
        return raws, seqs, [None] * len(items)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    raws, seqs = [], []
    for b, it in enumerate(items):
        seq = _trim(out[b][n_prompt:].tolist(), eos)
        seqs.append(seq)
        raws.append(it["_turn1"] + "\n" + processor.decode(seq, skip_special_tokens=True))
    return raws, seqs, [None] * len(items)


ISOLATE = False  # --isolate: turn 2 once per candidate with only that crop, rows assembled afterwards (rows cannot copy each other)
SAMPLES = 1  # --samples k: turn 2 sampled k times, verdict cells decided by majority vote (test-time self-consistency)
AUDIT = None  # --audit coa: turn 2 = one COA audit per candidate (train.coa), decision by the harness rule
AUDIT_BASE = False  # --audit-base: run the audit turn with the adapter disabled (zero-training probe)
_COA_TMPL = None
_COA_ANS_TMPL = None


@torch.no_grad()
def _coa_turn2(model, processor, template, items: list[dict], max_new: int, sub_batch: int = 8):
    """Propose -> audit: for every candidate box of an item, one isolated turn-2 call with the full scene
    (candidate outlined in red) + its close-up and the COA audit prompt; the harness decides from the
    typed lines (train.coa.decide).  The record's raw text carries an equivalent <candidates> table for
    the trace scorer, the answer, and the audits as JSON after the answer."""
    global _COA_TMPL
    from datagen import common as C
    from train import coa as COA

    if _COA_TMPL is None:
        _COA_TMPL = prompts.load("grounding_coa_audit", non_kill=True)
    tok = processor.tokenizer
    eos = {t for t in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None}
    convs, images, owners = [], [], []
    for i, it in enumerate(items):
        boxes = it["_tool_boxes"]
        if not boxes:
            continue
        image = _open(it)
        wh = it["_wh"]
        u1 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": template.render(expr=it["expr"])}]}
        t1 = it["_turn1"]
        a, z = t1.find("<tool_call>"), t1.find("</tool_call>")
        for k, b in enumerate(boxes, 1):
            call = "<tool_call>" + json.dumps({"name": "image_zoom_in", "arguments": {"boxes": [_rel(b, wh)]}}) + "</tool_call>"
            t1_k = (t1[:a] + call + t1[z + len("</tool_call>"):]) if 0 <= a < z else (t1 + "\n" + call)
            a1 = {"role": "assistant", "content": [{"type": "text", "text": t1_k}]}
            u2 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _COA_TMPL.render()}]}
            convs.append(processor.apply_chat_template([u1, a1, u2], tokenize=False, add_generation_prompt=True))
            images.extend([C.outline(image, b, "red"), TR.crop_view(image, b)])  # two images: the outlined scene (turn 1) + the close-up (turn 2)
            owners.append((i, k))
    texts: dict[tuple[int, int], str] = {}
    processor.tokenizer.padding_side = "left"
    ctx = model.disable_adapter() if (AUDIT_BASE and hasattr(model, "disable_adapter")) else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for s0 in range(0, len(convs), sub_batch):
            tx = convs[s0:s0 + sub_batch]
            inputs = processor(images=images[2 * s0: 2 * (s0 + len(tx))], text=tx, padding=True, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            n_prompt = inputs["input_ids"].shape[1]
            for j, key in enumerate(owners[s0:s0 + len(tx)]):
                texts[key] = processor.decode(_trim(out[j][n_prompt:].tolist(), eos), skip_special_tokens=True)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    # third step: the model's own answer from its audits (text only, batched); the rule-derived answer is kept as a diagnostic
    global _COA_ANS_TMPL
    if _COA_ANS_TMPL is None:
        _COA_ANS_TMPL = prompts.load("grounding_coa_answer", non_kill=True)
    per_item = []
    ans_texts = []
    for i, it in enumerate(items):
        wh = it["_wh"]
        boxes = it["_tool_boxes"]
        raw_audits = [texts.get((i, k), "") for k in range(1, len(boxes) + 1)]
        audits = [COA.parse_audit(t) for t in raw_audits]
        per_item.append((raw_audits, audits))
        if boxes:
            body = _COA_ANS_TMPL.render(expr=it["expr"], turn1=it["_turn1"], audits=COA.audits_text(raw_audits, [_rel(b, wh) for b in boxes]))
            ans_texts.append(processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": body}]}], tokenize=False, add_generation_prompt=True))
        else:
            ans_texts.append(None)
    model_ans: dict[int, str] = {}
    todo = [i for i, t in enumerate(ans_texts) if t is not None]
    for s0 in range(0, len(todo), 8):
        idx = todo[s0:s0 + 8]
        inputs = processor(text=[ans_texts[i] for i in idx], padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        n_prompt = inputs["input_ids"].shape[1]
        for j, i in enumerate(idx):
            model_ans[i] = processor.decode(_trim(out[j][n_prompt:].tolist(), eos), skip_special_tokens=True)
    raws, seqs = [], []
    for i, it in enumerate(items):
        wh = it["_wh"]
        boxes = it["_tool_boxes"]
        raw_audits, audits = per_item[i]
        ans_box, chosen, fits = COA.decide(audits, boxes)
        n_cond = len(re.findall(r"^\s*\d+\.", it["_turn1"].split("</conditions>")[0], re.M)) or 1
        table = COA.synthetic_table(audits, [_rel(b, wh) for b in boxes], fits, n_cond) if boxes else "<candidates>\n</candidates>\n"
        derived = json.dumps({"bbox_2d": _rel(ans_box, wh)}) if ans_box else '{"bbox_2d": null}'
        ma = model_ans.get(i, "")
        am = re.search(r"<answer>(.*?)</answer>", ma, re.S)
        model_answer = ("<answer>" + am.group(1).strip() + "</answer>") if am else '<answer>{"bbox_2d": null}</answer>'
        extra = json.dumps({"audits": raw_audits, "fits": fits, "chosen": chosen, "derived": derived, "model_answer_raw": ma,
                            "k0": not boxes, "unparsable": sum(1 for a in audits if not a["format_ok"])}, ensure_ascii=False)
        # the <candidates> table encodes the rule-derived answer (scorer: rejection_acc / positive_acc); <answer> is the model's own
        # decision (scorer: free_* metrics); consistent_rate = agreement between the two
        raws.append(it["_turn1"] + "\n" + table + model_answer + f"\n<coa>{extra}</coa>")
        seqs.append([])
    return raws, seqs, [None] * len(items)
SAMPLE_TEMP = 0.7


def _vote_table(parsed: list[dict], boxes, wh) -> str:
    """Majority vote per (row, clause) over parsed samples; reasons from the first sample carrying the
    winning verdict; ties -> unclear.  Rows are the tool-call candidates."""
    J = max((len(pr["conditions"]) for pr in parsed), default=0)
    lines = ["<candidates>"]
    for k, b in enumerate(boxes, 1):
        cells = []
        for j in range(1, J + 1):
            votes, reasons = {}, {}
            for pr in parsed:
                c = next((c for c in pr["candidates"] if c["id"] == k), None)
                v = c["verdicts"].get(j) if c else None
                if v in ("yes", "no", "unclear"):
                    votes[v] = votes.get(v, 0) + 1
                    reasons.setdefault(v, (c["seen"] or {}).get(j))
            if not votes:
                v = "unclear"
            else:
                top = max(votes.values())
                winners = [v for v, n in votes.items() if n == top]
                v = winners[0] if len(winners) == 1 else "unclear"
            why = reasons.get(v)
            cells.append(f"{j}: {v}" + (f" — {why}" if why else ""))
        excluded = any(c.split(":")[1].strip().startswith("no") for c in cells)
        lines.append(f'<object id="{k}" bbox="{json.dumps(_rel(b, wh))}">' + "; ".join(cells) + (" -> excluded" if excluded else " -> fits") + "</object>")
    lines.append("</candidates>")
    return "\n".join(lines) + "\n"


def _rel(b, wh) -> list[int]:
    return [round(b[0] / wh[0] * 1000), round(b[1] / wh[1] * 1000), round(b[2] / wh[0] * 1000), round(b[3] / wh[1] * 1000)]


@torch.no_grad()
def _isolated_turn2(model, processor, template, items: list[dict], max_new: int, sub_batch: int = 8):
    """Isolated verification: for every candidate box of an item, turn 2 is generated with the full
    image and that one close-up, so the row for candidate k is written without seeing the other
    rows.  The rows are then assembled into one table (ids and boxes rewritten to the candidate's),
    and the answer is the table-derived one, so free and derived answers agree."""
    tok = processor.tokenizer
    eos = {t for t in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None}
    convs, images, owners = [], [], []
    for i, it in enumerate(items):
        boxes = it["_tool_boxes"]
        if not boxes:
            continue
        image = _open(it)
        wh = it["_wh"]
        u1 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": template.render(expr=it["expr"])}]}
        u2 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": TR.turn2_user_text(1)}]}
        t1 = it["_turn1"]
        a, z = t1.find("<tool_call>"), t1.find("</tool_call>")
        for k, b in enumerate(boxes, 1):
            # the turn-1 text shown to the model names only this candidate's box, so it writes one row about this close-up
            call = "<tool_call>" + json.dumps({"name": "image_zoom_in", "arguments": {"boxes": [_rel(b, wh)]}}) + "</tool_call>"
            t1_k = (t1[:a] + call + t1[z + len("</tool_call>"):]) if 0 <= a < z else (t1 + "\n" + call)
            a1 = {"role": "assistant", "content": [{"type": "text", "text": t1_k}]}
            convs.append(processor.apply_chat_template([u1, a1, u2], tokenize=False, add_generation_prompt=True))
            images.extend([image, TR.crop_view(image, b)])
            owners.append((i, k))
    cells: dict[tuple[int, int], str | None] = {}
    processor.tokenizer.padding_side = "left"
    for s0 in range(0, len(convs), sub_batch):
        texts = convs[s0:s0 + sub_batch]
        inputs = processor(images=images[2 * s0: 2 * (s0 + len(texts))], text=texts, padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        n_prompt = inputs["input_ids"].shape[1]
        for j, key in enumerate(owners[s0:s0 + len(texts)]):
            txt = processor.decode(_trim(out[j][n_prompt:].tolist(), eos), skip_special_tokens=True)
            m = TR._OBJ.search(txt)
            cells[key] = m.group(3).strip() if m else None
    raws, seqs = [], []
    for i, it in enumerate(items):
        wh = it["_wh"]
        lines = []
        for k, b in enumerate(it["_tool_boxes"], 1):
            c = cells.get((i, k)) or "1: no -> excluded"  # no parseable row: the candidate is not confirmed
            lines.append(f'<object id="{k}" bbox="{json.dumps(_rel(b, wh))}">{c}</object>')
        table = "<candidates>\n" + "\n".join(lines) + "\n</candidates>\n"
        pr = TR.parse(it["_turn1"] + "\n" + table + '<answer>{"bbox_2d": null}</answer>', tuple(wh))
        ans = json.dumps({"bbox_2d": _rel(pr["derived_box"], wh)}) if pr.get("derived_type") == "box" and pr.get("derived_box") else '{"bbox_2d": null}'
        raws.append(it["_turn1"] + "\n" + table + f"<answer>{ans}</answer>")
        seqs.append([])
    return raws, seqs, [None] * len(items)


_OBJ_ID = re.compile(r'<object\s+id="?(\d+)"?')
_NUMCOLON = re.compile(r"(\d+)\s*:\s*$")


def cell_probs(seq: list[int], scores, tok, groups: dict[str, set[int]]) -> dict[str, dict[str, float]]:
    """Per candidate and per condition, the normalised probability of 'no' at the verdict token
    (the token after 'k:' inside an <object> line).  Keys are '<cand id>:<k>'."""
    out: dict[str, dict[str, float]] = {}
    text = ""
    for i, tid in enumerate(seq):
        m = _NUMCOLON.search(text)
        if m:
            cands = _OBJ_ID.findall(text)
            if cands:
                key = f"{cands[-1]}:{m.group(1)}"
                if key not in out and i < len(scores):
                    probs = torch.softmax(scores[i][0].float(), dim=-1)
                    p = {g: float(probs[list(ids)].sum()) for g, ids in groups.items()}
                    z = sum(p.values()) or 1.0
                    out[key] = {g: round(v / z, 4) for g, v in p.items()}
        text += tok.decode([tid])
    return out


_PRIMARY = None
_CROP_TMPL = None


HINT_PROCESSOR = None  # optional processor with a different pixel cap for the hint stage (--hint-max-pixels)


_TRACE_TMPL = None


@torch.no_grad()
def hint_stage(model, processor, template, chunk: list[dict], max_new: int = 64, n_crops: int = 1):
    """Self-hint crop, stage 1: the base weights (adapters disabled) answer the primary
    coordinate prompt; the box becomes `_hint_box` and its close-up the second image.  With
    n_crops > 1 the base also runs the zero-shot trace prompt and its candidate boxes (not
    overlapping the hint) supply further close-ups, `_hint_boxes`.
    Returns (views_of, text_of) for generate_batch with the crop trace prompt."""
    global _PRIMARY, _CROP_TMPL, _TRACE_TMPL
    if _PRIMARY is None:
        _PRIMARY = prompts.load("grounding_qwen3vl_primary")
        _CROP_TMPL = prompts.load("grounding_verify_trace_crop", non_kill=True)
        _TRACE_TMPL = prompts.load("grounding_verify_trace_crops", non_kill=True)
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else None
    if ctx is not None:
        ctx.__enter__()
    try:
        raws, _, _ = generate_batch(model, HINT_PROCESSOR or processor, _PRIMARY, chunk, max_new, want_scores=False)
        cand_raws = None
        if n_crops > 1:
            cand_raws, _, _ = generate_batch(model, processor, TRACE_TEMPLATE_YN(), chunk, 320, want_scores=False)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    for i, (it, raw) in enumerate(zip(chunk, raws)):
        wh = it["_wh"]
        parsed = P.parse_qwen3vl(raw, convention=P.RELATIVE_1000, original_wh=wh, sent_wh=wh)
        hb = list(parsed.box_xyxy_px) if (parsed.output_type == "box" and parsed.box_xyxy_px) else None
        it["_hint_box"] = hb
        boxes = [hb] if hb else []
        if cand_raws is not None:
            pr = TR.parse(cand_raws[i], wh)
            for c in sorted([c for c in pr["candidates"] if c["box"]], key=lambda c: -(c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1])):
                if len(boxes) >= n_crops:
                    break
                if all(P.iou(tuple(c["box"]), tuple(b)) < 0.5 for b in boxes):
                    boxes.append(c["box"])
        it["_hint_boxes"] = boxes

    def views_of(it, image):
        boxes = it.get("_hint_boxes") or []
        return [image] + [TR.crop_view(image, b) for b in boxes] if boxes else [image, image]

    def text_of(it):
        boxes = it.get("_hint_boxes") or []
        if n_crops > 1:
            return _TRACE_TMPL.render(expr=it["expr"], hints=TR.hints_text(boxes, it["_wh"]) if boxes else "[0, 0, 1000, 1000]")
        return _CROP_TMPL.render(expr=it["expr"], hint=TR.hint_text(boxes[0], it["_wh"]) if boxes else "[0, 0, 1000, 1000]")

    return views_of, text_of


_YN = None


def TRACE_TEMPLATE_YN():
    global _YN
    if _YN is None:
        _YN = prompts.load(TRACE_PROMPTS["yn"], non_kill=True)
    return _YN


def score_item(it: dict, raw: str, seq, scores, processor, null_ids, box_ids, trace: str | None, max_new: int, groups=None) -> dict:
    wh = it["_wh"]
    rec = {"id": it["id"], "n_gt": it["n_gt"]}
    if "_hint_box" in it:
        hb = it["_hint_box"]
        rec["hint_box"] = hb
        rec["hint_iou"] = round(max((P.iou(tuple(hb), tuple(g)) for g in it["gt_boxes"]), default=0.0), 4) if (hb and it.get("gt_boxes")) else None
    if "_tool_boxes" in it:
        tb = it["_tool_boxes"]
        rec["n_tool_boxes"] = len(tb)
        rec["cand_recall_iou"] = round(max((P.iou(tuple(b), tuple(g)) for b in tb for g in it["gt_boxes"]), default=0.0), 4) if (tb and it.get("gt_boxes")) else None
        rec["hint_iou"] = round(max((P.iou(tuple(tb[0]), tuple(g)) for g in it["gt_boxes"]), default=0.0), 4) if (tb and it.get("gt_boxes")) else None
    if trace:
        pr = TR.parse(raw, wh)
        if scores is not None and groups:
            rec["cell_pno"] = cell_probs(seq, scores, processor.tokenizer, groups)
        box = pr["derived_box"]
        otype = {"box": "box", "null": "none", "invalid": "invalid"}[pr["derived_type"]]
        fbox = pr["free_box"]
        free_iou = max((P.iou(tuple(fbox), tuple(g)) for g in it["gt_boxes"]), default=0.0) if fbox else 0.0
        rec.update(free_type=pr["free_type"], free_box=fbox, free_iou=round(free_iou, 4),
                   free_correct=bool((pr["free_type"] != "box") if it["n_gt"] == 0 else free_iou >= 0.5),
                   consistent=TR.consistent(pr), format_ok=pr["format_ok"], n_cand=pr["n_cand"], n_cond=len(pr["conditions"]),
                   p_null=None, p_box=None)
        if it["n_gt"] > 0 and it["gt_boxes"]:
            fn = TR.false_no_cells(pr, it["gt_boxes"][0])
            rec["false_no"] = list(fn) if fn else None
        if it.get("flipped_clause"):
            rec["accused"] = TR.evidence(pr, it["flipped_clause"])
            rec["accused_seen"] = TR.evidence(pr, it["flipped_clause"], need_seen=True)
    else:
        dec = _decision(seq, scores, processor.tokenizer, null_ids, box_ids)
        parsed = P.parse_qwen3vl(raw, convention=P.RELATIVE_1000, original_wh=wh, sent_wh=wh)
        box = list(parsed.box_xyxy_px) if parsed.box_xyxy_px else None
        otype = parsed.output_type
        rec.update(p_null=dec["p_null"], p_box=dec["p_box"])
    best_iou = max((P.iou(tuple(box), tuple(g)) for g in it["gt_boxes"]), default=0.0) if box else 0.0
    correct = (otype != "box") if it["n_gt"] == 0 else best_iou >= 0.5
    rec.update(output_type=otype, box=box, iou=round(best_iou, 4), correct=bool(correct), raw=raw[: max(400, max_new * 6)])
    ci = raw.find("<coa>")
    if ci >= 0:  # the audits of the propose -> audit protocol, kept whole (the raw text above is truncated)
        try:
            rec["coa"] = json.loads(raw[ci + 5: raw.rfind("</coa>")])
        except Exception:  # noqa: BLE001
            rec["coa_error"] = True
    for k in ("dimension", "size_bin", "kind", "set", "group"):
        if k in it:
            rec[k] = it[k]
    return rec


def run_set(model, processor, template, null_ids, box_ids, items: list[dict], out_path: Path, limit: int | None,
            max_new: int = 64, batch: int = 1, trace: str | None = None, want_cell_probs: bool = False, hint_crop: bool = False, n_crops: int = 1, turns: int = 1) -> None:
    done = done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    if limit:
        todo = todo[: max(0, limit - len(done))]
    print(f"  {out_path.stem}: {len(items)} items, {len(done)} done, {len(todo)} to run", flush=True)
    t0 = time.time()
    n = 0
    tok = processor.tokenizer
    groups = {"no": _first_ids(tok, [" no", "no", " No", "No"]), "yes": _first_ids(tok, [" yes", "yes", " Yes", "Yes"]),
              "unclear": _first_ids(tok, [" unclear", "unclear", " Unclear", "Unclear"])} if (trace and want_cell_probs) else None
    for bi in range(0, len(todo), batch):
        chunk = todo[bi: bi + batch]
        views_of = text_of = None
        if turns == 2:
            raws, seqs, scores = generate_two_turn(model, processor, template, chunk, max_new)
        else:
            if hint_crop:
                views_of, text_of = hint_stage(model, processor, template, chunk, n_crops=n_crops)
            raws, seqs, scores = generate_batch(model, processor, template, chunk, max_new, want_scores=(trace is None) or want_cell_probs,
                                                views_of=views_of, text_of=text_of)
        with open(out_path, "a", encoding="utf-8") as fh:
            for it, raw, seq, sc in zip(chunk, raws, seqs, scores):
                fh.write(json.dumps(score_item(it, raw, seq, sc, processor, null_ids, box_ids, trace, max_new, groups)) + "\n")
        n += len(chunk)
        if (bi // batch) % max(1, 40 // batch) == 0:
            torch.cuda.empty_cache()
            rate = (time.time() - t0) / n
            print(f"    [{n}/{len(todo)}] {rate:.2f}s/item, {(len(todo) - n) * rate / 60:.0f} min left", flush=True)


def summarize(tag_dir: Path) -> dict:
    def rows(name):
        f = tag_dir / f"{name}.jsonl"
        return [json.loads(l) for l in open(f, encoding="utf-8")] if f.is_file() else []

    def acc(rs, key="correct"):
        return round(sum(bool(r.get(key)) for r in rs) / len(rs), 4) if rs else None

    def frac(rs, pred):
        return round(sum(pred(r) for r in rs) / len(rs), 4) if rs else None

    def pn_stats(rs):
        ps = [r["p_null"] for r in rs if r.get("p_null") is not None]
        if not ps:
            return {}
        return {"p_null_sum": round(sum(ps), 6), "p_null_median_log10": round(math.log10(max(statistics.median(ps), 1e-30)), 2),
                "p_null_frac_ge_0.1": round(sum(p >= 0.1 for p in ps) / len(ps), 4)}

    def trace_stats(rs):
        if not rs or "format_ok" not in rs[0]:
            return {}
        fn = [r["false_no"] for r in rs if r.get("false_no")]
        return {"format_ok_rate": frac(rs, lambda r: bool(r.get("format_ok"))), "consistent_rate": frac(rs, lambda r: bool(r.get("consistent"))),
                "mean_n_cand": round(sum(r.get("n_cand", 0) for r in rs) / len(rs), 2),
                "false_no_cell_rate": round(sum(k for k, _ in fn) / max(1, sum(j for _, j in fn)), 4) if fn else None,
                "false_no_item_rate": round(sum(k > 0 for k, _ in fn) / len(fn), 4) if fn else None}

    s: dict = {}
    g = rows("gme")
    if g:
        rej = [r for r in g if r["n_gt"] == 0]
        pos = [r for r in g if r["n_gt"] > 0]
        pos_null = frac(pos, lambda r: r["output_type"] in D.NULL_TYPES)
        s["gme"] = {"n": len(g), "rejection_acc": acc(rej), "rejection_n": len(rej), "positive_acc": acc(pos), "positive_n": len(pos),
                    "positive_null_rate": pos_null, "positive_null_n": sum(r["output_type"] in D.NULL_TYPES for r in pos),
                    "net_rejection": round(acc(rej) - pos_null, 4) if rej and pos else None,
                    "by_dimension": {d: acc([r for r in g if r.get("dimension") == d]) for d in sorted({r.get("dimension") for r in g})},
                    "rejection_p": pn_stats(rej), "positive_p": pn_stats(pos)}
        if "format_ok" in g[0]:
            s["gme"].update(free_rejection_acc=acc(rej, "free_correct"), free_positive_acc=acc(pos, "free_correct"),
                            free_positive_null_rate=frac(pos, lambda r: r.get("free_type") == "null"),
                            rejection_trace=trace_stats(rej), positive_trace=trace_stats(pos))
    gg = rows("gmegray")
    if gg:
        rej = [r for r in gg if r["n_gt"] == 0]
        pos = [r for r in gg if r["n_gt"] > 0]
        s["gmegray"] = {"n": len(gg), "rejection_null_rate": frac(rej, lambda r: r["output_type"] != "box"),
                        "positive_null_rate": frac(pos, lambda r: r["output_type"] != "box"),
                        "rejection_p": pn_stats(rej), "positive_p": pn_stats(pos)}
    for name in ("own", "gray"):
        o = rows(name)
        if o:
            neg = [r for r in o if r.get("kind") == "negative"]
            pos = [r for r in o if r.get("kind") != "negative"]
            s[name] = {"n": len(o), "neg_null_rate": frac(neg, lambda r: r["output_type"] != "box"),
                       "pos_acc": acc(pos), "pos_null_rate": frac(pos, lambda r: r["output_type"] in D.NULL_TYPES),
                       "neg_p": pn_stats(neg), "pos_p": pn_stats(pos)}
            if o and "format_ok" in o[0]:
                by_group: dict[str, dict] = {}
                for r in o:
                    by_group.setdefault(r.get("group"), {})[r.get("kind")] = r
                pairs = [gp for gp in by_group.values() if "positive" in gp and "negative" in gp]
                acc_rows = [r for r in neg if r.get("output_type") in D.NULL_TYPES and r.get("accused") is not None]
                s[name].update(pair_n=len(pairs),
                               pair_acc=round(sum(gp["positive"]["correct"] and gp["negative"]["correct"] for gp in pairs) / len(pairs), 4) if pairs else None,
                               accusation_acc=frac(acc_rows, lambda r: bool(r["accused"])), accusation_n=len(acc_rows),
                               neg_trace=trace_stats(neg), pos_trace=trace_stats(pos))
    r = rows("refcoco")
    if r:
        s["refcoco"] = {sub: {"n": len([x for x in r if x.get("set") == sub]), "acc": acc([x for x in r if x.get("set") == sub]),
                              "null_rate": frac([x for x in r if x.get("set") == sub], lambda x: x["output_type"] in D.NULL_TYPES)}
                        for sub in sorted({x.get("set") for x in r})}
        if "format_ok" in r[0]:
            s["refcoco"]["trace"] = trace_stats(r)
    (tag_dir / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="4b", choices=list(D.MODELS))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--sets", default="gme,gmegray,own,gray,refcoco")
    ap.add_argument("--gme-pos-n", type=int, default=None, help="screening: sample this many positives (all Rejection kept)")
    ap.add_argument("--gme-gray-pos-n", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-val-scenes", type=int, default=18)
    ap.add_argument("--refcoco-n", type=int, default=300)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--prompt", default=None, help="prompt template name (default: primary, or the trace prompt with --trace)")
    ap.add_argument("--trace", default=None, choices=list(TRACE_PROMPTS), help="verification-trace protocol")
    ap.add_argument("--max-new", type=int, default=None, help="default 64, or 384 with --trace")
    ap.add_argument("--max-new1", type=int, default=400, help="turn-1 token budget under --turns 2 (conditions + tool call)")
    ap.add_argument("--isolate", action="store_true", help="--turns 2: verify each candidate in its own turn-2 call (one close-up), rows assembled afterwards")
    ap.add_argument("--samples", type=int, default=1, help="--turns 2: sample turn 2 k times and decide every verdict cell by majority vote")
    ap.add_argument("--audit", default=None, choices=[None, "coa"], help="--turns 2: per-candidate COA audit head instead of the joint table")
    ap.add_argument("--audit-base", action="store_true", help="run the audit turn with the adapter disabled (zero-training probe of the head)")
    ap.add_argument("--sample-temp", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-pixels", type=int, default=None, help="override the P21 pixel cap (resolution check)")
    ap.add_argument("--cell-probs", action="store_true", help="trace mode: record p(no) at every verdict cell (soft decisions, train.soft_decision)")
    ap.add_argument("--hint-crop", action="store_true", help="trace mode: self-hint crop (base box via the primary prompt, its close-up as a second image)")
    ap.add_argument("--hint-max-pixels", type=int, default=None, help="pixel cap for the hint stage only (e.g. 4915200)")
    ap.add_argument("--n-crops", type=int, default=1, help="close-ups per prompt under --hint-crop (>1: base zero-shot trace candidates add crops)")
    ap.add_argument("--turns", type=int, default=1, choices=[1, 2], help="2 = the v2 protocol (tool-call zoom on the model's own candidates)")
    ap.add_argument("--gme-rej-n", type=int, default=None, help="screening: sample this many Rejection items (default all 201)")
    args = ap.parse_args()
    tag_dir = D.TRAIN_ROOT / "eval" / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    max_new = args.max_new or (384 if args.trace else 64)
    global MAX_NEW1, ISOLATE, SAMPLES, SAMPLE_TEMP, AUDIT, AUDIT_BASE
    MAX_NEW1 = args.max_new1
    ISOLATE = bool(args.isolate)
    AUDIT, AUDIT_BASE = args.audit, bool(args.audit_base)
    SAMPLES, SAMPLE_TEMP = args.samples, args.sample_temp
    prompt_name = args.prompt or ("grounding_verify_trace_tool" if args.turns == 2 else (("grounding_verify_trace_crops" if args.n_crops > 1 else "grounding_verify_trace_crop") if args.hint_crop else (TRACE_PROMPTS[args.trace] if args.trace else "grounding_qwen3vl_primary")))
    if not args.summary_only:
        template = prompts.load(prompt_name, non_kill=prompt_name != "grounding_qwen3vl_primary")
        model, processor = load_model(args.model, args.adapter, args.max_pixels)
        if args.hint_max_pixels:
            global HINT_PROCESSOR
            HINT_PROCESSOR = T.load_capped_processor(*D.MODELS[args.model], max_pixels=args.hint_max_pixels)
        tok = processor.tokenizer
        null_ids, box_ids = _first_ids(tok, [" null", "null", " Null"]), _first_ids(tok, [" [", "["])
        (tag_dir / "model.json").write_text(json.dumps({"model": D.MODELS[args.model], "adapter": args.adapter, "prompt": prompt_name,
                                                        "trace": args.trace, "max_new": max_new, "batch": args.batch, "max_pixels": args.max_pixels}), encoding="utf-8")
        for name in [s.strip() for s in args.sets.split(",") if s.strip()]:
            run_set(model, processor, template, null_ids, box_ids, items_for(name, args), tag_dir / f"{name}.jsonl", args.limit,
                    max_new, args.batch, args.trace, args.cell_probs, args.hint_crop, args.n_crops, args.turns)
    s = summarize(tag_dir)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
