"""GRPO rollouts and rewards for the propose -> observe/judge -> answer protocol (train.coa head).

A rollout = turn 1 (conditions + tool call, <= k_max boxes) + one isolated audit per candidate (outlined
scene + close-up) + a text-only answer step.  Every reward term is computed from labels and rules:
  format      +0.1 parseable; -1 when turn 1 names no box or the answer step is unparsable;
              an unparsable audit counts as "no evidence" (the candidate fits) and costs -0.2 once
  consistency the model's answer must equal the answer the harness derives from its own audits, else
              the outcome terms are 0 (the model must reason from its evidence, not around it)
  outcome     positive: box IoU >= 0.5 -> +1, wrong box 0, null 0
              negative with a known described instance: null -> +0.5 only when the audit of the candidate
              matching that instance names a mismatch on the falsified clause whose seen value contains the
              original value (+0.5 more as the accusation term, credited to that audit segment); null with
              a mismatch elsewhere 0.2; box 0.  Negatives without labels (cross-scene): null -> +0.5, box 0
  accusation  -0.5 on the audit segment of the candidate matching a positive's ground-truth box when that
              audit names a mismatch (a false accusation), once per rollout
The proposer is credited only through the outcome.
"""

from __future__ import annotations

import json
import re

import torch

from datagen import common as C
from shared.harness import parsers as P
from train import coa as COA
from train import data as D
from train import traces as TR

_COND = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$", re.M)


def conditions_of(t1: str) -> list[str]:
    block = t1.split("</conditions>")[0]
    return [m.group(2) for m in _COND.finditer(block)]


def rewrite_t1(t1: str, rel_box) -> str:
    a, z = t1.find("<tool_call>"), t1.find("</tool_call>")
    call = "<tool_call>" + json.dumps({"name": "image_zoom_in", "arguments": {"boxes": [rel_box]}}) + "</tool_call>"
    return (t1[:a] + call + t1[z + len("</tool_call>"):]) if 0 <= a < z else (t1 + "\n" + call)


def audit_conv(processor, template, coa_tmpl, item: dict, image, t1: str, box_px, wh) -> tuple[str, list]:
    """Chat text and the two images of one audit call (unpadded inputs are built by the caller)."""
    u1 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": template.render(expr=item["expression"])}]}
    a1 = {"role": "assistant", "content": [{"type": "text", "text": rewrite_t1(t1, D.relative_1000(box_px, wh))}]}
    u2 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": coa_tmpl.render()}]}
    return processor.apply_chat_template([u1, a1, u2], tokenize=False, add_generation_prompt=True), [C.outline(image, box_px, "red"), TR.crop_view(image, box_px)]


def answer_conv(processor, ans_tmpl, item: dict, t1: str, audits_raw: list[str], boxes_px: list, wh) -> str:
    body = ans_tmpl.render(expr=item["expression"], turn1=t1, audits=COA.audits_text(audits_raw, [D.relative_1000(b, wh) for b in boxes_px]))
    return processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": body}]}], tokenize=False, add_generation_prompt=True)


def _gen_batched(model, processor, convs: list[str], images_per: list[list], max_new: int, temperature: float, eos: set, sub_batch: int, tok, sample: bool = True):
    """Left-padded batched generation; returns trimmed completion token lists in order."""
    from train.grpo_lora import trim

    out_tokens: list[list[int]] = []
    processor.tokenizer.padding_side = "left"
    for s0 in range(0, len(convs), sub_batch):
        tx = convs[s0:s0 + sub_batch]
        imgs = [im for lst in images_per[s0:s0 + sub_batch] for im in lst]
        kw = {"images": imgs} if imgs else {}
        inputs = processor(text=tx, padding=True, return_tensors="pt", **kw).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=sample, temperature=temperature if sample else None, top_p=1.0 if sample else None,
                                 top_k=0 if sample else None)
        n_prompt = inputs["input_ids"].shape[1]
        for j in range(len(tx)):
            out_tokens.append(trim(gen[j][n_prompt:].tolist(), eos, tok.pad_token_id))
    return out_tokens


@torch.no_grad()
def sample_coa(model, processor, template, coa_tmpl, ans_tmpl, item: dict, image, inputs1, args, eos: set, tok):
    """G rollouts of the propose -> audit -> answer protocol.  Returns (segments per rollout, rollouts)."""
    from train.grpo_lora import trim

    wh = tuple(item["image_wh"])
    n1 = inputs1["input_ids"].shape[1]
    gen1 = model.generate(**inputs1, max_new_tokens=args.max_new1, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                          num_return_sequences=args.group, stop_strings=["</tool_call>"], tokenizer=tok)
    comps1 = [trim(seq[n1:].tolist(), eos, tok.pad_token_id) for seq in gen1]
    raws1 = [processor.decode(c, skip_special_tokens=True) for c in comps1]
    boxes_per = [TR.parse_tool_boxes(r, wh, max_boxes=args.k_max) for r in raws1]
    # audits, all rollouts x candidates in one batched pass
    convs, imgs, owners = [], [], []
    for g, (r1, boxes) in enumerate(zip(raws1, boxes_per)):
        for k, b in enumerate(boxes):
            text, ims = audit_conv(processor, template, coa_tmpl, item, image, r1, b, wh)
            convs.append(text)
            imgs.append(ims)
            owners.append((g, k))
    aud_tokens = _gen_batched(model, processor, convs, imgs, args.max_new, args.temperature, eos, args.sample_batch, tok) if convs else []
    aud_text = {o: processor.decode(t, skip_special_tokens=True) for o, t in zip(owners, aud_tokens)}
    aud_tok = {o: t for o, t in zip(owners, aud_tokens)}
    # answer step, one text-only call per rollout that has candidates
    a_convs, a_owner = [], []
    for g, (r1, boxes) in enumerate(zip(raws1, boxes_per)):
        if boxes:
            a_convs.append(answer_conv(processor, ans_tmpl, item, r1, [aud_text[(g, k)] for k in range(len(boxes))], boxes, wh))
            a_owner.append(g)
    ans_tokens = _gen_batched(model, processor, a_convs, [[] for _ in a_convs], 48, args.temperature, eos, 8, tok) if a_convs else []
    ans_tok = {g: t for g, t in zip(a_owner, ans_tokens)}
    rollouts, segs = [], []
    for g, (c1, r1, boxes) in enumerate(zip(comps1, raws1, boxes_per)):
        seg = [(inputs1, c1)]
        audits_raw = []
        for k, b in enumerate(boxes):
            text, ims = audit_conv(processor, template, coa_tmpl, item, image, r1, b, wh)
            inp = processor(images=ims, text=text, return_tensors="pt").to(model.device)
            seg.append((inp, aud_tok[(g, k)]))
            audits_raw.append(aud_text[(g, k)])
        ans_text = ""
        if boxes:
            a_text = answer_conv(processor, ans_tmpl, item, r1, audits_raw, boxes, wh)
            inp = processor(text=a_text, return_tensors="pt").to(model.device)
            seg.append((inp, ans_tok[g]))
            ans_text = processor.decode(ans_tok[g], skip_special_tokens=True)
        rollouts.append({"t1": r1, "boxes": boxes, "audits_raw": audits_raw, "answer": ans_text, "n_seg": len(seg)})
        segs.append(seg)
    return segs, rollouts


def parse_answer(text: str, wh) -> tuple[str, list | None]:
    m = re.search(r"<answer>(.*?)</answer>", text or "", re.S)
    if not m:
        return "bad", None
    free = P.parse_qwen3vl(m.group(1), convention=P.RELATIVE_1000, original_wh=tuple(wh), sent_wh=tuple(wh))
    if free.output_type == "box" and free.box_xyxy_px:
        return "box", list(free.box_xyxy_px)
    if free.output_type in D.NULL_TYPES:
        return "null", None
    return "bad", None


def coa_reward(ro: dict, item: dict, args) -> tuple[float, float, str, dict]:
    """(r_base, r_acc, answer type, info).  r_acc is credited to the audit segment of the labelled candidate
    (info['acc_seg']); r_base to every segment."""
    wh = tuple(item["image_wh"])
    boxes = ro["boxes"]
    info: dict = {"k": len(boxes)}
    if not boxes:
        return -1.0, 0.0, "bad", info
    audits = [COA.parse_audit(t) for t in ro["audits_raw"]]
    n_unparsable = sum(1 for a in audits if not a["format_ok"])
    info["unparsable"] = n_unparsable
    r = 0.1 - (0.2 if n_unparsable else 0.0)
    for a in audits:  # an unparsable audit is "no evidence": the candidate fits
        if not a["format_ok"]:
            a["lines"] = []
            a["format_ok"] = True
    derived_box, _, fits = COA.decide(audits, boxes)
    a_type, a_box = parse_answer(ro["answer"], wh)
    if a_type == "bad":
        return -1.0, 0.0, "bad", info
    # consistent = the answer follows from the audits: null iff no candidate fits; a box must be one of the fitting candidates
    # (the model may choose among several fitting candidates; the rule's first-fitting box is only the diagnostic)
    consistent = (a_type == "null" and not any(fits)) or (a_type == "box" and any(f and P.iou(tuple(a_box), tuple(b)) >= 0.9 for f, b in zip(fits, boxes)))
    info["consistent"] = consistent
    lines_all = [ln for a in audits for ln in a["lines"]]
    info["unsure_frac"] = (sum(ln["verdict"] == "unsure" for ln in lines_all) / len(lines_all)) if lines_all else None
    info["echo_frac"] = (sum(COA.norm(ln["seen"]) == COA.norm(ln["claimed"]) for ln in lines_all) / len(lines_all)) if lines_all else None
    gt = item["answer"]["bbox_2d"]
    m = item.get("matrix") or {}
    r_acc = 0.0
    if gt is not None:  # positive (own scenes or RefCOCO answer-only)
        info["cand_recall"] = any(P.iou(tuple(b), tuple(gt)) >= 0.5 for b in boxes)
        k_gt = next((k for k, b in enumerate(boxes) if P.iou(tuple(b), tuple(gt)) >= 0.5), None)
        if k_gt is not None:
            info["acc_seg"] = 1 + k_gt
            if COA.named_mismatches(audits[k_gt]["lines"]):
                r_acc -= float(getattr(args, "r_false_acc", 0.0))  # default 0: losing the +1 already prices a false accusation
                info["false_accusation"] = True
        if consistent and a_type == "box" and P.iou(tuple(a_box), tuple(gt)) >= 0.5:
            r += 1.0
        return r, r_acc, a_type, info
    # negative
    intended = None
    if m.get("first_iid") is not None and m.get("flipped_clause"):
        intended = next((row for row in m["rows"] if row["iid"] == m["first_iid"]), None)
    if intended is None:  # cross-scene / unlabelled negative: null credit capped
        if consistent and a_type == "null":
            r += 0.5
        return r, r_acc, a_type, info
    k_int = next((k for k, b in enumerate(boxes) if P.iou(tuple(b), tuple(intended["box"])) >= 0.5), None)
    info["cand_recall"] = k_int is not None
    correct_acc = False
    if k_int is not None:
        info["acc_seg"] = 1 + k_int
        conds = conditions_of(ro["t1"])
        j = TR.align(conds, m["flipped_clause"]) if conds else None
        from_words = {w for w in COA.norm(m.get("flip_from") or "").split() if w not in COA.NEG}
        for ln in COA.named_mismatches(audits[k_int]["lines"]):
            if j is not None and ln["idx"] == j and (not from_words or (from_words & set(COA.norm(ln["seen"]).split()))):
                correct_acc = True
                break
    info["accusation_correct"] = correct_acc
    accused_intended = k_int is not None and bool(COA.named_mismatches(audits[k_int]["lines"]))
    if consistent and a_type == "null":
        if accused_intended:
            r += 0.5
            if correct_acc:
                r_acc += 0.5
        elif any(COA.named_mismatches(a["lines"]) for a in audits):
            r += 0.2
    return r, r_acc, a_type, info


def label_rollout(processor, template, coa_tmpl, ans_tmpl, item: dict, image, inputs1, args, tok, im_end: int, device):
    """Teacher-forced label rollout for injection: turn 1 from the label table, one audit per row, the label answer."""
    from train.sft_lora import _rationale_of

    m = item["matrix"]
    wh = tuple(item["image_wh"])
    t1, boxes, _ = TR.render_turns(m, item["image_wh"], args.trace, args.order)
    boxes = boxes[: args.k_max]
    rows = TR.ordered_rows(m, args.order)[: args.k_max]
    rof = _rationale_of(m)
    audits_raw = [COA.render_audit(m, r, rof) for r in rows]
    ans_rel = next((D.relative_1000(r["box"], item["image_wh"]) for r in rows if r["iid"] == m.get("answer_iid")), None)
    ans_text = "<answer>" + json.dumps({"bbox_2d": ans_rel}) + "</answer>"
    seg = [(inputs1, tok(t1, add_special_tokens=False)["input_ids"] + [im_end])]
    for b, a in zip(boxes, audits_raw):
        text, ims = audit_conv(processor, template, coa_tmpl, item, image, t1, b, wh)
        seg.append((processor(images=ims, text=text, return_tensors="pt").to(device), tok(a, add_special_tokens=False)["input_ids"] + [im_end]))
    a_text = answer_conv(processor, ans_tmpl, item, t1, audits_raw, boxes, wh)
    seg.append((processor(text=a_text, return_tensors="pt").to(device), tok(ans_text, add_special_tokens=False)["input_ids"] + [im_end]))
    ro = {"t1": t1, "boxes": boxes, "audits_raw": audits_raw, "answer": ans_text, "n_seg": len(seg)}
    return seg, ro
