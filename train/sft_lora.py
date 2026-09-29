"""Stage A of the PoC: LoRA SFT of Qwen3-VL on the exported triplets, with the unlock curve.

    python -m train.sft_lora --name sft_v1 [--model 4b] [--epochs 3] [--lr 1e-4] [--r 16]
                             [--grad-accum 8] [--eval-every 20] [--drop-4b-refused]
                             [--decision-weight 1] [--neg-repeat 1] [--extra a.jsonl,b.jsonl]
                             [--init-adapter <dir>]

--decision-weight W multiplies the loss on the decision token (the first token that differs
between `null` and `[`); --neg-repeat k repeats every negative k times per epoch; --extra adds
item files in the export format (e.g. cross-scene negatives); --init-adapter warm-starts.

Loss only on the assistant answer (the fenced JSON).  Every --eval-every optimizer steps the
held-out scenes are scored with greedy generation and the decision-token probabilities
(scripts.pilot_abstain_signal._decision): null rate on negatives, IoU>=0.5 on positives,
median p(null).  Output under $VLMG_DATA_ROOT/train/<name>/: adapter/, curve.jsonl,
split.json, args.json.
"""

from __future__ import annotations

import argparse
import json
import re
import math
import random
import statistics
import time
from pathlib import Path

import torch

from scripts.pilot_abstain_signal import _decision, _first_ids
from shared.harness import parsers as P
from shared.harness import prompts
from shared.harness import tokens as T
from train import data as D
from train import traces as TR
from train.eval_suite import TRACE_PROMPTS, generate_batch
import train.eval_suite as E
from train import coa as COA

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def target_of(item: dict) -> str:
    """The assistant text to learn: the rendered verification trace when the item carries one,
    else the fenced JSON answer."""
    return item["trace"] if item.get("trace") else D.target_text(item)


def build_example(processor, template, item: dict, image, decision_ids: set[int] | None = None, decision_weight: float = 1.0,
                  target_text: str | None = None, views: list | None = None, fields: dict | None = None):
    """Token ids for prompt+answer, with labels masked over the prompt; per-token loss weights.
    `views` (default [image]) are the prompt images in order; `fields` extra prompt fields."""
    views = views or [image]
    user = {"role": "user", "content": [{"type": "image"} for _ in views] + [{"type": "text", "text": template.render(expr=item["expression"], **(fields or {}))}]}
    prompt_text = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": target_text or target_of(item)}]}],
        tokenize=False, add_generation_prompt=False)
    assert full_text.startswith(prompt_text), "chat template: answer text is not a suffix of the prompt text"
    full = processor(images=views, text=full_text, return_tensors="pt")
    prompt = processor(images=views, text=prompt_text, return_tensors="pt")
    n_prompt = prompt["input_ids"].shape[1]
    assert torch.equal(full["input_ids"][0, :n_prompt], prompt["input_ids"][0]), "prompt tokens differ between prompt-only and full encodings"
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    full["labels"] = labels
    weights = (labels != -100).float()
    dec_pos = -1
    if decision_ids:
        ids = full["input_ids"][0].tolist()
        for pos in range(n_prompt, len(ids)):
            if ids[pos] in decision_ids:
                dec_pos = pos
                weights[0, pos] = decision_weight
                break
    full["loss_weights"] = weights
    full["decision_pos"] = torch.tensor([dec_pos])
    return full


def _verdict_weights(tok, ids, start: int, w: tuple, target_row: int | None) -> torch.Tensor:
    """Per-token loss weights of the turn-2 completion (tokens from `start`): verdict words after 'k:' get
    w[0] in the target row and w[1] in the other rows; the rationale text after a verdict gets w[2];
    everything else 1.  Token spans come from decoding the labelled tokens one by one."""
    w_t, w_o, w_r = w
    pieces = [tok.decode([int(t)]) for t in ids[start:]]
    text, spans = "", []
    for p in pieces:
        spans.append((len(text), len(text) + len(p)))
        text += p
    weights = [1.0] * len(pieces)

    def tokens_in(a, b):
        return [i for i, (s, e) in enumerate(spans) if e > a and s < b]

    for row, om in enumerate(re.finditer(r"<object[^>]*>(.*?)</object>", text, re.S), 1):
        base = om.start(1)
        for cm in re.finditer(r"(\d+)\s*:\s*(yes|no|unclear)\b([^;]*?)(?=;|->|$)", om.group(1), re.S):
            for i in tokens_in(base + cm.start(2), base + cm.end(2)):
                weights[i] = w_t if row == target_row else w_o
            if cm.end(3) > cm.start(3):
                for i in tokens_in(base + cm.start(3), base + cm.end(3)):
                    weights[i] = w_r
    return torch.tensor(weights)


def build_example_turns(processor, template, item: dict, image, crops: list, t1: str, t2: str, weights: tuple | None = None, target_row: int | None = None):
    """Two-turn label conversation (v2 protocol): user(image, prompt) -> assistant(t1: conditions +
    zoom tool call) -> user(crops, continue) -> assistant(t2: table + answer).  Labels cover both
    assistant turns; the crops' tokens are masked like any user turn."""
    u1 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": template.render(expr=item["expression"])}]}
    a1 = {"role": "assistant", "content": [{"type": "text", "text": t1}]}
    u2 = {"role": "user", "content": [{"type": "image"} for _ in crops] + [{"type": "text", "text": TR.turn2_text_for(len(crops))}]}
    a2 = {"role": "assistant", "content": [{"type": "text", "text": t2}]}
    tmpl = lambda msgs, gen: processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    p1, f1, p2, f = tmpl([u1], True), tmpl([u1, a1], False), tmpl([u1, a1, u2], True), tmpl([u1, a1, u2, a2], False)
    assert f1.startswith(p1) and p2.startswith(f1) and f.startswith(p2), "chat template: turns are not prefix-consistent"
    n_p1 = processor(images=[image], text=p1, return_tensors="pt")["input_ids"].shape[1]
    n_f1 = processor(images=[image], text=f1, return_tensors="pt")["input_ids"].shape[1]
    views = [image] + list(crops)
    n_p2 = processor(images=views, text=p2, return_tensors="pt")["input_ids"].shape[1]
    full = processor(images=views, text=f, return_tensors="pt")
    labels = full["input_ids"].clone()
    labels[:, :] = -100
    labels[:, n_p1:n_f1] = full["input_ids"][:, n_p1:n_f1]
    labels[:, n_p2:] = full["input_ids"][:, n_p2:]
    full["labels"] = labels
    full["loss_weights"] = (labels != -100).float()
    if weights:
        full["loss_weights"][0, n_p2:] = _verdict_weights(processor.tokenizer, full["input_ids"][0], n_p2, weights, target_row)
    full["decision_pos"] = torch.tensor([-1])
    return full


def build_example_turn1(processor, template, item: dict, image, t1: str):
    """Turn-1 supervision alone (conditions + the full zoom tool call): user(image, prompt) -> assistant(t1)."""
    u1 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": template.render(expr=item["expression"])}]}
    a1 = {"role": "assistant", "content": [{"type": "text", "text": t1}]}
    p1 = processor.apply_chat_template([u1], tokenize=False, add_generation_prompt=True)
    f = processor.apply_chat_template([u1, a1], tokenize=False, add_generation_prompt=False)
    n_p1 = processor(images=[image], text=p1, return_tensors="pt")["input_ids"].shape[1]
    full = processor(images=[image], text=f, return_tensors="pt")
    labels = full["input_ids"].clone()
    labels[:, :n_p1] = -100
    full["labels"] = labels
    full["loss_weights"] = (labels != -100).float()
    full["decision_pos"] = torch.tensor([-1])
    return full


def _rationale_of(m: dict):
    def f(row: dict, j: int):
        v = row["verdicts"][j]
        cond = m["conditions"][j]
        info = TR.rationales().get(tuple(row.get("origin", ())), {}).get(cond)
        if info and info["blind"] == v and info["label"] == v and info.get("rationale"):
            r = TR._clean_reason(info["rationale"])
            if v != "no" or COA.named_value(r, cond):
                return r
        if v == "no":
            return COA.seen_from_bank(tuple(row.get("origin", ())), cond)  # the instance's own checked value for that attribute
        return None
    return f


def _claim_prefix_weights(tok, ids, start: int, w_claim: float) -> torch.Tensor:
    """Down-weight the copied claim segment of every audit line (from the line number to ' | seen:')."""
    pieces = [tok.decode([int(t)]) for t in ids[start:]]
    text, spans = "", []
    for p in pieces:
        spans.append((len(text), len(text) + len(p)))
        text += p
    weights = [1.0] * len(pieces)
    for m in re.finditer(r"^\s*\d+\.\s*(.*?)\s*\|\s*seen:", text, re.M):
        a, b = m.start(1), m.end(1)
        for i, (s0, e0) in enumerate(spans):
            if e0 > a and s0 < b:
                weights[i] = w_claim
    # a seen value that merely repeats the claim (no observed value was available for that cell) must not be taught: it is the
    # parroting we penalise in RL; its span gets weight 0 (the verdict token after it still counts)
    for m in re.finditer(r"^\s*\d+\.\s*(.*?)\s*\|\s*seen:\s*(.*?)\s*\|", text, re.M):
        if COA.norm(m.group(1)) == COA.norm(m.group(2)):
            a, b = m.start(2), m.end(2)
            for i, (s0, e0) in enumerate(spans):
                if e0 > a and s0 < b:
                    weights[i] = 0.0
    return torch.tensor(weights)


def build_example_answer(processor, ans_tmpl, item: dict, t1: str, audits_raw: list[str], boxes_rel: list, answer_text: str):
    """The answer step (text only): user(statement, turn-1 text, the K audit blocks) -> assistant(<answer>...)."""
    body = ans_tmpl.render(expr=item["expression"], turn1=t1, audits=COA.audits_text(audits_raw, boxes_rel))
    u = {"role": "user", "content": [{"type": "text", "text": body}]}
    a = {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
    p = processor.apply_chat_template([u], tokenize=False, add_generation_prompt=True)
    f = processor.apply_chat_template([u, a], tokenize=False, add_generation_prompt=False)
    n_p = processor(text=p, return_tensors="pt")["input_ids"].shape[1]
    full = processor(text=f, return_tensors="pt")
    labels = full["input_ids"].clone()
    labels[:, :n_p] = -100
    full["labels"] = labels
    full["loss_weights"] = (labels != -100).float()
    full["decision_pos"] = torch.tensor([-1])
    return full


def build_example_coa(processor, template, coa_tmpl, item: dict, image, m: dict, t1: str, boxes_px: list, k: int, order: str):
    """One isolated COA audit example for candidate k (1-based) of the label table: user(image, prompt) ->
    assistant(turn 1 rewritten to this candidate's box, unlabelled) -> user(outlined scene, close-up, audit prompt)
    -> assistant(audit block, labelled).  Mirrors eval_suite._coa_turn2."""
    from datagen import common as C

    rows = TR.ordered_rows(m, order)
    row, box = rows[k - 1], boxes_px[k - 1]
    rel = D.relative_1000(box, item["image_wh"])
    a, z = t1.find("<tool_call>"), t1.find("</tool_call>")
    call = "<tool_call>" + json.dumps({"name": "image_zoom_in", "arguments": {"boxes": [rel]}}) + "</tool_call>"
    t1_k = (t1[:a] + call + t1[z + len("</tool_call>"):]) if 0 <= a < z else (t1 + "\n" + call)
    audit = COA.render_audit(m, row, _rationale_of(m))
    u1 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": template.render(expr=item["expression"])}]}
    a1 = {"role": "assistant", "content": [{"type": "text", "text": t1_k}]}
    u2 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": coa_tmpl.render()}]}
    a2 = {"role": "assistant", "content": [{"type": "text", "text": audit}]}
    views = [C.outline(image, box, "red"), TR.crop_view(image, box)]  # two images, same as eval_suite._coa_turn2
    p2 = processor.apply_chat_template([u1, a1, u2], tokenize=False, add_generation_prompt=True)
    f = processor.apply_chat_template([u1, a1, u2, a2], tokenize=False, add_generation_prompt=False)
    assert f.startswith(p2), "chat template: turns are not prefix-consistent"
    n_p2 = processor(images=views, text=p2, return_tensors="pt")["input_ids"].shape[1]
    full = processor(images=views, text=f, return_tensors="pt")
    labels = full["input_ids"].clone()
    labels[:, :n_p2] = -100
    full["labels"] = labels
    full["loss_weights"] = (labels != -100).float()
    if CLAIM_WEIGHT < 1.0:
        full["loss_weights"][0, n_p2:] = _claim_prefix_weights(processor.tokenizer, full["input_ids"][0], n_p2, CLAIM_WEIGHT)
    full["decision_pos"] = torch.tensor([-1])
    return full


CLAIM_WEIGHT = 0.2  # loss weight of the copied claim segment in audit lines (the model still learns the format, mostly learns 'seen' and verdicts)


def expand_coa(items: list[dict], order: str) -> list[dict]:
    """Per item: one turn-1 example (_coa_k = 0), one audit example per label row (capped like the table) and the
    answer step (_coa_k = -1).  Items whose rendered audits do not reproduce the label answer under the harness rule
    are dropped (the labels must be consistent with the rule the model is trained to follow)."""
    out, dropped = [], 0
    for it in items:
        m = it.get("matrix")
        if not m and it.get("answer", {}).get("bbox_2d") is not None and it.get("kind") == "positive" and it.get("expression"):
            # answer-only positives (RefCOCO train): a one-clause label table so turn 1 (decomposition + box) and the audit
            # of the true box are supervised too; without this the decomposition of short expressions is untrained
            gt = it["answer"]["bbox_2d"]
            m = {"conditions": [it["expression"]], "rows": [{"iid": 0, "box": gt, "verdicts": ["yes"], "seen": {}, "origin": ()}],
                 "answer_iid": 0, "flipped_idx": None, "flipped_clause": None, "first_iid": 0, "flip_from": None, "flip_to": None}
            it = {**it, "matrix": m, "source": "answer_only"}
        if not m:
            out.append(it)
            continue
        rows = TR.ordered_rows(m, order)
        rof = _rationale_of(m)
        audits = [COA.parse_audit(COA.render_audit(m, r, rof)) for r in rows]
        if it.get("source") == "audit":  # recomposed single-row audit targets: the row must be rejected by the rule
            if audits[0]["format_ok"] and COA.named_mismatches(audits[0]["lines"]):
                out.append({**it, "_coa_k": 1})
            else:
                dropped += 1
            continue
        ans, _, _ = COA.decide(audits, [r["box"] for r in rows])
        label_box = next((r["box"] for r in rows if r["iid"] == m.get("answer_iid")), None) if m.get("answer_iid") is not None else None
        if not ((ans is None and label_box is None) or (ans is not None and ans == label_box)):
            dropped += 1
            continue
        if it.get("source") != "answer_only":  # one-box tables (RefCOCO) must not teach turn 1 to propose a single box (recall 74 -> 55)
            out.append({**it, "_coa_k": 0})
        out.extend({**it, "_coa_k": k} for k in range(1, len(rows) + 1))
        out.append({**it, "_coa_k": -1})
    if dropped:
        print(f"  coa: {dropped} items dropped (label audits inconsistent with the rule)", flush=True)
    if COA_PARTS != {"turn1", "audit", "answer"}:  # --coa-parts: e.g. a proposer-only SFT teaches turn 1 and leaves the audits to RL
        kind = lambda k: "turn1" if k == 0 else ("answer" if k == -1 else "audit")
        out = [it for it in out if "_coa_k" not in it or kind(it["_coa_k"]) in COA_PARTS]
        print(f"  coa: parts {sorted(COA_PARTS)} -> {len(out)} examples", flush=True)
    return out


COA_PARTS = {"turn1", "audit", "answer"}


def null_unlikelihood(logits: torch.Tensor, dec_pos: int, null_ids: list[int]) -> torch.Tensor:
    """-log(1 - p(null)) at the decision position: pushes the null mass down on positives."""
    logp = torch.log_softmax(logits[0, dec_pos - 1].float(), dim=-1)
    p_null = torch.exp(torch.logsumexp(logp[list(null_ids)], dim=0))
    return -torch.log1p(-p_null.clamp(max=1 - 1e-6))


def weighted_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Token cross-entropy weighted per token, normalised by the total weight."""
    logits = logits[:, :-1].float()
    tgt = labels[:, 1:]
    w = weights[:, 1:]
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1).clamp(min=0), reduction="none")
    ce = ce.view(tgt.shape) * (tgt != -100)
    return (ce * w).sum() / w.sum().clamp(min=1.0)


def sparse_loss(model, ex: dict, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Same value as weighted_loss(model(**ex).logits, labels, weights) (weights are zero where
    labels are -100), but the LM head runs only at the positions that predict a labelled token
    (logits_to_keep as an index tensor), so the vocabulary-sized logits of the image and prompt
    tokens never exist.  Two-turn examples with six native-resolution crops spilled VRAM into
    system memory through the full logits (steps of 10-20 min instead of 0.2)."""
    assert labels.shape[0] == 1, "one example at a time"
    idx = (labels[0, 1:] != -100).nonzero(as_tuple=True)[0]  # logits at position t predict the label at t + 1
    logits = model(**ex, logits_to_keep=idx).logits[0].float()
    tgt = labels[0, idx + 1]
    w = weights[0, idx + 1]
    ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
    return (ce * w).sum() / w.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, processor, template, items: list[dict], null_ids, box_ids, gray: bool = False,
             trace: str | None = None, max_new: int | None = None, batch: int = 8, hint_crop: bool = False, n_crops: int = 1, turns: int = 1) -> dict:
    """Held-out scenes: null rate on negatives, IoU>=0.5 on positives; in trace mode the answer
    is the one derived from the verdict table, and format validity is reported."""
    from train.eval_suite import generate_two_turn, hint_stage

    model.eval()
    max_new = max_new or (384 if trace else 64)
    rows = []
    evs = [{"id": it["id"], "expr": it["expression"], "image": it["image"], "image_wh": it["image_wh"], "gray": gray,
            "kind": it["kind"], "gt": it["answer"]["bbox_2d"]} for it in items]
    for bi in range(0, len(evs), batch):
        chunk = evs[bi: bi + batch]
        views_of = text_of = None
        if turns == 2:
            raws, seqs, scores = generate_two_turn(model, processor, template, chunk, max_new)
        else:
            if hint_crop:
                views_of, text_of = hint_stage(model, processor, template, chunk, n_crops=n_crops)
            raws, seqs, scores = generate_batch(model, processor, template, chunk, max_new, want_scores=trace is None, views_of=views_of, text_of=text_of)
        for it, raw, seq, sc in zip(chunk, raws, seqs, scores):
            wh = tuple(it["_wh"])
            gt = it["gt"]
            if trace:
                pr = TR.parse(raw, wh)
                otype = {"box": "box", "null": "none", "invalid": "invalid"}[pr["derived_type"]]
                box = pr["derived_box"]
                p_null = None
                fmt = pr["format_ok"]
            else:
                dec = _decision(seq, sc, processor.tokenizer, null_ids, box_ids)
                parsed = P.parse_qwen3vl(raw, convention=P.RELATIVE_1000, original_wh=wh, sent_wh=wh)
                otype, box, p_null, fmt = parsed.output_type, (list(parsed.box_xyxy_px) if parsed.box_xyxy_px else None), dec["p_null"], parsed.parse_ok
            if gt is None:
                correct = otype != "box"
            else:
                correct = bool(box) and P.iou(tuple(box), tuple(gt)) >= 0.5
            rows.append({"id": it["id"], "kind": it["kind"], "output_type": otype, "correct": bool(correct), "p_null": p_null, "format_ok": bool(fmt)})
    model.train()
    neg = [r for r in rows if r["kind"] == "negative"]
    pos = [r for r in rows if r["kind"] != "negative"]
    pn = [r["p_null"] for r in neg if r["p_null"] is not None]
    return {
        "n": len(rows),
        "neg_null_rate": (sum(r["output_type"] != "box" for r in neg) / len(neg)) if neg else None,
        "neg_p_null_median": statistics.median(pn) if pn else None,
        "neg_p_null_frac_ge_0.1": (sum(p >= 0.1 for p in pn) / len(pn)) if pn else None,
        "pos_acc": (sum(r["correct"] for r in pos) / len(pos)) if pos else None,
        "pos_null_rate": (sum(r["output_type"] in D.NULL_TYPES for r in pos) / len(pos)) if pos else None,
        "format_ok_rate": sum(r["format_ok"] for r in rows) / len(rows) if rows else None,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="4b", choices=list(D.MODELS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val-scenes", type=int, default=18)
    ap.add_argument("--drop-4b-refused", action="store_true", help="drop negatives the 4B base already refuses")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap the training items")
    ap.add_argument("--decision-weight", type=float, default=1.0)
    ap.add_argument("--null-unlikelihood", type=float, default=0.0, help="weight of -log(1-p(null)) on positives")
    ap.add_argument("--neg-repeat", type=int, default=1)
    ap.add_argument("--extra", default=None, help="comma-separated item files appended to the training set")
    ap.add_argument("--coa-parts", default="turn1,audit,answer", help="--audit coa: which example kinds to train (turn1,audit,answer); 'turn1' alone = proposer-only SFT")
    ap.add_argument("--extra-n", type=int, default=None, help="seeded cap on the items taken from each --extra file")
    ap.add_argument("--init-adapter", default=None, help="warm-start from this LoRA adapter")
    ap.add_argument("--no-gray-eval", action="store_true")
    ap.add_argument("--trace", default=None, choices=list(TRACE_PROMPTS), help="learn verification traces rendered from the labels (train.traces)")
    ap.add_argument("--order", default="iid", choices=TR.ORDERS, help="candidate order in the label traces: iid, or commit (described object first)")
    ap.add_argument("--hint-crop", action="store_true", help="two-image prompt: full scene + close-up of the base model's own box (train.traces.train_hint_box)")
    ap.add_argument("--hint-noise", type=float, default=0.0, help="probability of replacing the hint by another instance's box (teaches escalation)")
    ap.add_argument("--n-crops", type=int, default=1, help="close-ups per prompt under --hint-crop (1 = the hint only; >1 adds the largest other instances)")
    ap.add_argument("--turns", type=int, default=1, choices=[1, 2], help="2 = the v2 protocol: conditions + zoom tool call, crops, then the table")
    ap.add_argument("--train-merger", action="store_true", help="also train the vision merger MLPs (saved in the adapter via modules_to_save)")
    ap.add_argument("--val-loss-every", type=int, default=20, help="teacher-forced loss on the held-out label traces every N optimizer steps (0 = off)")
    ap.add_argument("--verdict-weight", default=None, help="two-turn loss weights 'target,other,reason': verdict tokens of the target row, of other rows, and rationale tokens (e.g. 4,2,0.5)")
    ap.add_argument("--audit", default=None, choices=[None, "coa"], help="--turns 2: per-candidate COA audit examples (turn-1 example + one audit example per label row) instead of the joint table")
    ap.add_argument("--scene-frac", type=float, default=1.0, help="seeded fraction of the training scenes used for SFT (the rest is left for RL)")
    ap.add_argument("--scale-aug", type=float, default=0.0, help="probability of shrinking the scene onto a gray canvas so the object is GME-sized")
    ap.add_argument("--scale-frac", type=lambda s: tuple(float(x) for x in s.split(",")), default=(0.02, 0.10),
                    help="target area fraction range of the described object under --scale-aug")
    ap.add_argument("--eval-max-new", type=int, default=None)
    args = ap.parse_args()
    global COA_PARTS
    COA_PARTS = {x.strip() for x in args.coa_parts.split(",") if x.strip()}  # before the step count, which expands the examples

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup

    torch.manual_seed(args.seed)
    out = D.TRAIN_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")

    items = D.load_items()
    train_items, val_items = D.split(items, args.n_val_scenes, args.seed)
    dropped = []
    if args.drop_4b_refused:
        lk = D.policy_4b_lookup()
        keep = []
        for it in train_items:
            v = D.base4b_view(it, lk)
            if it["kind"] == "negative" and v and v.get("output_type") in D.NULL_TYPES:
                dropped.append(it["id"])
            else:
                keep.append(it)
        train_items = keep
    if args.limit:
        train_items = train_items[: args.limit]
    val_groups = {it["group"] for it in val_items}
    extra_n = 0
    for spec in [x for x in (args.extra or "").split(",") if x.strip()]:
        f, _, cap = spec.partition("@")  # path@n caps that file; --extra-n caps the rest
        cap_n = int(cap) if cap else args.extra_n
        ex_items = [it for it in D.load_items(Path(f).expanduser()) if it["group"] not in val_groups and it.get("src_group") not in val_groups]
        if cap_n and len(ex_items) > cap_n:
            ex_items = random.Random(args.seed).sample(ex_items, cap_n)
        extra_n += len(ex_items)
        train_items.extend(ex_items)
    if args.neg_repeat > 1:
        train_items.extend([it for it in train_items if it["kind"] == "negative"] * (args.neg_repeat - 1))
    if args.scene_frac < 1.0:  # SFT on a seeded subset of scenes; RL later sees the rest fresh
        groups = sorted({it["group"] for it in train_items})
        random.Random(args.seed + 3).shuffle(groups)
        keep_groups = set(groups[: max(1, int(len(groups) * args.scene_frac))])
        train_items = [it for it in train_items if it["group"] in keep_groups]
    trace_drops = {}
    if args.trace:
        raw_items = list(train_items)
        train_items, trace_drops = TR.build_all(train_items, args.trace, args.order)
        if args.audit == "coa":  # answer-only positives (RefCOCO train) are kept: expand_coa gives them a one-clause label table
            kept = {id(it) for it in train_items}
            kept_ids = {it["id"] for it in train_items}  # build_all returns new dicts: compare by id, not identity (179 labelled positives were duplicated as answer-only items in sft_coa4)
            extra_ao = [it for it in raw_items if it["id"] not in kept_ids and id(it) not in kept and it.get("answer", {}).get("bbox_2d") is not None and it.get("kind") == "positive"
                        and not it.get("matrix") and not it.get("matrix_pre") and TR.label_matrix(it)[0] is None]
            train_items = train_items + extra_ao
            print(f"  coa: {len(extra_ao)} answer-only positives kept for turn-1/audit/answer supervision", flush=True)
    (out / "split.json").write_text(json.dumps({"train": [i["id"] for i in train_items], "val": [i["id"] for i in val_items],
                                                "dropped_4b_refused": dropped, "trace_drops": trace_drops}, indent=1), encoding="utf-8")
    kinds = lambda xs: {k: sum(1 for i in xs if i["kind"] == k) for k in ("positive", "sibling_positive", "negative")}
    print(f"train {len(train_items)} {kinds(train_items)} (extra {extra_n}, trace drops {trace_drops}) | val {len(val_items)} {kinds(val_items)} | dropped {len(dropped)}", flush=True)
    if args.trace:
        (out / "train_traces.jsonl").write_text("".join(json.dumps({"id": i["id"], "expression": i["expression"], "trace": i.get("trace", "")}) + "\n" for i in train_items), encoding="utf-8")

    hf, rev = D.MODELS[args.model]
    if args.turns == 2:
        assert args.trace, "--turns 2 needs --trace"
        template = prompts.load("grounding_verify_trace_tool", non_kill=True)
    elif args.hint_crop:
        assert args.trace == "yn", "--hint-crop is implemented for the yes/no trace"
        template = prompts.load("grounding_verify_trace_crops" if args.n_crops > 1 else "grounding_verify_trace_crop", non_kill=True)
    else:
        template = prompts.load(TRACE_PROMPTS[args.trace], non_kill=True) if args.trace else prompts.load("grounding_qwen3vl_primary")
    processor = T.load_capped_processor(hf, rev)
    T.assert_cap_is_in_force(processor)
    tok = processor.tokenizer
    null_ids, box_ids = _first_ids(tok, [" null", "null", " Null"]), _first_ids(tok, [" [", "["])
    model = AutoModelForImageTextToText.from_pretrained(hf, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        extra = {"modules_to_save": ["merger", "deepstack_merger_list.0", "deepstack_merger_list.1", "deepstack_merger_list.2"]} if args.train_merger else {}
        cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=LORA_TARGETS, task_type="CAUSAL_LM", **extra)
        model = get_peft_model(model, cfg)
    decision_ids = set(null_ids) | set(box_ids)
    n_lora = 0
    touched = set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights in fp32 for stable small-batch updates
            n_lora += p.numel()
            touched.add(n.split(".lora")[0].rsplit(".", 1)[0].split(".")[0:3].__repr__())
    print(f"LoRA params {n_lora / 1e6:.1f}M; {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    vis = [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    if args.train_merger:
        assert vis and all("merger" in n for n in vis), f"unexpected trainable vision params: {vis[:3]}"
        print(f"vision merger trainable: {sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'visual' in n) / 1e6:.1f}M params", flush=True)
    else:
        assert not vis, f"LoRA touched the vision tower: {vis[:3]}"

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(train_items) / args.grad_accum)
    if args.audit == "coa":  # the schedule must count the expanded examples (turn-1 + audits + answer per item)
        n_ex = len(expand_coa(train_items, args.order))
        steps_per_epoch = max(1, n_ex // args.grad_accum)
        print(f"  coa: {n_ex} examples per epoch -> {steps_per_epoch} steps per epoch", flush=True)
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total_steps)), total_steps)
    curve = open(out / "curve.jsonl", "w", encoding="utf-8")

    val_traced = TR.build_all(val_items, args.trace, args.order)[0] if args.trace else []

    @torch.no_grad()
    def val_loss() -> float | None:
        """Teacher-forced cross-entropy on the held-out scenes' label traces (overfitting check)."""
        if not val_traced or (args.hint_crop and args.turns != 2):  # the crop prompt needs hint fields; skipped for that path
            return None
        model.eval()
        tot, n = 0.0, 0
        for it in val_traced:
            image = D.open_image(it)
            if args.turns == 2:
                t1, boxes, t2 = TR.render_turns(it["matrix"], it["image_wh"], args.trace, args.order)
                ex = build_example_turns(processor, template, it, image, TR.turn2_views(image, boxes), t1, t2, vw, target_row_of(it["matrix"])).to(model.device)
            else:
                ex = build_example(processor, template, it, image).to(model.device)
            labels, weights = ex.pop("labels"), ex.pop("loss_weights")
            ex.pop("decision_pos")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tot += float(sparse_loss(model, ex, labels, weights))
            n += 1
        model.train()
        return tot / max(1, n)

    def log_eval(step: int):
        t0 = time.time()
        real = evaluate(model, processor, template, val_items, null_ids, box_ids, trace=args.trace, max_new=args.eval_max_new, hint_crop=args.hint_crop, n_crops=args.n_crops, turns=args.turns)
        rec = {"step": step, **{k: v for k, v in real.items() if k != "rows"}, "gray_neg_null_rate": float("nan")}
        if not args.no_gray_eval:
            gray = evaluate(model, processor, template, [i for i in val_items if i["kind"] == "negative"], null_ids, box_ids, gray=True,
                            trace=args.trace, max_new=args.eval_max_new, hint_crop=args.hint_crop, n_crops=args.n_crops, turns=args.turns)
            rec.update(gray_neg_null_rate=gray["neg_null_rate"], gray_neg_p_null_median=gray["neg_p_null_median"])
        rec["eval_s"] = round(time.time() - t0)
        curve.write(json.dumps(rec) + "\n")
        curve.flush()
        pn = rec.get("neg_p_null_median")
        print(f"  [eval step {step}] neg null {rec['neg_null_rate']:.2f} (gray {rec['gray_neg_null_rate']:.2f}), "
              f"neg p(null) median {pn if pn is None else f'{pn:.2e}'} | "
              f"pos acc {rec['pos_acc']:.2f}, pos null {rec['pos_null_rate']:.2f} | format ok {rec['format_ok_rate']:.2f} | {rec['eval_s']}s", flush=True)
        return rec

    log_eval(0)
    model.train()
    aug_rng = random.Random(args.seed + 7)
    hint_lookup = D.policy_4b_lookup() if args.hint_crop else None
    step, micro, t0 = 0, 0, time.time()
    running = []
    vw = tuple(float(x) for x in args.verdict_weight.split(",")) if args.verdict_weight else None
    target_row_of = lambda m: next((k for k, r in enumerate(TR.ordered_rows(m, args.order), 1) if r["iid"] == m.get("first_iid")), None)
    coa_tmpl = prompts.load("grounding_coa_audit", non_kill=True) if args.audit == "coa" else None
    coa_ans_tmpl = prompts.load("grounding_coa_answer", non_kill=True) if args.audit == "coa" else None
    E.AUDIT = args.audit  # the in-training eval uses the same head
    for epoch in range(args.epochs):
        TR.ORDER_SALT = epoch  # seeded row orders vary across epochs
        order = D.training_order(expand_coa(train_items, args.order) if args.audit == "coa" else train_items, args.seed + epoch)
        for it in order:
            image, target_text, tfm, m2 = D.open_image(it), None, (1.0, 0, 0), None
            if args.trace and args.scale_aug > 0 and it.get("matrix") and aug_rng.random() < args.scale_aug:
                lo, hi = args.scale_frac
                image, m2, tfm = TR.scale_augment(image, it["matrix"], it["image_wh"], aug_rng.uniform(lo, hi), aug_rng)
                target_text = TR.render(m2, it["image_wh"], args.trace, args.order)
            views, fields = None, None
            if args.turns == 2:
                m_use = m2 if target_text is not None else it["matrix"]
                t1, boxes, t2 = TR.render_turns(m_use, it["image_wh"], args.trace, args.order)
                if args.audit == "coa":
                    k = it.get("_coa_k", 0)
                    if k == -1:
                        rows_ = TR.ordered_rows(m_use, args.order)
                        rof = _rationale_of(m_use)
                        audits_raw = [COA.render_audit(m_use, r_, rof) for r_ in rows_]
                        ans_box = next((D.relative_1000(r_["box"], it["image_wh"]) for r_ in rows_ if r_["iid"] == m_use.get("answer_iid")), None)
                        answer_text = "<answer>" + json.dumps({"bbox_2d": ans_box}) + "</answer>"
                        ex = build_example_answer(processor, coa_ans_tmpl, it, t1, audits_raw, [D.relative_1000(b, it["image_wh"]) for b in boxes], answer_text).to(model.device)
                    else:
                        ex = (build_example_turn1(processor, template, it, image, t1) if k == 0
                              else build_example_coa(processor, template, coa_tmpl, it, image, m_use, t1, boxes, k, args.order)).to(model.device)
                else:
                    crops = TR.turn2_views(image, boxes)
                    ex = build_example_turns(processor, template, it, image, crops, t1, t2, vw, target_row_of(m_use)).to(model.device)
            else:
                if args.hint_crop:
                    boxes = [TR.transform_box(b, tfm) for b in TR.train_hint_boxes(it, hint_lookup, args.n_crops, aug_rng, args.hint_noise)]
                    if boxes:
                        views = [image] + [TR.crop_view(image, b) for b in boxes]
                        fields = {"hint": TR.hint_text(boxes[0], it["image_wh"]), "hints": TR.hints_text(boxes, it["image_wh"])}
                    else:
                        views, fields = [image, image], {"hint": "[0, 0, 1000, 1000]", "hints": "[0, 0, 1000, 1000]"}
                ex = build_example(processor, template, it, image, decision_ids, args.decision_weight, target_text, views, fields).to(model.device)
            labels, weights, dec_pos = ex.pop("labels"), ex.pop("loss_weights"), int(ex.pop("decision_pos")[0])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.null_unlikelihood > 0 and it["kind"] != "negative" and dec_pos >= 0:
                    logits = model(**ex).logits  # the unlikelihood term needs the decision position's logits
                    loss = weighted_loss(logits, labels, weights) + args.null_unlikelihood * null_unlikelihood(logits, dec_pos, null_ids)
                else:
                    loss = sparse_loss(model, ex, labels, weights)  # LM head only at labelled positions
            loss = loss / args.grad_accum
            loss.backward()
            running.append(loss.item() * args.grad_accum)
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                with open(out / "train_log.jsonl", "a", encoding="utf-8") as tl:  # structured per-step record
                    tl.write(json.dumps({"kind": "sft", "epoch": epoch + 1, "step": step, "total": total_steps,
                                         "loss": sum(running[-args.grad_accum:]) / len(running[-args.grad_accum:]),
                                         "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 2)}) + "\n")
                if step % 5 == 0:
                    print(f"  epoch {epoch + 1} step {step}/{total_steps} loss {sum(running[-args.grad_accum * 5:]) / len(running[-args.grad_accum * 5:]):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f} min", flush=True)
                if args.val_loss_every and step % args.val_loss_every == 0:
                    vl = val_loss()
                    with open(out / "train_log.jsonl", "a", encoding="utf-8") as tl:
                        tl.write(json.dumps({"kind": "sft_val", "epoch": epoch + 1, "step": step, "total": total_steps, "val_loss": vl, "min": round((time.time() - t0) / 60, 2)}) + "\n")
                    print(f"  [val loss step {step}] {vl:.4f}", flush=True)
                if step % args.eval_every == 0:
                    log_eval(step)
    if step % args.eval_every != 0:
        log_eval(step)
    model.save_pretrained(out / "adapter")
    curve.close()
    print(f"saved {out / 'adapter'}; curve {out / 'curve.jsonl'}; {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
