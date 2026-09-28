"""Stage B of the PoC: GRPO on top of the SFT adapter (hand-written; no TRL in the env).

    python -m train.grpo_lora --name grpo_v1 --init-adapter ~/vlmg-data/train/sft_v1/adapter
        [--epochs 2] [--group 8] [--temperature 1.0] [--lr 2e-5] [--prompts-per-step 4]
        [--beta 0.0] [--r-pos-null -0.5] [--r-neg-box 0.0] [--eval-every 10] [--extra a.jsonl]
        [--trace yn|obs --reward v2|v3|v4] [--max-scenes N]

Coordinate mode (no --trace), reward per sample: positive item -> 1 if the box has IoU>=0.5
with the target, 0 for another box, --r-pos-null for a null answer (over-refusal penalty);
negative item -> 1 for null, --r-neg-box for any box; unparsable -> -1.

Trace mode (--trace, notes/DESIGN-FINAL-LIGHT-2026-09-24.md section 5): the completion is a
verification trace (train.traces).  Unparsable / over-long / malformed -> -1.  Otherwise
0.1 format + the answer term on the *free* answer:
  v2  independent: positive box IoU>=0.5 -> 1, other box 0, null -> --r-pos-null; negative null 1, box --r-neg-box
  v3  pair-level: a negative's null earns the same-scene positive's success rate in this step
      (its group of rollouts); cross-scene negatives (no twin) keep the v2 rule
  v4  v3 plus: the null of a flipped-clause negative is scaled by 0.2 unless some candidate marks
      the flipped clause "no" (evidence gate); a "no" on the candidate matching the ground-truth
      box of a positive costs 0.5 (false accusation); free answer != derived answer costs 0.5
Scenes are batched together so the positive's rollouts are sampled before its negative's.
Advantages are group-normalised; the objective is the sequence-mean log-probability weighted
by the advantage (one update per batch of samples, importance ratio 1) plus an optional KL to
the frozen start adapter (--beta > 0 loads it a second time as adapter "ref").  Prompts whose
group has identical rewards carry no gradient and are skipped after sampling.  --inject-gt neg
adds the label trace / null answer as an extra completion when a negative's group has no null.
Output: adapter/, curve.jsonl, args.json.
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

from scripts.pilot_abstain_signal import _first_ids
from shared.harness import parsers as P
from shared.harness import prompts
from shared.harness import tokens as T
from train import data as D
from train import traces as TR
from train.eval_suite import TRACE_PROMPTS
import train.eval_suite as E
from train import grpo_coa as GC
from train.sft_lora import evaluate


def reward_of(raw: str, item: dict, args) -> tuple[float, str]:
    wh = tuple(item["image_wh"])
    parsed = P.parse_qwen3vl(raw, convention=P.RELATIVE_1000, original_wh=wh, sent_wh=wh)
    gt = item["answer"]["bbox_2d"]
    if parsed.output_type == "box" and parsed.box_xyxy_px:
        if gt is None:
            return args.r_neg_box, "box"
        return (1.0 if P.iou(tuple(parsed.box_xyxy_px), tuple(gt)) >= 0.5 else 0.0), "box"
    if parsed.output_type in D.NULL_TYPES:
        return (1.0 if gt is None else args.r_pos_null), "null"
    return -1.0, "bad"


_REASON_STOP = {"not", "the", "and", "with", "its", "has", "have", "are", "but", "this", "that", "there", "visible", "instead",
                "rather", "than", "does", "appears", "looks", "seems", "actually", "one", "two", "some", "any", "only", "just", "very"}


def trace_reward(raw: str, item: dict, args, s_pos: float | None) -> tuple[float, str, dict]:
    """Reward of a verification trace (see the module docstring); s_pos is the same-scene
    positive's success rate in this step, None when it was not sampled."""
    wh = tuple(item["image_wh"])
    pr = TR.parse(raw, wh)
    if getattr(args, "derive", "current") == "named":  # a bare 'no' does not exclude: the accusation must name what was seen
        pr["derived_type"], pr["derived_box"], pr["derived_cand"] = TR.rederive(pr, "named")
    if not pr["format_ok"]:
        return -1.0, "bad", {"n_cand": pr["n_cand"]}
    gt = item["answer"]["bbox_2d"]
    info: dict = {"n_cand": pr["n_cand"]}
    r = 0.1
    derived = getattr(args, "answer", "free") == "derived"  # score the table-derived answer instead of the free one
    ft = pr["derived_type"] if derived else pr["free_type"]
    fbox = pr["derived_box"] if derived else pr["free_box"]
    if gt is not None:
        if ft == "box":
            iou = P.iou(tuple(fbox), tuple(gt))
            if iou >= 0.5:
                r += (0.5 + 0.5 * iou) if getattr(args, "iou_soft", False) else 1.0  # soft: 0.75 at IoU 0.5 -> 1.0 at IoU 1 (finer boxes)
        else:
            r += args.r_pos_null
        if args.reward == "v4":
            fn = TR.false_no_cells(pr, gt)
            if fn and fn[0] > 0:
                r -= 0.5
                info["false_accusation"] = True
    else:
        if ft == "null":
            base = 1.0 if (args.reward == "v2" or s_pos is None) else max(s_pos, getattr(args, "s_floor", 0.0))
            clause = (item.get("matrix") or {}).get("flipped_clause")
            if args.reward == "v4" and clause:
                ev = TR.evidence(pr, clause, need_seen=(args.trace == "obs"))  # obs format: the accusation must name what was seen
                info["evidence"] = ev
                if not ev:
                    base *= 0.2
            r += base
        else:
            r += args.r_neg_box
    if args.reward == "v4" and not TR.consistent(pr):
        r -= 0.5
        info["inconsistent"] = True
    wr = getattr(args, "r_reason", 0.0)
    m_ = item.get("matrix") or {}
    if wr > 0 and gt is None and m_.get("flipped_clause") and m_.get("flip_from"):
        j = TR.align(pr["conditions"], m_["flipped_clause"])
        words = {w_ for w_ in re.findall(r"[a-z0-9]+", m_["flip_from"].lower()) if len(w_) > 2}
        known = words | set(re.findall(r"[a-z0-9]+", m_["flipped_clause"].lower())) | _REASON_STOP

        def names_value(reason) -> bool:
            # guard against enumeration ("not red, blue, green, gray ..."): only the first 12 words count and at most
            # 4 content words outside the original value, the clause and function words are allowed
            toks = re.findall(r"[a-z0-9]+", (reason or "").lower())[:12]
            return bool(words & set(toks)) and sum(1 for t in toks if len(t) > 2 and t not in known) <= 4

        named = j is not None and any(c["verdicts"].get(j) == "no" and names_value(c["seen"].get(j)) for c in pr["candidates"])
        info["reason_named"] = named
        if named:
            r += wr
    wc = getattr(args, "r_cell", 0.0)
    if wc > 0 and m_.get("rows"):
        agree = total = 0
        cols = [TR.align(pr["conditions"], c) for c in m_["conditions"]]  # label clause -> parsed condition index
        for c in pr["candidates"]:
            if not c.get("box"):
                continue
            row = max(m_["rows"], key=lambda r_: P.iou(tuple(c["box"]), tuple(r_["box"])))
            if P.iou(tuple(c["box"]), tuple(row["box"])) < 0.5:
                continue
            for lab, j in zip(row["verdicts"], cols):
                if j is None or lab == "unclear":
                    continue
                v = c["verdicts"].get(j)
                if v in ("yes", "no"):
                    total += 1
                    agree += v == lab
        if total:
            info["cell_acc"] = agree / total
            r += wc * (2.0 * agree / total - 1.0)  # -wc .. +wc: chance level earns nothing
    w = getattr(args, "r_commit", 0.0)
    if w > 0:  # commit reward: the first candidate should be the described object (commit-first traces)
        tb = TR.intended_box(item.get("matrix"))
        c1 = pr["candidates"][0]["box"] if pr["candidates"] else None
        if tb is not None and c1 is not None:
            ciou = P.iou(tuple(c1), tuple(tb))
            r += w * ciou
            info["commit_iou"] = ciou
    return r, ("null" if ft == "null" else "box"), info


def trim(comp: list[int], eos: set[int], pad: int | None) -> list[int]:
    """Completion tokens up to and including the first end token; trailing padding dropped."""
    out = []
    for t in comp:
        out.append(t)
        if t in eos:
            break
    while out and pad is not None and out[-1] == pad and out[-1] not in eos:
        out.pop()
    return out


def completion_logprobs(model, inputs, comp: list[int]) -> torch.Tensor:
    """Log-probabilities of the completion tokens given the prompt inputs (gradient flows)."""
    comp_t = torch.tensor([comp], device=model.device)
    ids = torch.cat([inputs["input_ids"], comp_t], dim=1)
    kw = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
    if "mm_token_type_ids" in inputs:  # per-token modality (0 = text) must cover the completion too
        mtt = inputs["mm_token_type_ids"]
        kw["mm_token_type_ids"] = torch.cat([mtt, torch.zeros((1, len(comp)), dtype=mtt.dtype, device=mtt.device)], dim=1)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=len(comp) + 1, **kw)
    logits = out.logits[0, :-1].float()  # positions n_prompt-1 .. L-2 predict the completion
    assert logits.shape[0] == len(comp), (logits.shape, len(comp))
    return torch.log_softmax(logits, dim=-1).gather(1, comp_t[0].unsqueeze(1)).squeeze(1)


def completion_stats(model, inputs, comp: list[int]) -> tuple[torch.Tensor, float]:
    """(log-probabilities of the completion tokens, mean token entropy in nats) - the entropy is
    a monitoring quantity only (detached)."""
    comp_t = torch.tensor([comp], device=model.device)
    ids = torch.cat([inputs["input_ids"], comp_t], dim=1)
    kw = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
    if "mm_token_type_ids" in inputs:
        mtt = inputs["mm_token_type_ids"]
        kw["mm_token_type_ids"] = torch.cat([mtt, torch.zeros((1, len(comp)), dtype=mtt.dtype, device=mtt.device)], dim=1)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=len(comp) + 1, **kw)
    logits = out.logits[0, :-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    ent = float((-(lp.exp() * lp).sum(-1)).mean().detach())
    return lp.gather(1, comp_t[0].unsqueeze(1)).squeeze(1), ent


def candidate_reward(boxes: list, item: dict, args, info: dict) -> float:
    """Turn-1 terms of the v2 protocol: --r-recall if the described object is among the named
    boxes (IoU >= 0.5); -0.3 when no valid box was named; -0.05 per box matching no labelled
    instance (when the item has a label matrix)."""
    if not boxes:
        info["no_boxes"] = True
        return -0.3
    r = 0.0
    tb = TR.intended_box(item.get("matrix")) or ((item.get("answer") or {}).get("bbox_2d"))
    if tb is not None:
        hit = any(P.iou(tuple(b), tuple(tb)) >= 0.5 for b in boxes)
        info["cand_recall"] = hit
        r += args.r_recall * (1.0 if hit else 0.0)
    m = item.get("matrix")
    if m:
        inst = [row["box"] for row in m["rows"]]
        stray = sum(1 for b in boxes if all(P.iou(tuple(b), tuple(x)) < 0.5 for x in inst))
        r -= getattr(args, "r_stray", 0.05) * stray
    return r


def turn2_prompt(processor, template, item: dict, image, t1: str, boxes: list) -> tuple[str, list]:
    """Chat text and images of the turn-2 prompt: user(image, prompt) -> assistant(t1) -> user(crops)."""
    crops = TR.turn2_views(image, boxes)
    u1 = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": template.render(expr=item["expression"])}]}
    a1 = {"role": "assistant", "content": [{"type": "text", "text": t1}]}
    u2 = {"role": "user", "content": [{"type": "image"} for _ in crops] + [{"type": "text", "text": TR.turn2_user_text(len(boxes)) if boxes else "No valid boxes were given; answer from the full image. Continue with step 2."}]}
    return processor.apply_chat_template([u1, a1, u2], tokenize=False, add_generation_prompt=True), [image] + crops


def turn2_inputs(processor, template, item: dict, image, t1: str, boxes: list, device):
    """Processor inputs (unpadded, batch 1) of the turn-2 prompt; used for the log-probabilities."""
    text, images = turn2_prompt(processor, template, item, image, t1, boxes)
    return processor(images=images, text=text, return_tensors="pt").to(device)


@torch.no_grad()
def sample_two_turn(model, processor, template, item: dict, image, inputs1, args, eos: set, tok):
    """G rollouts of the v2 protocol: turn 1 sampled until </tool_call>, boxes cropped, turn 2
    sampled with the crops.  Returns (segments per rollout, full raw texts, boxes per rollout)."""
    n1 = inputs1["input_ids"].shape[1]
    gen1 = model.generate(**inputs1, max_new_tokens=args.max_new1, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                          num_return_sequences=args.group, stop_strings=["</tool_call>"], tokenizer=tok)
    comps1 = [trim(seq[n1:].tolist(), eos, tok.pad_token_id) for seq in gen1]
    raws1 = [processor.decode(c, skip_special_tokens=True) for c in comps1]
    boxes_per = [TR.parse_tool_boxes(r, tuple(item["image_wh"])) for r in raws1]
    prompts2 = [turn2_prompt(processor, template, item, image, r1, bx) for r1, bx in zip(raws1, boxes_per)]
    inputs2 = [processor(images=imgs, text=text, return_tensors="pt").to(model.device) for text, imgs in prompts2]  # unpadded, for the backward
    comps2: list[list[int]] = []
    sb = max(1, getattr(args, "sample_batch", 4))
    processor.tokenizer.padding_side = "left"
    for s0 in range(0, len(prompts2), sb):  # turn 2 sampled in left-padded batches instead of one rollout at a time
        chunk = prompts2[s0:s0 + sb]
        batch = processor(images=[im for _, imgs in chunk for im in imgs], text=[t for t, _ in chunk], padding=True, return_tensors="pt").to(model.device)
        nb = batch["input_ids"].shape[1]
        gen2 = model.generate(**batch, max_new_tokens=args.max_new, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0)
        comps2.extend(trim(seq[nb:].tolist(), eos, tok.pad_token_id) for seq in gen2)
    segs, raws = [], []
    for c1, r1, inp2, c2 in zip(comps1, raws1, inputs2, comps2):
        segs.append([(inputs1, c1), (inp2, c2)])
        raws.append(r1 + "\n" + processor.decode(c2, skip_special_tokens=True))
    return segs, raws, boxes_per


RANK = {"positive": 0, "sibling_positive": 1, "negative": 2}


def scene_batches(items: list[dict], per_step: int, seed: int, max_scenes: int | None) -> list[list[dict]]:
    """Scenes shuffled (those with a positive+negative pair first when capped); every item of a
    scene lands in the same step with the positive first, so its rollouts are scored before
    the negative's."""
    by_group: dict[str, list[dict]] = {}
    for it in items:
        by_group.setdefault(it["group"], []).append(it)
    rng = random.Random(seed)
    groups = list(by_group)
    rng.shuffle(groups)
    if max_scenes:
        extra = [g for g in groups if g.startswith("refcoco:")]  # answer-only items ride along, not counted as scenes
        groups = [g for g in groups if not g.startswith("refcoco:")]
        paired = [g for g in groups if {"positive", "negative"} <= {it["kind"] for it in by_group[g] if it.get("source") != "cross"}]
        rest = [g for g in groups if g not in set(paired)]
        groups = (paired + rest)[:max_scenes] + extra
        rng.shuffle(groups)
    batches: list[list[dict]] = []
    cur: list[dict] = []
    for g in groups:
        scene = sorted(by_group[g], key=lambda it: (RANK[it["kind"]], it.get("source") == "cross"))
        if cur and len(cur) + len(scene) > per_step:
            batches.append(cur)
            cur = []
        cur.extend(scene)
    if cur:
        batches.append(cur)
    return batches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="4b", choices=list(D.MODELS))
    ap.add_argument("--init-adapter", default=None, help="start adapter; without it a fresh LoRA of rank --r is trained from the base")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=None, help="default 48, or 384 with --trace")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--r-pos-null", type=float, default=-0.5)
    ap.add_argument("--r-neg-box", type=float, default=0.0)
    ap.add_argument("--neg-adv-scale", type=float, default=1.0, help="RC-GRPO alpha: scale advantages of negative items")
    ap.add_argument("--s-floor", type=float, default=0.0, help="negative null credit = max(s+, floor)")
    ap.add_argument("--no-std", action="store_true", help="advantage = r - mean (no division by the group std; Dr.GRPO)")
    ap.add_argument("--derive", default="current", choices=["current", "named"], help="named: only a 'no' whose reason names the observed value excludes a candidate")
    ap.add_argument("--r-stray", type=float, default=0.05, help="penalty per turn-1 box matching no labelled instance")
    ap.add_argument("--inject-gt", default="none", choices=["none", "neg", "both", "all"],
                    help="neg: forced null completion when a negative's group has none (RC-GRPO); "
                         "all: ground truth in every group with reward smoothing (ViSurf)")
    ap.add_argument("--trace", default=None, choices=list(TRACE_PROMPTS))
    ap.add_argument("--reward", default="v2", choices=["v2", "v3", "v4"], help="trace-mode reward variant")
    ap.add_argument("--answer", default="free", choices=["free", "derived"], help="which answer the trace reward scores")
    ap.add_argument("--r-commit", type=float, default=0.0, help="weight of IoU(first candidate, described object) added to the trace reward")
    ap.add_argument("--hint-crop", action="store_true", help="two-image prompt: full scene + close-up of the base model's own box")
    ap.add_argument("--hint-noise", type=float, default=0.0, help="probability of replacing the first hint by another instance's box")
    ap.add_argument("--n-crops", type=int, default=1, help="close-ups per prompt under --hint-crop")
    ap.add_argument("--turns", type=int, default=1, choices=[1, 2], help="2 = the v2 protocol (tool-call zoom on the model's own candidates)")
    ap.add_argument("--max-new1", type=int, default=400, help="turn-1 token budget under --turns 2 (conditions + tool call; 200 truncated long descriptions)")
    ap.add_argument("--sample-batch", type=int, default=4, help="turn-2 rollouts sampled per generate call under --turns 2 (memory)")
    ap.add_argument("--audit", default=None, choices=[None, "coa"], help="--turns 2: propose -> per-candidate observe/judge audits -> answer step (train.grpo_coa)")
    ap.add_argument("--k-max", type=int, default=4, help="--audit coa: candidates audited per rollout (turn-1 boxes beyond this are dropped)")
    ap.add_argument("--norm-tokens", type=int, default=1024, help="--audit coa: fixed token normaliser of the policy loss")
    ap.add_argument("--r-false-acc", type=float, default=0.0, help="--audit coa: penalty on the true object's audit segment for a named false accusation (0 = priced by the forgone box reward)")
    ap.add_argument("--r-echo", type=float, default=0.0, help="--audit coa: dense sycophancy penalty on the described instance's audit segment when the falsified clause's line is judged match (0 = off)")
    ap.add_argument("--r-recall", type=float, default=0.5, help="turn-1 reward when the described object is among the named boxes")
    ap.add_argument("--r-reason", type=float, default=0.0, help="negatives: bonus when the flipped cell is 'no' and its reason names the original value (keyword match)")
    ap.add_argument("--iou-soft", action="store_true", help="positive box reward 0.5+0.5*IoU above the 0.5 threshold instead of a flat 1")
    ap.add_argument("--r-cell", type=float, default=0.0, help="dense cell reward: w*(2*acc-1) where acc = agreement of verdict cells with the label matrix over candidates matching labelled instances")
    ap.add_argument("--token-level", action="store_true", help="policy loss on summed token log-probs / max_new (DAPO) instead of the per-sequence mean")
    ap.add_argument("--order", default="iid", choices=TR.ORDERS, help="candidate order of the injected label traces")
    ap.add_argument("--max-scenes", type=int, default=None, help="cap the scenes per epoch (paired scenes first)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0, help="also save the adapter to <name>/adapter_step<N> every N steps (0 = off)")
    ap.add_argument("--mini-gme", type=int, default=0, help="N Rejection + N positives of GME evaluated at every --eval-every (curve of the number that matters)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val-scenes", type=int, default=18)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--extra-n", type=int, default=None, help="seeded cap on the items taken from each --extra file")
    ap.add_argument("--kinds", default="positive,sibling_positive,negative")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-gray-eval", action="store_true")
    ap.add_argument("--skip-eval0", action="store_true", help="skip the step-0 eval (known from the start adapter)")
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = D.TRAIN_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    max_new = args.max_new = args.max_new or (384 if args.trace else 48)  # written back: the two-turn sampler and the token-level normalizer read args.max_new

    train_items, val_items = D.split(D.load_items(), args.n_val_scenes, args.seed)
    val_groups = {it["group"] for it in val_items}
    for spec in [x for x in (args.extra or "").split(",") if x.strip()]:
        f, _, cap = spec.partition("@")  # path@n caps that file; --extra-n caps the rest
        cap_n = int(cap) if cap else args.extra_n
        ex_items = [it for it in D.load_items(Path(f).expanduser()) if it["group"] not in val_groups and it.get("src_group") not in val_groups]
        if cap_n and len(ex_items) > cap_n:
            ex_items = random.Random(args.seed).sample(ex_items, cap_n)
        train_items.extend(ex_items)
    keep = set(args.kinds.split(","))
    train_items = [it for it in train_items if it["kind"] in keep]
    trace_drops = {}
    if args.trace:
        answer_only = [it for it in train_items if it.get("source") == "refcoco"]  # no label matrix: answer reward only
        train_items, trace_drops = TR.build_all([it for it in train_items if it.get("source") != "refcoco"], args.trace, args.order)
        train_items.extend(answer_only)
        if answer_only:
            print(f"answer-only items (RefCOCO train): {len(answer_only)}", flush=True)
    if args.limit:
        train_items = train_items[: args.limit]
    kinds = lambda xs: {k: sum(1 for i in xs if i["kind"] == k) for k in ("positive", "sibling_positive", "negative")}
    print(f"train {len(train_items)} {kinds(train_items)} (trace drops {trace_drops}) | val {len(val_items)} {kinds(val_items)}", flush=True)

    hf, rev = D.MODELS[args.model]
    coa_tmpl = ans_tmpl = None
    if args.audit == "coa":
        coa_tmpl = prompts.load("grounding_coa_audit", non_kill=True)
        ans_tmpl = prompts.load("grounding_coa_answer", non_kill=True)
        E.AUDIT = "coa"  # the in-training eval uses the same head
    if args.turns == 2:
        assert args.trace, "--turns 2 needs --trace"
        template = prompts.load("grounding_verify_trace_tool", non_kill=True)
    elif args.hint_crop:
        assert args.trace == "yn", "--hint-crop is implemented for the yes/no trace"
        template = prompts.load("grounding_verify_trace_crops" if args.n_crops > 1 else "grounding_verify_trace_crop", non_kill=True)
    else:
        template = prompts.load(TRACE_PROMPTS[args.trace], non_kill=True) if args.trace else prompts.load("grounding_qwen3vl_primary")
    hint_lookup = D.policy_4b_lookup() if args.hint_crop else None
    hint_rng = random.Random(args.seed + 11)
    processor = T.load_capped_processor(hf, rev)
    T.assert_cap_is_in_force(processor)
    tok = processor.tokenizer
    null_ids, box_ids = _first_ids(tok, [" null", "null", " Null"]), _first_ids(tok, [" [", "["])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos = {t for t in (tok.eos_token_id, im_end) if t is not None}
    model = AutoModelForImageTextToText.from_pretrained(hf, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        from peft import LoraConfig, get_peft_model
        from train.sft_lora import LORA_TARGETS

        model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    if args.beta > 0:
        assert args.init_adapter, "--beta needs --init-adapter (the reference is the start adapter)"
        model.load_adapter(args.init_adapter, adapter_name="ref")
        model.set_adapter("default")
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout) and "lora" in n:
            m.p = 0.0  # the sampling policy and the trained policy must be the same function
    params = [p for n, p in model.named_parameters() if p.requires_grad and ".ref." not in n]
    for p in params:
        p.data = p.data.float()
    assert not [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    batches0 = scene_batches(train_items, args.prompts_per_step, args.seed + 100, args.max_scenes)
    total_steps = len(batches0) * args.epochs
    print(f"{len(batches0)} steps/epoch over {sum(len(b) for b in batches0)} prompts", flush=True)
    curve = open(out / "curve.jsonl", "a", encoding="utf-8")

    mini_gme = []
    if args.mini_gme:
        from train.eval_suite import gme_subset

        for g in gme_subset(args.mini_gme, args.seed, rej_n=args.mini_gme):
            with __import__("PIL.Image", fromlist=["Image"]).open(g["image"]) as im:
                wh = list(im.size)
            mini_gme.append({"id": g["id"], "expression": g["expr"], "image": g["image"], "image_wh": wh,
                             "kind": "negative" if g["n_gt"] == 0 else "positive", "answer": {"bbox_2d": g["gt_boxes"][0] if g["n_gt"] else None}})
        print(f"GME mini-set for in-training eval: {len(mini_gme)} items", flush=True)

    def log_eval(step: int, extra: dict):
        t0 = time.time()
        real = evaluate(model, processor, template, val_items, null_ids, box_ids, trace=args.trace, hint_crop=args.hint_crop, n_crops=args.n_crops, turns=args.turns, batch=4 if args.turns == 2 else 8)
        rec = {"step": step, **{k: v for k, v in real.items() if k != "rows"}, **extra}
        if not args.no_gray_eval:
            gray = evaluate(model, processor, template, [i for i in val_items if i["kind"] == "negative"], null_ids, box_ids, gray=True, trace=args.trace, hint_crop=args.hint_crop, n_crops=args.n_crops, turns=args.turns)
            rec["gray_neg_null_rate"] = gray["neg_null_rate"]
        if mini_gme:  # a small GME slice inside training: the number that matters, on a curve
            g = evaluate(model, processor, template, mini_gme, null_ids, box_ids, trace=args.trace, hint_crop=args.hint_crop, n_crops=args.n_crops, turns=args.turns, batch=4 if args.turns == 2 else 8)  # two-turn batches of 8 GME items spill VRAM
            rec.update(gme_rej_null=g["neg_null_rate"], gme_pos_acc=g["pos_acc"], gme_pos_null=g["pos_null_rate"],
                       gme_net=(g["neg_null_rate"] or 0) - (g["pos_null_rate"] or 0))
        rec["eval_s"] = round(time.time() - t0)
        curve.write(json.dumps(rec) + "\n")
        curve.flush()
        pn = rec.get("neg_p_null_median")
        gme_s = f" | GME mini: rej {rec['gme_rej_null']:.2f} pos {rec['gme_pos_acc']:.2f} posnull {rec['gme_pos_null']:.2f}" if "gme_rej_null" in rec else ""
        print(f"  [eval step {step}] neg null {rec['neg_null_rate']:.2f} (gray {rec.get('gray_neg_null_rate', float('nan')):.2f}), "
              f"neg p(null) median {pn if pn is None else f'{pn:.2e}'} | pos acc {rec['pos_acc']:.2f}, pos null {rec['pos_null_rate']:.2f} | "
              f"format ok {rec['format_ok_rate']:.2f}{gme_s} | {rec['eval_s']}s", flush=True)
        model.train()

    if not args.skip_eval0:
        log_eval(0, {})
    model.train()
    step, t0 = 0, time.time()
    window = []  # per-prompt stats since the last print
    for epoch in range(args.epochs):
        batches = scene_batches(train_items, args.prompts_per_step, args.seed + 100 + epoch, args.max_scenes)
        for batch in batches:
            n_used = 0
            s_pos: dict[str, float] = {}
            step_ent: list[float] = []
            step_len: list[int] = []
            for it in batch:
                image = D.open_image(it)
                views, fields = [image], {}
                if args.hint_crop:
                    boxes = TR.train_hint_boxes(it, hint_lookup, args.n_crops, hint_rng, args.hint_noise)
                    if boxes:
                        views = [image] + [TR.crop_view(image, b) for b in boxes]
                        fields = {"hint": TR.hint_text(boxes[0], it["image_wh"]), "hints": TR.hints_text(boxes, it["image_wh"])}
                    else:
                        views, fields = [image, image], {"hint": "[0, 0, 1000, 1000]", "hints": "[0, 0, 1000, 1000]"}
                msgs = [{"role": "user", "content": [{"type": "image"} for _ in views] + [{"type": "text", "text": template.render(expr=it["expression"], **fields)}]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(images=views, text=text, return_tensors="pt").to(model.device)
                n_prompt = inputs["input_ids"].shape[1]
                model.eval()
                rollouts = None
                if args.turns == 2 and args.audit == "coa":
                    segs, rollouts = GC.sample_coa(model, processor, template, coa_tmpl, ans_tmpl, it, image, inputs, args, eos, tok)
                    raws = [ro["t1"] + "\n" + "\n".join(ro["audits_raw"]) + "\n" + ro["answer"] for ro in rollouts]
                    boxes_per = None
                    comps = [None] * len(raws)
                elif args.turns == 2:
                    segs, raws, boxes_per = sample_two_turn(model, processor, template, it, image, inputs, args, eos, tok)
                    comps = [None] * len(raws)
                else:
                    with torch.no_grad():
                        gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=args.temperature,
                                             top_p=1.0, top_k=0, num_return_sequences=args.group)
                    comps = [trim(seq[n_prompt:].tolist(), eos, tok.pad_token_id) for seq in gen]
                    segs = [[(inputs, c)] for c in comps]
                    raws = [processor.decode(c, skip_special_tokens=True) for c in comps]
                    boxes_per = None
                model.train()
                is_cross = it.get("source") == "cross"
                if rollouts is not None:
                    rs3 = [GC.coa_reward(ro, it, args) for ro in rollouts]
                    rewards = [rb + ra for rb, ra, _, _ in rs3]
                    types = [t for _, _, t, _ in rs3]
                    infos = []
                    for rb, ra, _, i in rs3:
                        i["r_base"], i["r_acc"] = rb, ra
                        infos.append(i)
                elif args.trace:
                    sp = s_pos.get(it["group"]) if (it["kind"] == "negative" and not is_cross) else None
                    rs = [trace_reward(r, it, args, sp) for r in raws]
                    rewards = [r for r, _, _ in rs]
                    types = [t for _, t, _ in rs]
                    infos = [i for _, _, i in rs]
                    if boxes_per is not None:
                        for k in range(len(rewards)):
                            rewards[k] += candidate_reward(boxes_per[k], it, args, infos[k])
                else:
                    rs2 = [reward_of(r, it, args) for r in raws]
                    rewards = [r for r, _ in rs2]
                    types = [t for _, t in rs2]
                    infos = [{} for _ in rs2]
                if it["kind"] == "positive" and it["answer"]["bbox_2d"] is not None:
                    s_pos[it["group"]] = sum(r >= 1.0 for r in rewards) / len(rewards)
                window.append({"kind": "cross" if is_cross else it["kind"], "mean_r": sum(rewards) / len(rewards), "null_frac": types.count("null") / len(types),
                               "bad_frac": types.count("bad") / len(types), "used": 0.0,
                               "ev": sum(1 for i in infos if i.get("evidence")) / len(infos), "fa": sum(1 for i in infos if i.get("false_accusation")) / len(infos),
                               "inc": sum(1 for i in infos if i.get("inconsistent")) / len(infos),
                               "rec": sum(1 for i in infos if i.get("cand_recall")) / len(infos),
                               "rsn": sum(1 for i in infos if i.get("reason_named")) / len(infos),
                               "kk": (sum(i["k"] for i in infos if "k" in i) / max(1, sum(1 for i in infos if "k" in i))) if any("k" in i for i in infos) else None,
                               "cons": (sum(1 for i in infos if i.get("consistent")) / len(infos)) if rollouts is not None else None,
                               "acc": (sum(1 for i in infos if i.get("accusation_correct")) / len(infos)) if rollouts is not None else None,
                               "syc": (sum(1 for i in infos if i.get("sycophantic")) / len(infos)) if rollouts is not None else None,
                               "uns": (sum(i["unsure_frac"] for i in infos if i.get("unsure_frac") is not None) / max(1, sum(1 for i in infos if i.get("unsure_frac") is not None))) if rollouts is not None else None,
                               "echo": (sum(i["echo_frac"] for i in infos if i.get("echo_frac") is not None) / max(1, sum(1 for i in infos if i.get("echo_frac") is not None))) if rollouts is not None else None,
                               "cell": (sum(i["cell_acc"] for i in infos if i.get("cell_acc") is not None) / max(1, sum(1 for i in infos if i.get("cell_acc") is not None))) if any(i.get("cell_acc") is not None for i in infos) else None})
                need_neg = it["kind"] == "negative" and "null" not in types
                need_pos = it["kind"] in ("positive", "sibling_positive") and it["answer"]["bbox_2d"] is not None and max(rewards) < 1.0 and bool(it.get("matrix"))  # answer-only items (RefCOCO) have no label trace
                if args.inject_gt == "all" or (args.inject_gt in ("neg", "both") and need_neg) or (args.inject_gt == "both" and need_pos):
                    if args.turns == 2 and args.audit == "coa":
                        gt_segs, gt_ro = GC.label_rollout(processor, template, coa_tmpl, ans_tmpl, it, image, inputs, args, tok, im_end, model.device)
                        rb, ra, gt_t, gi = GC.coa_reward(gt_ro, it, args)
                        gi["r_base"], gi["r_acc"] = rb, ra
                        infos.append(gi)
                        gt_r = rb + ra
                    elif args.turns == 2:
                        t1, lboxes, t2 = TR.render_turns(it["matrix"], it["image_wh"], args.trace, args.order)
                        c1 = tok(t1, add_special_tokens=False)["input_ids"]
                        c2 = tok(t2, add_special_tokens=False)["input_ids"] + [im_end]
                        inputs2 = turn2_inputs(processor, template, it, image, t1, lboxes, model.device)
                        gt_segs = [(inputs, c1), (inputs2, c2)]
                        gt_r, gt_t, gi = trace_reward(t1 + "\n" + t2, it, args, s_pos.get(it["group"]) if not is_cross else None)
                        gt_r += candidate_reward(lboxes, it, args, gi)
                    else:
                        gt_text = it["trace"] if args.trace else D.target_text(it)
                        gt_comp = tok(gt_text, add_special_tokens=False)["input_ids"] + [im_end]
                        gt_segs = [(inputs, gt_comp)]
                        if args.trace:
                            gt_r, gt_t, _ = trace_reward(processor.decode(gt_comp, skip_special_tokens=True), it, args, s_pos.get(it["group"]) if not is_cross else None)
                        else:
                            gt_r, gt_t = reward_of(processor.decode(gt_comp, skip_special_tokens=True), it, args)
                    if gt_t == "bad":
                        print(f"  !! label completion unparsable for {it['id']}", flush=True)
                    if max(rewards) >= gt_r:
                        gt_r = sum(rewards) / len(rewards)  # ViSurf smoothing: no push when the policy already gets it
                    segs.append(gt_segs)
                    rewards.append(gt_r)
                if max(rewards) - min(rewards) < 1e-6:
                    continue  # no signal in this group
                mean_r = sum(rewards) / len(rewards)
                std_r = statistics.pstdev(rewards) + 1e-4
                scale = args.neg_adv_scale if it["kind"] == "negative" else 1.0
                n_used += 1
                window[-1]["used"] = 1.0
                if rollouts is not None:  # per-segment advantages: r_base on every segment, r_acc on the labelled candidate's audit
                    mean_base = sum(i["r_base"] for i in infos) / len(infos)
                    mean_acc = sum(i["r_acc"] for i in infos) / len(infos)
                for gi_, (seg_list, r) in enumerate(zip(segs, rewards)):
                    adv = scale * (r - mean_r) / (1.0 if args.no_std else std_r)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        stats = [completion_stats(model, inp, comp) for inp, comp in seg_list]
                        logp = torch.cat([lp for lp, _ in stats])
                    step_ent.append(sum(e for _, e in stats) / len(stats))
                    step_len.append(sum(len(comp) for _, comp in seg_list))
                    if rollouts is not None:
                        info_g = infos[gi_]
                        adv_b = scale * (info_g["r_base"] - mean_base)
                        adv_a = scale * (info_g["r_acc"] - mean_acc)
                        acc_seg = info_g.get("acc_seg")
                        loss = torch.zeros((), device=logp.device)
                        for si, (lp, _) in enumerate(stats):
                            a_si = adv_b + (adv_a if (acc_seg is not None and si == acc_seg) else 0.0)
                            loss = loss - a_si * lp.sum() / float(args.norm_tokens)
                    else:
                        loss = -(adv * (logp.sum() / float(args.max_new) if args.token_level else logp.mean()))  # DAPO-style token-level vs sequence-mean
                    if args.beta > 0:
                        model.set_adapter("ref")
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            ref = torch.cat([completion_logprobs(model, inp, comp) for inp, comp in seg_list]).detach()
                        model.set_adapter("default")
                        d = ref - logp
                        loss = loss + args.beta * (torch.exp(d) - d - 1).mean()
                    (loss / (len(segs) * len(batch))).backward()
            if n_used:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            with open(out / "train_log.jsonl", "a", encoding="utf-8") as tl:  # structured per-step record (this step's prompts)
                recent = window[-len(batch):] if window else []
                by = lambda k: [w for w in recent if w["kind"] == k]
                mean = lambda ws, key: (lambda v: (sum(v) / len(v)) if v else None)([w[key] for w in ws if w.get(key) is not None])  # skip prompts without the metric (e.g. cell_acc on RefCOCO items)
                tl.write(json.dumps({"kind": "grpo", "epoch": epoch + 1, "step": step, "total": total_steps, "n_prompts": len(batch), "n_used": n_used,
                                     "pos_r": mean(by("positive") + by("sibling_positive"), "mean_r"), "pos_null": mean(by("positive") + by("sibling_positive"), "null_frac"),
                                     "pos_fa": mean(by("positive") + by("sibling_positive"), "fa"), "neg_r": mean(by("negative"), "mean_r"), "neg_null": mean(by("negative"), "null_frac"),
                                     "neg_ev": mean(by("negative"), "ev"), "cross_r": mean(by("cross"), "mean_r"), "cross_null": mean(by("cross"), "null_frac"),
                                     "bad": mean(recent, "bad_frac"), "inc": mean(recent, "inc"), "cand_recall": mean(recent, "rec"), "reason_named": mean(by("negative"), "rsn"), "cell_acc": mean(recent, "cell"),
                                     "k": mean(recent, "kk"), "consistent": mean(recent, "cons"), "acc_correct": mean(by("negative"), "acc"), "sycophantic": mean(by("negative"), "syc"), "unsure": mean(recent, "uns"), "echo": mean(recent, "echo"),
                                     "entropy": (sum(step_ent) / len(step_ent)) if step_ent else None, "comp_len": (sum(step_len) / len(step_len)) if step_len else None,
                                     "min": round((time.time() - t0) / 60, 2)}) + "\n")
            if step % args.log_every == 0:
                neg = [w for w in window if w["kind"] == "negative"]
                crs = [w for w in window if w["kind"] == "cross"]
                pos = [w for w in window if w["kind"] in ("positive", "sibling_positive")]
                f = lambda xs, k: (sum(x[k] for x in xs) / len(xs)) if xs else float("nan")
                print(f"  epoch {epoch + 1} step {step}/{total_steps} | neg: r {f(neg, 'mean_r'):.2f} null {f(neg, 'null_frac'):.2f} ev {f(neg, 'ev'):.2f} used {f(neg, 'used'):.2f} | "
                      f"cross: r {f(crs, 'mean_r'):.2f} null {f(crs, 'null_frac'):.2f} | "
                      f"pos: r {f(pos, 'mean_r'):.2f} null {f(pos, 'null_frac'):.2f} fa {f(pos, 'fa'):.2f} used {f(pos, 'used'):.2f} | "
                      f"bad {f(pos + neg + crs, 'bad_frac'):.2f} inc {f(pos + neg + crs, 'inc'):.2f} | {(time.time() - t0) / 60:.1f} min", flush=True)
                window = []
            if args.save_every and step % args.save_every == 0 and step < total_steps:
                model.save_pretrained(out / f"adapter_step{step}", selected_adapters=["default"])  # screen mid-run checkpoints (50 vs 100 steps) without a second run
                print(f"  checkpoint saved: {out / f'adapter_step{step}'}", flush=True)
            if step % args.eval_every == 0:
                log_eval(step, {})
                model.save_pretrained(out / "adapter", selected_adapters=["default"])
    if step % args.eval_every != 0:
        log_eval(step, {})
    model.save_pretrained(out / "adapter", selected_adapters=["default"])
    curve.close()
    print(f"saved {out / 'adapter'}; {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
