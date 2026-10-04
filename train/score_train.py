"""Supervised scoring stage (notes/DESIGN-SCORE-SELECT-2026-10-04.md): labels act on the verdict probabilities of the model's own audits.

    python -m train.score_train gen   --adapter <dir> --out <audits.jsonl> [--half]
    python -m train.score_train train --name score3k --init-adapter <dir> --audits <audits.jsonl> [--half]
    python -m train.score_train holistic --name hol3k --init-adapter <dir> [--half]      # control: one yes / no per candidate

gen     the model writes, for every item of train.score_data, its conditions (turn 1, greedy) and one audit per candidate box
        (greedy), exactly as at inference.  Resumable.
train   every audit is replayed with teacher forcing on the model's own text; at each line's verdict token the probabilities of
        " match" / " mismatch" / " unsure" are read and trained with three losses, each used where its label exists:
          line       cross-entropy to the label verdict (described instance, rule-flipped word, box geometry, all-true lines)
          candidate  the candidate passes iff no line is a mismatch: p_pass = prod_j (1 - p_mismatch_j); binary cross-entropy to
                     "must pass" / "must be vetoed" (a noisy-OR over the lines, so an unlabelled veto finds its own line)
          item       softmax over the candidates' pass logits and a null option at logit 0; target = the answer or the null.
                     Computed in two passes (scores without gradient, then one graph per candidate with d loss / d score
                     fixed), so only one audit is on the GPU at a time.
Output: <name>/adapter, train_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch

from shared.harness import prompts
from train import coa as COA
from train import data as D
from train import grpo_coa as GC
from train import traces as TR

ITEMS = D.TRAIN_ROOT / "score_items.jsonl"


def load_items(half: bool) -> list[dict]:
    its = D.load_items(ITEMS)
    return [i for i in its if i["half"]] if half else its


# --- gen ---------------------------------------------------------------------


@torch.no_grad()
def gen(args) -> None:
    import train.eval_suite as E

    items = load_items(args.half)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.is_file() else set()
    todo = [i for i in items if i["id"] not in done]
    if args.limit:
        todo = todo[: max(0, args.limit - len(done))]
    print(f"gen: {len(items)} items, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    model, processor = E.load_model(args.model, args.adapter)
    tok = processor.tokenizer
    eos = {t for t in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if t is not None}
    template = prompts.load("grounding_verify_trace_tool", non_kill=True)
    coa_tmpl = prompts.load("grounding_coa_audit", non_kill=True)
    t0, n = time.time(), 0
    for b0 in range(0, len(todo), args.batch):
        chunk = todo[b0: b0 + args.batch]
        ev = [{"id": i["id"], "expr": i["expression"], "image": i["image"], "image_wh": i["image_wh"]} for i in chunk]
        t1s, _, _ = E.generate_batch(model, processor, template, ev, args.max_new1, want_scores=False, stop="</tool_call>")
        convs, imgs, owner = [], [], []
        for it, t1 in zip(chunk, t1s):
            image = D.open_image(it)
            for k, c in enumerate(it["cands"]):
                text, ims = GC.audit_conv(processor, template, coa_tmpl, it, image, t1, c["box"], tuple(it["image_wh"]))
                convs.append(text)
                imgs.append(ims)
                owner.append((it["id"], k))
        toks = GC._gen_batched(model, processor, convs, imgs, args.max_new, 1.0, eos, args.batch, tok, sample=False)
        auds: dict[str, dict[int, str]] = {}
        for (iid, k), tk in zip(owner, toks):
            auds.setdefault(iid, {})[k] = processor.decode(tk, skip_special_tokens=True)
        with open(out, "a", encoding="utf-8") as fh:
            for it, t1 in zip(chunk, t1s):
                fh.write(json.dumps({"id": it["id"], "t1": t1, "audits": [auds.get(it["id"], {}).get(k, "") for k in range(len(it["cands"]))]}, ensure_ascii=False) + "\n")
        n += len(chunk)
        if (b0 // args.batch) % 10 == 0:
            torch.cuda.empty_cache()
            rate = (time.time() - t0) / n
            print(f"  [{n}/{len(todo)}] {rate:.2f}s/item, {(len(todo) - n) * rate / 60:.0f} min left", flush=True)
    print(f"gen done: {out}", flush=True)


# --- labels ------------------------------------------------------------------


def line_labels(cand: dict, lines: list[dict]) -> list[int | None]:
    """Per parsed audit line: 0 = match, 1 = mismatch, None = no label."""
    spec = cand.get("lines")
    if spec is None:
        return [None] * len(lines)
    if spec == "all_true":
        return [0] * len(lines)
    if "false_word" in spec:
        hit = [bool(re.search(rf"\b{re.escape(spec['false_word'])}\b", ln["claimed"], re.I)) for ln in lines]
        if sum(hit) != 1:  # the flipped word is not in exactly one condition: no line labels (the candidate label still applies)
            return [None] * len(lines)
        return [1 if h else 0 for h in hit]
    clauses = list(spec["verdicts"])
    out = []
    for ln in lines:
        j = TR.align(clauses, ln["claimed"])
        v = spec["verdicts"][clauses[j - 1]] if j is not None else None
        out.append({"yes": 0, "no": 1}.get(v))
    return out


def verdict_positions(comp: list[int], ids: list[int], bar: int) -> list[int]:
    vs = set(ids)
    return [t for t in range(1, len(comp)) if comp[t] in vs and comp[t - 1] == bar]


# --- train -------------------------------------------------------------------


def forward_verdicts(model, inp, comp: list[int], pos: list[int], ids: list[int]):
    """Log-probabilities (full vocabulary) of the three verdict words at the verdict positions of a teacher-forced audit: [L, 3]."""
    comp_t = torch.tensor([comp], device=model.device)
    full = torch.cat([inp["input_ids"], comp_t], dim=1)
    kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
    if "mm_token_type_ids" in inp:
        mtt = inp["mm_token_type_ids"]
        kw["mm_token_type_ids"] = torch.cat([mtt, torch.zeros((1, len(comp)), dtype=mtt.dtype, device=mtt.device)], dim=1)
    n_prompt = inp["input_ids"].shape[1]
    idx = torch.tensor([n_prompt + t - 1 for t in pos], device=model.device)  # the logits at position p predict the token at p + 1
    out = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False, logits_to_keep=idx, **kw)
    return torch.log_softmax(out.logits[0].float(), dim=-1)[:, ids]


def pass_logit(lp3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(log p_pass, pass logit) of a candidate from its lines' [match, mismatch, unsure] log-probabilities."""
    p3 = torch.softmax(lp3, dim=-1)  # renormalised over the three verdicts
    log_pass = torch.log1p(-p3[:, 1].clamp(max=1 - 1e-6)).sum()
    log_veto = torch.log(-torch.expm1(log_pass.clamp(max=-1e-6)))
    return log_pass, log_pass - log_veto


def train(args) -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup

    import train.eval_suite as E
    from shared.harness import tokens as T

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = D.TRAIN_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    auds = {r["id"]: r for r in (json.loads(l) for l in open(Path(args.audits).expanduser(), encoding="utf-8") if l.strip())}
    items = [i for i in load_items(args.half) if i["id"] in auds]
    if args.limit:
        items = items[: args.limit]
    print(f"train: {len(items)} items with audits, {sum(len(i['cands']) for i in items)} candidates", flush=True)

    hf, rev = D.MODELS[args.model]
    processor = T.load_capped_processor(hf, rev)
    T.assert_cap_is_in_force(processor)
    tok = processor.tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    ids, bar = E._verdict_ids(tok)
    assert ids and bar is not None, "verdict words are not single tokens"
    model = AutoModelForImageTextToText.from_pretrained(hf, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, str(Path(args.init_adapter).expanduser()), is_trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    assert not [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    template = prompts.load("grounding_verify_trace_tool", non_kill=True)
    coa_tmpl = prompts.load("grounding_coa_audit", non_kill=True)
    n_cand = sum(len(i["cands"]) for i in items)
    total_steps = max(1, math.ceil(n_cand * args.epochs / args.grad_accum))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total_steps)), total_steps)
    model.train()
    t0, micro, step, saved = time.time(), 0, 0, 0
    win: list[dict] = []
    for epoch in range(args.epochs):
        order = list(items)
        rng.shuffle(order)
        for it in order:
            rec = auds[it["id"]]
            image = D.open_image(it)
            wh = tuple(it["image_wh"])
            prepared = []
            for c, a_text in zip(it["cands"], rec["audits"]):
                pa = COA.parse_audit(a_text)
                comp = tok(a_text, add_special_tokens=False)["input_ids"] + [im_end]
                pos = verdict_positions(comp, ids, bar)
                if not pa["format_ok"] or not pos or len(pos) != len(pa["lines"]):
                    prepared.append(None)  # no usable verdict tokens in this audit
                    continue
                text, ims = GC.audit_conv(processor, template, coa_tmpl, it, image, rec["t1"], c["box"], wh)
                inp = processor(images=ims, text=text, return_tensors="pt").to(model.device)
                prepared.append((c, inp, comp, pos, line_labels(c, pa["lines"])))
            usable = [k for k, p in enumerate(prepared) if p is not None]
            if not usable:
                continue
            # pass 1 (no gradient): pass logits of all candidates, for the item-level softmax with a null option at logit 0
            grad_s = {}
            if args.w_list > 0 and len(usable) == len(prepared) and len(prepared) >= 2:
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    s0 = [float(pass_logit(forward_verdicts(model, p[1], p[2], p[3], ids))[1]) for p in prepared]
                model.train()
                pi = torch.softmax(torch.tensor(s0 + [0.0]), dim=0)
                tgt = it["answer"] if it["answer"] is not None else len(s0)
                grad_s = {k: float(pi[k]) - float(k == tgt) for k in range(len(s0))}  # d(-log pi_target) / d s_k
            # pass 2: one graph per candidate
            for k in usable:
                c, inp, comp, pos, labs = prepared[k]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp3 = forward_verdicts(model, inp, comp, pos, ids)
                log_pass, s = pass_logit(lp3)
                l_cand = -log_pass if c["pass"] else -torch.log(-torch.expm1(log_pass.clamp(max=-1e-6)))
                lab_idx = [(j, lab) for j, lab in enumerate(labs) if lab is not None]
                l_line = -torch.stack([lp3[j, lab] for j, lab in lab_idx]).mean() if lab_idx else torch.zeros((), device=lp3.device)
                loss = args.w_cand * l_cand + args.w_line * l_line
                if k in grad_s:
                    loss = loss + args.w_list * grad_s[k] * s
                (loss / args.grad_accum).backward()
                p_pass = float(torch.exp(log_pass.detach()))
                win.append({"pass": c["pass"], "p_pass": p_pass, "l_cand": float(l_cand.detach()), "l_line": float(l_line.detach()) if lab_idx else None,
                            "line_acc": (sum(int(torch.argmax(lp3[j].detach()) == lab) for j, lab in lab_idx) / len(lab_idx)) if lab_idx else None})
                micro += 1
                if micro % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % args.log_every == 0:
                        m = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
                        row = {"kind": "score", "epoch": epoch + 1, "step": step, "total": total_steps,
                               "p_pass_pos": m([w["p_pass"] for w in win if w["pass"]]), "p_pass_neg": m([w["p_pass"] for w in win if not w["pass"]]),
                               "l_cand": m([w["l_cand"] for w in win]), "l_line": m([w["l_line"] for w in win if w["l_line"] is not None]),
                               "line_acc": m([w["line_acc"] for w in win if w["line_acc"] is not None]), "lr": sched.get_last_lr()[0],
                               "min": round((time.time() - t0) / 60, 2)}
                        with open(out / "train_log.jsonl", "a", encoding="utf-8") as tl:
                            tl.write(json.dumps(row) + "\n")
                        print(f"  step {step}/{total_steps} | pass prob: must-pass {row['p_pass_pos']:.3f} must-veto {row['p_pass_neg']:.3f} | "
                              f"line acc {row['line_acc']:.3f} | cand loss {row['l_cand']:.3f} line loss {row['l_line']:.3f} | {row['min']:.1f} min", flush=True)
                        win = []
            if args.save_every and step // args.save_every > saved:
                saved = step // args.save_every
                model.save_pretrained(out / f"adapter_step{saved * args.save_every}")
    model.save_pretrained(out / "adapter")
    print(f"saved {out / 'adapter'}; {(time.time() - t0) / 60:.1f} min", flush=True)


def train_holistic(args) -> None:
    """Control for the claim-level stage: the same items, candidates, views and labels, but one yes / no per candidate
    (prompt grounding_holistic_verify) instead of one verdict per condition.  Candidate loss = cross-entropy to yes / no at
    the answer token; item loss = the same softmax over pass logits (log p(yes) - log p(no)) with a null option at 0."""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup

    import train.eval_suite as E
    from shared.harness import tokens as T

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = D.TRAIN_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    items = load_items(args.half)
    if args.limit:
        items = items[: args.limit]
    n_cand = sum(len(i["cands"]) for i in items)
    print(f"holistic: {len(items)} items, {n_cand} candidates", flush=True)
    hf, rev = D.MODELS[args.model]
    processor = T.load_capped_processor(hf, rev)
    T.assert_cap_is_in_force(processor)
    tok = processor.tokenizer
    y_ids, n_ids = E.yes_no_ids(tok)
    model = AutoModelForImageTextToText.from_pretrained(hf, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, str(Path(args.init_adapter).expanduser()), is_trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    tmpl = prompts.load("grounding_holistic_verify", non_kill=True)
    total_steps = max(1, math.ceil(n_cand * args.epochs / args.grad_accum))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total_steps)), total_steps)

    def yes_no(inp):
        lp = torch.log_softmax(model(**inp, logits_to_keep=1).logits[0, -1].float(), dim=-1)
        return torch.logsumexp(lp[y_ids], dim=0), torch.logsumexp(lp[n_ids], dim=0)

    model.train()
    t0, micro, step = time.time(), 0, 0
    win: list[dict] = []
    for epoch in range(args.epochs):
        order = list(items)
        rng.shuffle(order)
        for it in order:
            image = D.open_image(it)
            inps = []
            for c in it["cands"]:
                text, ims = E.holistic_conv(processor, tmpl, it["expression"], image, c["box"])
                inps.append(processor(images=ims, text=text, return_tensors="pt").to(model.device))
            grad_s = {}
            if args.w_list > 0 and len(inps) >= 2:
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    s0 = [float(a - b) for a, b in (yes_no(inp) for inp in inps)]
                model.train()
                pi = torch.softmax(torch.tensor(s0 + [0.0]), dim=0)
                tgt = it["answer"] if it["answer"] is not None else len(s0)
                grad_s = {k: float(pi[k]) - float(k == tgt) for k in range(len(s0))}
            for k, (c, inp) in enumerate(zip(it["cands"], inps)):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ly, ln = yes_no(inp)
                l_cand = -ly if c["pass"] else -ln
                loss = args.w_cand * l_cand
                if k in grad_s:
                    loss = loss + args.w_list * grad_s[k] * (ly - ln)
                (loss / args.grad_accum).backward()
                win.append({"pass": c["pass"], "p_pass": float(torch.sigmoid((ly - ln).detach())), "l_cand": float(l_cand.detach())})
                micro += 1
                if micro % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % args.log_every == 0:
                        m = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
                        row = {"kind": "holistic", "epoch": epoch + 1, "step": step, "total": total_steps,
                               "p_pass_pos": m([w["p_pass"] for w in win if w["pass"]]), "p_pass_neg": m([w["p_pass"] for w in win if not w["pass"]]),
                               "l_cand": m([w["l_cand"] for w in win]), "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 2)}
                        with open(out / "train_log.jsonl", "a", encoding="utf-8") as tl:
                            tl.write(json.dumps(row) + "\n")
                        print(f"  step {step}/{total_steps} | p(yes): must-pass {row['p_pass_pos']:.3f} must-veto {row['p_pass_neg']:.3f} | "
                              f"cand loss {row['l_cand']:.3f} | {row['min']:.1f} min", flush=True)
                        win = []
    model.save_pretrained(out / "adapter")
    print(f"saved {out / 'adapter'}; {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--adapter", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--model", default="4b")
    g.add_argument("--half", action="store_true", help="only the half-size subset (3,000 items)")
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--max-new1", type=int, default=400)
    g.add_argument("--max-new", type=int, default=400)
    g.add_argument("--limit", type=int, default=None, help="stop after this many items in the output file (smoke test)")
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--init-adapter", required=True)
    t.add_argument("--audits", required=True)
    t.add_argument("--model", default="4b")
    t.add_argument("--half", action="store_true")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--grad-accum", type=int, default=8, help="candidates per optimizer step")
    t.add_argument("--w-line", type=float, default=1.0)
    t.add_argument("--w-cand", type=float, default=1.0)
    t.add_argument("--w-list", type=float, default=0.5)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--save-every", type=int, default=0)
    h = sub.add_parser("holistic")
    h.add_argument("--name", required=True)
    h.add_argument("--init-adapter", required=True)
    h.add_argument("--model", default="4b")
    h.add_argument("--half", action="store_true")
    h.add_argument("--epochs", type=int, default=1)
    h.add_argument("--lr", type=float, default=5e-5)
    h.add_argument("--grad-accum", type=int, default=8)
    h.add_argument("--w-cand", type=float, default=1.0)
    h.add_argument("--w-list", type=float, default=0.5)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--limit", type=int, default=None)
    h.add_argument("--log-every", type=int, default=10)
    a = ap.parse_args()
    {"gen": gen, "train": train, "holistic": train_holistic}[a.cmd](a)
