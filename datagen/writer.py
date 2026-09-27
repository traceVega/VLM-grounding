"""Stages `write`, `sibling`, `blind`: Qwen3.5-9B (judge_a) as writer, decomposer and text-only judge.

Thinking is switched off (`enable_thinking=False`); greedy decoding.  The writer
sees the image with the target outlined in red; the flip step sees the plain
image so the replacement can be chosen to be false of every instance; the
decomposer and the blind judge see no image.
"""

from __future__ import annotations

import time

import torch
from PIL import Image

from datagen import common as C
from shared.harness import prompts

HF = "Qwen/Qwen3.5-9B"
REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


class Qwen35:
    def __init__(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t0 = time.time()
        self.proc = AutoProcessor.from_pretrained(HF, revision=REV)
        self.model = AutoModelForImageTextToText.from_pretrained(
            HF, revision=REV, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"Qwen3.5-9B loaded in {time.time() - t0:.0f}s, "
              f"{torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    def ask(self, text: str, image=None, max_new_tokens: int = 160) -> str:
        images = [] if image is None else (list(image) if isinstance(image, (list, tuple)) else [image])
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        chat = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        if images:
            inputs = self.proc(images=images, text=chat, return_tensors="pt")
        else:
            inputs = self.proc(text=chat, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _flip_pairs(e_t: str, clauses: list[str], old_span, news) -> list[tuple[str, str, str]]:
    """(old_span, new_span, expr_neg) for every NEW value that substitutes cleanly, after
    aligning clause-form answers to the expression (common.align_span)."""
    out = []
    for nw in news or []:
        al = C.align_span(e_t, clauses, old_span, nw)
        if not al:
            continue
        c = C.substitute_once(e_t, al[0], al[1])
        if c:
            out.append((al[0], al[1], c))
    return out


def _scene_image(scene: dict) -> Image.Image:
    return Image.open(C.image_path(scene["image_id"])).convert("RGB")


def _inst(scene: dict, iid: int) -> dict:
    return next(i for i in scene["instances"] if i["iid"] == iid)


CLOSEUP = True  # v2 prompts: full scene with the outline plus a close-up crop of the target


def _views(image: Image.Image, box):
    """What the writer sees: the outlined scene, plus the close-up when CLOSEUP is on."""
    outlined = C.outline(image, box, "red")
    return [outlined, C.closeup(image, box)] if CLOSEUP else outlined


def run_write(run: str, limit: int | None) -> None:
    root = C.run_root(run)
    scenes = C.read_jsonl(root / "scenes.jsonl")
    out = root / "write.jsonl"
    done = C.done_keys(out)
    todo = [s for s in scenes if (s["image_id"],) not in done]
    if limit:
        todo = todo[:limit]
    print(f"write: {len(scenes)} scenes, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    p_write = prompts.load("expr_write_clausal_v2" if CLOSEUP else "expr_write_clausal", non_kill=True)
    p_decomp = prompts.load("expr_decompose", non_kill=True)
    p_flip = prompts.load("expr_flip_clause_v2", non_kill=True)
    m = Qwen35()
    t0 = time.time()
    for n, sc in enumerate(todo, 1):
        rng = C.seeded(sc["image_id"], "write")
        n_clauses = rng.choice(C.N_CLAUSES_CHOICES)
        image = _scene_image(sc)
        target = _inst(sc, sc["target_iid"])
        raw_t = m.ask(p_write.render(category=sc["category"], n_siblings=sc["n_candidates"],
                                     n_clauses=n_clauses), _views(image, target["box"]), 120)
        e_t = C.clean_line(raw_t)
        raw_d = m.ask(p_decomp.render(expr=e_t), None, 160)
        head, clauses = C.parse_decomposition(raw_d)
        # v2 flip: the model names an exact OLD span and three NEW values; the code
        # substitutes, so the negative differs from the target in exactly one span.
        clause_list = "\n".join(f"- {c}" for c in clauses)
        raw_f = m.ask(p_flip.render(expr=e_t, category=sc["category"], n_siblings=sc["n_candidates"],
                                    clauses=clause_list), image, 160)
        old_span, news = C.parse_flip_v2(raw_f)
        flip_retried = False
        pairs = _flip_pairs(e_t, clauses, old_span, news)
        if not pairs:
            flip_retried = True
            raw_f = m.ask(p_flip.render(expr=e_t, category=sc["category"], n_siblings=sc["n_candidates"],
                                        clauses=clause_list)
                          + "\n\nOLD must be an exact substring of the expression; copy it verbatim.",
                          image, 160)
            old_span, news = C.parse_flip_v2(raw_f)
            pairs = _flip_pairs(e_t, clauses, old_span, news)
        flip_failed = not pairs
        e_neg = pairs[0][2] if pairs else ""
        ch_from, ch_to = (pairs[0][0], pairs[0][1]) if pairs else (old_span, None)  # the spans actually substituted
        neg_alternatives = [{"expr_neg": c, "changed_from": o, "changed_to": nw} for o, nw, c in pairs]
        raw_dn = m.ask(p_decomp.render(expr=e_neg), None, 160)
        _, clauses_neg = C.parse_decomposition(raw_dn)
        fidx = C.flipped_index(clauses, clauses_neg)
        if fidx is None and ch_from:
            hits = [i for i, c in enumerate(clauses) if ch_from.lower() in c.lower()]
            fidx = hits[0] if len(hits) == 1 else None
        if (fidx is not None and fidx < len(clauses_neg) and fidx < len(clauses)
                and clauses_neg[fidx].strip().lower() == clauses[fidx].strip().lower()):
            flip_failed = True  # the "flipped" clause is unchanged: the checker would verify the original detail
        if flip_failed:
            fidx = None
        rec = {
            "image_id": sc["image_id"], "n_clauses_asked": n_clauses,
            "flip_failed": flip_failed, "flip_retried": flip_retried,
            "expr_target": e_t, "head": head, "clauses": clauses,
            "expr_neg": e_neg, "changed_from": ch_from, "changed_to": ch_to,
            "clauses_neg": clauses_neg, "flipped_idx": fidx, "neg_alternatives": neg_alternatives,
            "clause_neg": clauses_neg[fidx] if fidx is not None and fidx < len(clauses_neg) else None,
            "raw": {"write": raw_t, "decompose": raw_d, "flip": raw_f, "decompose_neg": raw_dn},
            "writer": f"{HF}@{REV[:8]}", "prompt": p_write.name,
        }
        C.append_jsonl(out, rec)
        if n % 5 == 0 or n == len(todo):
            rate = (time.time() - t0) / n
            print(f"  [{n}/{len(todo)}] {rate:.1f}s/scene | {e_t[:90]}", flush=True)


def run_sibling(run: str, limit: int | None) -> None:
    root = C.run_root(run)
    scenes = C.by_image(C.read_jsonl(root / "scenes.jsonl"))
    written = C.by_image(C.read_jsonl(root / "write.jsonl"))
    pol = {(r["image_id"], r["which"]): r for r in C.read_jsonl(root / "policy.jsonl")}
    out = root / "sibling.jsonl"
    done = C.done_keys(out)
    todo = [iid for iid in written if (iid,) not in done and (iid, "target") in pol]
    if limit:
        todo = todo[:limit]
    print(f"sibling: {len(written)} written, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    p_sib = prompts.load("expr_write_sibling_v2" if CLOSEUP else "expr_write_sibling", non_kill=True)
    p_decomp = prompts.load("expr_decompose", non_kill=True)
    m = Qwen35()
    t0 = time.time()
    for n, image_id in enumerate(todo, 1):
        sc, w = scenes[image_id], written[image_id]
        boxed = pol[(image_id, "target")].get("boxed_iid")
        others = [i for i in sc["instances"] if not i["is_target"]]
        if boxed is not None and boxed != sc["target_iid"]:
            s_iid, reason = boxed, "base boxed this sibling on the target expression"
        else:
            s_iid, reason = max(others, key=lambda i: i["area"])["iid"], "largest sibling (base boxed the target)"
        image = _scene_image(sc)
        n_clauses = len(w["clauses"]) or w["n_clauses_asked"]
        raw = m.ask(p_sib.render(category=sc["category"], n_siblings=sc["n_candidates"],
                                 n_clauses=n_clauses, expr_target=w["expr_target"]),
                    _views(image, _inst(sc, s_iid)["box"]), 120)
        e_s = C.clean_line(raw)
        raw_d = m.ask(p_decomp.render(expr=e_s), None, 160)
        _, clauses_s = C.parse_decomposition(raw_d)
        C.append_jsonl(out, {"image_id": image_id, "sibling_iid": s_iid, "sibling_reason": reason,
                             "expr_sibling": e_s, "clauses_sibling": clauses_s,
                             "raw": {"write": raw, "decompose": raw_d}})
        if n % 5 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {(time.time() - t0) / n:.1f}s/scene | {e_s[:90]}", flush=True)


def run_blind(run: str, limit: int | None) -> None:
    root = C.run_root(run)
    written = C.read_jsonl(root / "write.jsonl")
    out = root / "blind.jsonl"
    done = C.done_keys(out)
    todo = [w for w in written if (w["image_id"],) not in done and w["expr_neg"]]
    if limit:
        todo = todo[:limit]
    print(f"blind: {len(written)} written, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    p = prompts.load("blind_pair_judge", non_kill=True)
    m = Qwen35()
    for n, w in enumerate(todo, 1):
        rng = C.seeded(w["image_id"], "blind")
        neg_is_a = rng.random() < 0.5
        a, b = (w["expr_neg"], w["expr_target"]) if neg_is_a else (w["expr_target"], w["expr_neg"])
        raw = m.ask(p.render(a=a, b=b), None, 8)
        letter = (C.first_word(raw) or "")[:1].upper()
        picked_neg = (letter == "A") == neg_is_a if letter in ("A", "B") else None
        C.append_jsonl(out, {"image_id": w["image_id"], "neg_is_a": neg_is_a, "raw": raw,
                             "judge_letter": letter, "judge_found_flip": picked_neg})
    rows = C.read_jsonl(out)
    found = [r["judge_found_flip"] for r in rows if r["judge_found_flip"] is not None]
    if found:
        print(f"blind judge finds the altered sentence {sum(found)}/{len(found)} = "
              f"{sum(found) / len(found):.0%} (chance 50%)", flush=True)
