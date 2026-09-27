"""Alternative flipped negatives (more minimal pairs per scene, no API): the writer proposed up
to three replacement values for the flipped span (write.jsonl `neg_alternatives`) and the
primary export used one.  Verify the others locally with Qwen3-VL-8B on every candidate
instance (outline + close-up, batched clause prompt, the Gemini checker's protocol) and export
the usable ones as extra negatives.

    python -m train.alt_negatives check [--limit N]   -> $VLMG_DATA_ROOT/train/alt_check.jsonl (resumable)
    python -m train.alt_negatives export              -> $VLMG_DATA_ROOT/train/alt_negatives.jsonl (item format, source "alt")

Usable iff the intended object is judged "no" on the alternative clause and no other instance
satisfies the whole altered sentence (alternative clause not "no" and every other clause not
"no" in the primary Gemini matrix).  Items carry `flipped` and `alt_verdicts` so train.traces
can render their label traces.
"""

from __future__ import annotations

import argparse
import json
import time

from datagen import common as C
from train import data as D
from train import traces as TR

CHECK = D.TRAIN_ROOT / "alt_check.jsonl"
OUT = D.TRAIN_ROOT / "alt_negatives.jsonl"


def scene_alts() -> list[dict]:
    """Scenes with an exported item and at least one alternative distinct from the primary negative."""
    exported = {}
    for it in D.load_items():
        exported.setdefault((it["run"], it["group"]), []).append(it)
    recs = TR.records()
    out = []
    for run in TR.RUNS:
        f = C.run_root(run) / "write.jsonl"
        if not f.is_file():
            continue
        for w in C.read_jsonl(f):
            key = (run, w["image_id"])
            rec = recs.get(key)
            if key not in exported or rec is None or w.get("flipped_idx") is None or not w.get("changed_from"):
                continue
            fidx, old = w["flipped_idx"], w["changed_from"]
            if fidx >= len(w["clauses"]) or old not in w["clauses"][fidx]:
                continue
            primary = (rec["expressions"].get("e_T_neg") or "").strip().lower()
            alts = []
            for a in w.get("neg_alternatives") or []:
                if not a.get("changed_to") or (a.get("expr_neg") or "").strip().lower() == primary:
                    continue
                alts.append({"expr_neg": a["expr_neg"], "changed_to": a["changed_to"], "clause": w["clauses"][fidx].replace(old, a["changed_to"], 1)})
            if alts:
                out.append({"run": run, "image_id": w["image_id"], "fidx": fidx, "changed_from": old, "clauses": w["clauses"], "alts": alts,
                            "category": rec["category"], "instances": rec["instances"], "target_iid": rec["target_iid"]})
    return out


def check(args) -> None:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText

    from datagen.checker_api import _shrink, parse_batched
    from shared.harness import prompts
    from shared.harness import tokens as T

    scenes = scene_alts()
    done = set()
    if CHECK.is_file():
        for l in open(CHECK, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add((r["run"], r["image_id"]))
    todo = [s for s in scenes if (s["run"], s["image_id"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"alt check: {len(scenes)} scenes, {sum(len(s['alts']) for s in scenes)} alternatives, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    tmpl = prompts.load("verifier_clauses_batched", non_kill=True)
    hf, rev = D.MODELS[args.model]
    processor = T.load_capped_processor(hf, rev)
    model = AutoModelForImageTextToText.from_pretrained(hf, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    t0 = time.time()
    for n, s in enumerate(todo, 1):
        full = Image.open(C.image_path(s["image_id"])).convert("RGB")
        small, sh = _shrink(full)
        items = "\n".join(f"{k + 1}. The {s['category']} {a['clause']}." for k, a in enumerate(s["alts"]))
        text = tmpl.render(category=s["category"], items=items)
        verdicts = {}
        for inst in s["instances"]:
            views = [C.outline(small, [v * sh for v in inst["box"]], "red"), C.closeup(full, inst["box"])]
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]
            chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=views, text=chat, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
            raw = processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            verdicts[str(inst["iid"])] = parse_batched(raw, len(s["alts"]))
        with open(CHECK, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run": s["run"], "image_id": s["image_id"], "fidx": s["fidx"], "changed_from": s["changed_from"],
                                 "alts": s["alts"], "verdicts": verdicts, "checker": f"{hf}@{rev[:8]}"}) + "\n")
        if n % 10 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {(time.time() - t0) / 60:.1f} min", flush=True)


def export(args) -> None:
    recs = TR.records()
    base = {}
    for it in D.load_items():
        base.setdefault((it["run"], it["group"]), it)
    n_alt = n_ok = 0
    items = []
    for l in open(CHECK, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        rec = recs.get((r["run"], r["image_id"]))
        b = base.get((r["run"], r["image_id"]))
        if rec is None or b is None:
            continue
        cv = rec.get("clause_verdicts") or {}
        t = rec["target_iid"]
        for k, a in enumerate(r["alts"]):
            n_alt += 1
            av = {iid: (vs[k] if k < len(vs) else "unparsed") for iid, vs in r["verdicts"].items()}
            if av.get(str(t)) != "no":
                continue
            satisfier = False
            for inst in rec["instances"]:
                iid = str(inst["iid"])
                if inst["iid"] == t:
                    continue
                others = [v for j, v in enumerate(cv.get(iid) or []) if j != r["fidx"]]
                if av.get(iid) != "no" and others and all(v != "no" for v in others):
                    satisfier = True
                    break
            if satisfier:
                continue
            n_ok += 1
            items.append({"image": b["image"], "image_wh": b["image_wh"], "group": b["group"], "category": b["category"], "label": b["label"],
                          "run": r["run"], "source": "alt", "n_candidates": b["n_candidates"], "id": f"{r['run']}:{r['image_id']}:alt{k}",
                          "kind": "negative", "expression": a["expr_neg"], "answer": {"bbox_2d": None}, "target_iid": None,
                          "flipped": {"idx": r["fidx"], "from": r["changed_from"], "to": a["changed_to"], "clause_neg": a["clause"]},
                          "alt_verdicts": av})
    with open(OUT, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{n_ok} usable of {n_alt} alternatives -> {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "export"])
    ap.add_argument("--model", default="8b", choices=list(D.MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    {"check": check, "export": export}[args.stage](args)


if __name__ == "__main__":
    main()
