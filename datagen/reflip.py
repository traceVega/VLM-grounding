"""Stage `reflip`: a second chance for negatives that failed for a recoverable reason.

Reasons, read from `records-<primary>.jsonl`:
  * flip_failed  the writer produced no usable e_T⁻ (empty, unchanged, or the flipped
                 clause could not be identified);
  * satisfied    some candidate satisfies the whole negative expression
                 (`zero_satisfier_sentence` is False).

For each such scene the stage tries, in order, the writer's stored alternative values,
then up to MAX_FRESH freshly generated flips that must change a detail not tried before.
Every candidate negative is verified on every instance by the batched API checker (one
statement per call) and accepted when no instance satisfies the flipped detail together
with all the other details (the report's whole-sentence rule, using the primary checker's
verdicts for the other details).  The blind judge is re-run on the accepted pair.

Output: `reflip.jsonl` (one row per scene tried).  Afterwards run
`policy --which neg2` (base model on the new negatives) and `report` again; the report
overlays accepted rows onto the scene record and marks it `reflipped`.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from datagen import checker_api as A
from datagen import common as C
from datagen.writer import Qwen35
from shared.harness import prompts

MAX_FRESH = 2
GRAMMAR_PROMPT = ("Is the following English sentence grammatical and coherent as a description of one object? "
                  "Contradictory details are fine; broken syntax is not. Reply with exactly one word: yes or no.\n\nSentence: {s}")


def grammatical(m, sentence: str) -> bool:
    return C.first_word(m.ask(GRAMMAR_PROMPT.format(s=sentence), None, 4)) in ("yes", "yeah", "yep")


def _hits(clauses: list[str], span: str) -> list[int]:
    pat = re.compile(r"(?<!\w)" + re.escape(span.strip()) + r"(?!\w)", re.IGNORECASE)
    return [i for i, c in enumerate(clauses) if pat.search(c)]


def _candidate(e_t: str, clauses: list[str], old: str | None, new: str | None):
    """(expr_neg, flipped_idx, clause_neg) for one OLD -> NEW substitution, or None."""
    al = C.align_span(e_t, clauses, old, new)
    if not al:
        return None
    old, new = al
    ow, nw = old.split(), new.split()
    if len(ow) > 3 or len(nw) > 3:
        return None  # a whole clause swapped for a new attribute reads as a contradiction, not a changed value (poc155: 23/53)
    if len(ow) != len(nw) and ow[0].lower() != nw[0].lower() and ow[-1].lower() != nw[-1].lower():
        return None  # 'white license plate' -> 'green' drops the noun ('displays a green'); poc155 round 2: 7/54
    e_neg = C.substitute_once(e_t, old, new)
    if not e_neg:
        return None
    hits = _hits(clauses, old)
    if len(hits) != 1:
        return None
    cl = C.substitute_once(clauses[hits[0]], old, new)
    if not cl or cl.strip().lower() == clauses[hits[0]].strip().lower():
        return None
    return e_neg, hits[0], cl


def run(run: str, limit: int | None, primary: str = "gemini-b", checker: str = "gemini-b") -> None:
    from google.genai import types

    root = C.run_root(run)
    sfx = "" if primary == "gemma4" else f"-{primary}"
    recs = C.by_image(C.read_jsonl(root / f"records{sfx}.jsonl"))
    scenes = C.by_image(C.read_jsonl(root / "scenes.jsonl"))
    written = C.by_image(C.read_jsonl(root / "write.jsonl"))
    out = root / "reflip.jsonl"
    done = C.done_keys(out)
    todo = []
    force_file = root / "reflip_force.txt"
    force = {l.strip() for l in force_file.read_text(encoding="utf-8").splitlines() if l.strip()} if force_file.is_file() else set()
    for iid, r in recs.items():
        if (iid,) in done:
            continue
        g = r["gates"]
        if iid in force:
            todo.append((iid, "flip_failed"))  # reviewer rejected the writer's negative: treat as failed
        elif not g.get("flip_identified"):
            todo.append((iid, "flip_failed"))
        elif g.get("zero_satisfier_sentence") is False:
            todo.append((iid, "satisfied"))
    if limit:
        todo = todo[:limit]
    print(f"reflip: {len(recs)} records, {len(done)} done, {len(todo)} to try "
          f"({sum(1 for _, r in todo if r == 'flip_failed')} flip_failed, {sum(1 for _, r in todo if r == 'satisfied')} satisfied)",
          flush=True)
    if not todo:
        return
    model, _, _ = A.MODELS[checker]
    client = A._client()
    tmpl = prompts.load("verifier_clauses_batched", non_kill=True)
    cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=512,
                                      thinking_config=types.ThinkingConfig(thinking_level="low"))
    p_flip = prompts.load("expr_flip_clause_v2", non_kill=True)
    p_blind = prompts.load("blind_pair_judge", non_kill=True)
    m = Qwen35()
    calls = 0

    def verify(sc: dict, clause: str) -> dict[str, str]:
        nonlocal calls
        full = Image.open(C.image_path(sc["image_id"])).convert("RGB")
        small, s = A._shrink(full)
        text = tmpl.render(category=sc["category"], items=f"1. The {sc['category']} {clause}.")

        def one(inst):
            views = [C.outline(small, [v * s for v in inst["box"]], "red"), C.closeup(full, inst["box"])]
            for attempt in range(A.RETRIES):
                try:
                    resp = client.models.generate_content(model=model, contents=[*views, text], config=cfg)
                    return A.parse_batched(resp.text or "", 1)[0]
                except Exception:
                    time.sleep(min(60, 2 ** attempt))
            return "unparsed"

        with ThreadPoolExecutor(max_workers=A.WORKERS) as ex:
            vs = list(ex.map(one, sc["instances"]))
        calls += len(vs)
        return {str(i["iid"]): v for i, v in zip(sc["instances"], vs)}

    def satisfiers(rec: dict, fidx: int, nv: dict[str, str]) -> list[int]:
        cv = rec["clause_verdicts"] or {}
        sat = []
        for iid_s, v in nv.items():
            if v not in ("yes", "unclear"):
                continue
            others = [x for k, x in enumerate(cv.get(iid_s) or []) if k != fidx]
            if all(x in ("yes", "unclear") for x in others):
                sat.append(int(iid_s))
        return sat

    t0 = time.time()
    for n, (iid, reason) in enumerate(todo, 1):
        sc, w, rec = scenes[iid], written[iid], recs[iid]
        e_t, clauses = w["expr_target"], w["clauses"]
        tried_spans = [w["changed_from"]] if w.get("changed_from") else []
        tried, raw_flips = [], []
        accepted = None

        def try_cands(cands):
            nonlocal accepted
            for old, new, src in cands:
                c = _candidate(e_t, clauses, old, new)
                if not c:
                    tried.append({"from": old, "to": new, "src": src, "result": "no_substitution"})
                    continue
                e_neg, fidx, cl = c
                if not grammatical(m, e_neg):
                    tried.append({"from": old, "to": new, "src": src, "result": "ungrammatical", "expr_neg": e_neg})
                    continue
                nv = verify(sc, cl)
                sat = satisfiers(rec, fidx, nv)
                tried.append({"from": old, "to": new, "src": src, "satisfiers": sat, "neg_verdicts": nv})
                if not sat:
                    old, new = C.align_span(e_t, clauses, old, new)  # the spans actually substituted
                    accepted = {"expr_neg": e_neg, "changed_from": old, "changed_to": new,
                                "flipped_idx": fidx, "clause_neg": cl, "neg_clause_verdicts": nv}
                    return

        used = (w.get("changed_to") or "").strip().lower()
        alts = [(a.get("changed_from") or w.get("changed_from"), a["changed_to"], "alternative") for a in (w.get("neg_alternatives") or [])
                if (a.get("changed_to") or "").strip().lower() != used or reason == "flip_failed"]
        try_cands(alts)
        fresh = 0
        if accepted is None:
            image = Image.open(C.image_path(iid)).convert("RGB")
            clause_list = "\n".join(f"- {c}" for c in clauses)
            while accepted is None and fresh < MAX_FRESH:
                fresh += 1
                avoid = ""
                spans = [t for t in tried_spans if t]
                if spans:
                    avoid = ("\n\nDo NOT change any of these spans, they were already tried: "
                             + "; ".join(f'"{t}"' for t in spans) + ". Pick a different detail of the expression.")
                raw = m.ask(p_flip.render(expr=e_t, category=sc["category"], n_siblings=sc["n_candidates"],
                                          clauses=clause_list) + avoid, image, 160)
                raw_flips.append(raw)
                old, news = C.parse_flip_v2(raw)
                if old:
                    tried_spans.append(old)
                try_cands([(old, nw, f"fresh{fresh}") for nw in news])
        blind = None
        if accepted:
            rng = C.seeded(iid, "blind-reflip")
            neg_is_a = rng.random() < 0.5
            a_, b_ = (accepted["expr_neg"], e_t) if neg_is_a else (e_t, accepted["expr_neg"])
            rawb = m.ask(p_blind.render(a=a_, b=b_), None, 8)
            letter = (C.first_word(rawb) or "")[:1].upper()
            blind = {"neg_is_a": neg_is_a, "raw": rawb, "judge_letter": letter,
                     "judge_found_flip": ((letter == "A") == neg_is_a) if letter in ("A", "B") else None}
        row = {"image_id": iid, "reason": reason, "accepted": accepted is not None,
               **(accepted or {"expr_neg": None, "changed_from": None, "changed_to": None, "flipped_idx": None,
                               "clause_neg": None, "neg_clause_verdicts": None}),
               "blind": blind, "tried": tried, "n_fresh": fresh, "raw_flips": raw_flips, "checker": model}
        C.append_jsonl(out, row)
        print(f"  [{n}/{len(todo)}] {iid} {reason}: {'ACCEPTED ' + repr(accepted['changed_to']) if accepted else 'no usable flip'} "
              f"after {len(tried)} tries, {calls} checker calls, {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = C.read_jsonl(out)
    print(f"reflip: accepted {sum(1 for r in rows if r['accepted'])}/{len(rows)}; next: "
          f"policy --run {run} --which neg2, then report --run {run} --primary-checker {primary}", flush=True)
