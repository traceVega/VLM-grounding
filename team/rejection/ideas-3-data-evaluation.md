# Ideas 3 — data and evaluation lens: which negatives to build, how to verify them, how to measure rejection so gains are real

Author: `Ideas3`. Date: 2026-09-16. Inputs: the five Phase 1 files, `notes/RL-DESIGN-CANDIDATE-VERIFICATION.md` §1/§5/§9, `notes/SATURATION-AUDIT-2026-09-06.md` §2.2/§5/§7, `notes/CAPABILITY-MECHANISMS-2026-09-06.md`, `notes/BENCHMARK-LANDSCAPE.md`, `README-INFRA.md`, the three measurement notes, and one CPU query of `~/vlmg-data/prepared/groundingme/items.parquet`. No GPU job was run. Labels as in the charter; `[P]` = project measurement; `[EST]` = my cost estimate, not measured.

## 0. What I assume, what I dispute, and three facts that shape every idea below

**Assumed reading of the base's abstention signal.** I take the Diagnostician's sub-axis reading (`diagnosis-local.md` §6): p(null) rises when *no candidate of the head-noun gestalt* remains (text string gone, tiny blob gone), and does not rise when a same-category candidate remains with one false clause — the deconfounded null result is Spearman ρ=0.06 (n=87) for length within Discriminative/non-Text, and the three ≤20-word Rejection/Text items sit at log10 p(null) −16.9/−17.9/−15.5 `[P]`. The length reading (`CORRECTION` §1, `RL-DESIGN` §1.2) stays alive only through idea I3-2, which is built to settle it directly. If I3-2 falsifies the sub-axis reading, ideas I3-4 and I3-7 change rank (stated where relevant).

**Disputes with `RL-DESIGN` §1.4 gate 4 (load-bearing rate).** Gate 4 requires the flipped clause to be load-bearing. Under the sub-axis reading that is the *wrong* rung for GroundingME: GME's negatives are paragraphs where everything else still fits one present object and a single detail is false — i.e. the falsified clause is most often *not* what makes the candidate unique, the other 5–7 clauses do. Flipping a load-bearing clause produces "two partial matches" (target fails c2′, the sibling fails c1 or c3); flipping a decorative clause produces "one candidate that matches everything but one detail". The second is the GME regime and, I predict, the harder one for a partial-match accepter. Both are valid zero-satisfiers; they are different rungs and must be mined and reported separately (I3-1, I3-4). Load-bearing rate of the *positive* expression remains a necessary condition (siblings must exist so clauses can matter) — it is not the thing to flip on.

**Three facts from the parquet `[P: this session]`.** (i) GroundingME's prepared ids are `gme_00000..01004` with `image_sha256` and `image_phash` only; SA-1B source ids are not recoverable from our snapshot, so any SA-1B-based mining (I3-7) can only be de-duplicated perceptually. (ii) 1,005 items sit on 879 unique images and only **6** Rejection images are shared with a positive item — GME itself offers almost no same-image present/absent pairs; every matched-positive comparison so far (Diagnostician §1, AUROC 0.565 upper bound) is cross-image. I3-2 manufactures same-image pairs. (iii) Rejection word count q10/q25/q50/q75/q90 = 38/46/54/64/74; the charter's "39 words" is not the median of these items (median 54, as the Diagnostician found).

**Three constraints from Phase 1 that every idea obeys.** (a) Negatives that the current policy already rejects teach nothing — SAM 3 discards them and its data term was worth +14.7 cgF1 against +1.5 for the head `[VERIFIED: 2511.16719 via survey-C]`; Jedi's 2.67M unmined easy negatives were worth 0 `[VERIFIED: 2505.13227 via survey-D]`; Motto's top-K hurts at large K because easy negatives dominate `[VERIFIED: 2607.24407 via survey-D]`. So every set below carries a **hardness gate against the base policy** and reports the discard rate. (b) In-distribution negatives give near-perfect in-domain rejection and collapse OOD (GME 97.3 → 27.9; Ground-V 83.7 → 33.9 `[VERIFIED via survey-A]`), so every set is reported on a **source-disjoint** and **type-stratified** basis. (c) LLM-generated hard negatives carry text-only artifacts — SugarCrepe found "blind models with no access to the image outperform state-of-the-art vision-language models" on earlier hard-negative sets and fixed it with LLM generation plus "an adversarial refinement mechanism to maximally reduce biases" `[VERIFIED: 2306.14610 abs]`. So every text-side negative gets a **blind-detectability check**.

Mechanism ids D1–D8 are the Diagnostician's (`diagnosis-local.md` §7).

---

## I3-1. Policy-adversarial one-clause negative mining (PAM)

**Mechanism.** D2 (partial-match acceptance) as the target; D8 (pass@k of `null` ≈ 0) as the reason mining must be against the policy: only negatives the base *boxes* can produce a forced-null contrast with a non-zero advantage, and the easy ones dilute that contrast (Motto).

**Method.** Start from the existing OpenImages instance bank (9,692 scenes, 311,282 instance rows, `n_head_noun_instances` is already a column `[P: README-INFRA]`), filter to ≥3 same-category instances, area quantiles matched to GME (0.16% / 1.0% / 2.7%). Writer (Qwen3.5-9B, `expr_write_clausal`) writes one positive with n clauses. For each positive generate **K=8 one-clause-false variants**, two per sub-axis (appearance / component / text / state), tagged by flip type (decorative vs load-bearing, from the delete-one listener test). Gate 3 first, before any verifier: run the base policy (Qwen3-VL-4B-Instruct, GME-native prompt, decision-token harness) on every variant; keep a variant only if p(box) ≥ 0.99 **and** the emitted box lands on a same-category instance mask (IoU ≥ 0.5 with the target or a sibling) — the "same-category candidate accepted" signature the Diagnostician found on GME (object-sized boxes, p(box) ≥ 0.9996 on all 201 `[P]`). Record which instance it landed on. Then the cross-family gates: Molmo2 uniqueness on the positive (gate 2), Gemma4-12B per-sibling yes/no on the flipped clause (gate 1′ zero-satisfier), blind-detectability (below). Output per accepted negative: base log10 p(null), flip type, sub-axis, clause count of the positive, which instance was boxed. Yield = accepted / generated, per sub-axis × flip type.

**Closest prior work and delta.** SAM 3 §D.4 mines noun-phrase concept negatives (ontology siblings + Llama-4 proposals) and keeps a candidate only if the *current* model predicts masks for it overlapping the positive `[VERIFIED: 2511.16719]` — same acceptance logic, but at concept level with a detector; ours is clause level with a coordinate MLLM and the acceptance test is "boxed a same-category instance". Motto selects top-K hard negatives online by loss, unverified, K=20 optimal `[VERIFIED: 2607.24407]`; ours is offline, verified, with the discard rate reported. FineCops-Ref / RefBench-PRO / HumanRef mutate with an LLM and verify absence but have **no policy-hardness gate** — their in-domain 58–71 and OOD collapse are the result `[VERIFIED via survey-A]`. ROD-MLLM's LLM-mined "confusing absent objects" are the nearest MLLM precedent and were never gated against the policy either.

**Kill experiment.** *Precondition / instrument-only* (it moves no benchmark; it decides whether GME-regime negatives can be manufactured at all from the bank we have). Decides three things: (1) yield after all gates ≥ 30% of generated variants; (2) survivors sit in the GME regime: median base log10 p(null) ≤ −12 (GME Rejection IQR is [−18.8, −15.8] `[P]`; the OpenImages class-label removals sit at −0.2 — anything above −8 means the flip is *detectable* by the base and the negative is not GME-hard); (3) per sub-axis and flip-type yield, which decides what the bank can and cannot supply. **Pre-registered stop rule:** after 200 source scenes × 8 variants, if survival < 20% or the survivors' median log10 p(null) > −8, the bank cannot produce GME-regime negatives; switch image source (SA-1B-like, I3-7) before spending any training GPU-hour. **Cheaper decisive variant:** run gate 3 alone first (policy only, 1,600 prompts ≈ 25 min): if fewer than 30% of variants are boxed-on-a-sibling-or-target with p(box) ≥ 0.99, stop — the later gates can only lower yield.

**Cost table.**

| dataset | model(s), sequential | GPU RAM | host RAM | GPU-h | where | person-days | USD |
|---|---|---:|---:|---:|---|---:|---:|
| OpenImages bank (exists); 200 scenes → 200 pos + 1,600 neg variants | Qwen3.5-9B writer 19 GB; Qwen3-VL-4B policy 9 GB; Molmo2-8B 17 GB; Gemma4-12B 24 GB | ≤ 24 GB | ≤ 20 GB (stream images) | ≈ 2.0 `[EST: writer 15 min at ~3 s/long output; policy 1,600 × 1 s; Molmo2 5 min; Gemma 1,600 × ~4 siblings = 6,400 crop-VQA at ~0.5 s ≈ 55 min]` | local | 2 (prompts drafted in RL-DESIGN §9.4; `certify.py` + flip-type tagging to write) | 0 (+$20 for a 50-item human spot check, 1 h, optional) |

**Predicted outcome.** Yield after all gates 35–50% for appearance/state, 15–25% for component, < 10% for text (few OpenImages objects carry legible strings) `[SPECULATION]`; survivors' median log10 p(null) ≈ −15 (the base boxes the target or a sibling with full confidence) `[LIKELY: Diagnostician §2–3 pattern]`; **decorative flips harder than load-bearing flips** by ≥ 2 orders of p(null) `[SPECULATION, the dispute above]`. Killed if survivors' median p(null) > −8 (flip detectable → bank negatives are easier than GME's) or yield < 20%.

**Traps and controls.** *Text shortcut:* blind-detectability check — a text-only judge (Gemma4 without image, or a logistic model on the writer's own token log-probs) must not separate flipped from original expressions at AUROC > 0.60; any sub-axis above that gets SugarCrepe-style adversarial regeneration (re-flip until the blind judge fails). *Easy-negative dominance:* gate 3 removes them by construction; report the discard fraction per sub-axis so a reader can see what was thrown away. *Writer–policy same lineage:* writer is qwen, policy is qwen; `lineage.py` allows it because the verifier differs, but the writer's blind spots are the policy's blind spots, so the clauses it never writes are the ones the policy never sees — run a 50-scene Gemma-writer arm and compare yield and p(null). *In-distribution leakage:* pHash/DINOv2 exclusion of GME/OpenRef/PR-Bench/RefCOCO/Ref-L4/Ref-Adv images from the bank (I3-6) before mining. *Edit artifact:* none — this arm edits no pixels.

---

## I3-2. Clause-form vs paragraph-form rewrite of GroundingME's own 201 negatives (the open question survey A names)

**Mechanism.** D2 versus the length reading (D1's territory): does the zero live in the *paragraph* (many true clauses supporting a present candidate) or in the *clause type* (a present same-category object with one false property, at any length)?

**Method.** Inference only; GME images are never trained on. Step 1, identify the falsified clause of each of the 201 items (GME does not publish it): decompose with `expr_decompose`, check each atomic clause against the image with Gemma4-12B (`verifier_clause_instance` on the head-noun instances from SAM 3), take the clause(s) judged false; human check on 50 items with the existing local UI. Step 2, five forms per item, wording verbatim from the paragraph: **F0** original paragraph (n=201, the benchmark); **F1** head noun + the false clause only (≤ 20 words); **F2** head noun + false clause + one true clause; **F3** the full paragraph with the false clause moved to the front; **F4 (positive control, same image)** head noun + one true clause only — must box the object, measures whether shortening itself breaks grounding; **F5 (optional, same image, same length)** the paragraph with the false clause corrected to the true value (checker or human supplies the value) — a same-image, length-matched *positive*, which GME lacks (only 6 shared images `[P]`). Run the 8B-Instruct decision-token harness on all forms; report decline rate, log10 p(null), and AUROC F0-vs-F5 and F1-vs-F4, stratified by sub-axis.

**Closest prior work and delta.** FINER shows clause count is the difficulty dial for present-object false clauses in VQA (~80% → ~20% as clauses accumulate) `[VERIFIED: 2603.17662 via survey-A]`; our own length split is confounded with sub-axis (`CORRECTION` §1 vs Diagnostician §6); RefBench-PRO's yes/no rewrite (~chance) changes the *task*, not the expression form `[VERIFIED: 2512.06276]`. Delta: the first manipulation of expression form on the benchmark's own negatives with the image and the false clause held fixed, plus same-image positive controls.

**Kill experiment.** *Diagnostic, and a design decision.* Decides how training negatives must be written. **Pre-registered:** if F1 declines on the 151 non-Text items are ≤ 5% (Wilson upper < 10%) **and** the F0→F1 median shift is < +2 orders, the length reading is dead: a short checkable clause on a present object is as invisible as a paragraph, so the negative *type* (same-category present + false property) is the regime and length is free — I3-1 negatives are in-regime and I3-7 is not needed for regime reasons. If F1 declines ≥ 25% on non-Text, length/support is a real dial: fund I3-4's curriculum and I3-7. Middle band (5–25%): report per sub-axis; Text is expected to behave differently. F4 must box the object at ≥ 80%, else the forms are broken and nothing is decided. **Cheaper decisive variant:** 50 Text + 50 Appearance items × F0/F1/F4 = 300 prompts, 7 min.

**Cost table.**

| dataset | model(s) | GPU RAM | host RAM | GPU-h | where | person-days | USD |
|---|---|---:|---:|---:|---|---:|---:|
| GME 201 Rejection (+ F5 needs the true value) | Gemma4-12B clause check 24 GB; SAM 3 5 GB; Qwen3-VL-8B-Instruct 20 GB | ≤ 24 GB | ≤ 18 GB | ≈ 0.8 `[EST: 201 × ~6 clauses × ~3 instances ≈ 3,600 crop-VQA ≈ 30 min; 201 × 5 forms × 1.4 s ≈ 25 min]` | local | 1 (+0.5 for the 50-item human check of the identified clause) | 0 (in-house) to $150 (201 items × 2 min × $18/h external, if the false-clause identification is fully human) `[EST]` |

**Predicted outcome.** F1 declines 3% on non-Text, ~10% on Text; F0→F1 shift < +1 order on Appearance/Component/State `[LIKELY: the three short Text items and the n=87 null]`. Killed by F1 declines ≥ 25% on non-Text.

**Traps and controls.** *False-clause misidentification:* if the checker marks a true clause, F1 becomes a positive and a box is the right answer — the 50-item human check bounds the error; report F1 results on the human-verified 50 separately. *Text shortcut in the forms:* clauses are copied verbatim; no rewording. *Same-image pairs:* F4/F5 give the same-image AUROC; report it beside the cross-image 0.565. *Head-noun instance absent:* if SAM 3 finds no head-noun instance on a Rejection image, the item is a category-absent negative in disguise — count and report those (the benchmark's "head noun present by construction" is `[LIKELY]`, this measures it). *Parse failures:* the decision-token harness has none; say so.

---

## I3-3. Text-swap N0′ on the 50 Discriminative/Text positives, with N0 and a control edit

**Mechanism.** D2 at its sharpest cell — text identity: the Diagnostician's reading is that a text-bearing object is matched on "has text", not on "has *this* text" (Rejection/Text 49/50 carry a quoted string, all boxed `[P]`). Also partially settles `GME-PRECHECK` explanation 2 (edit vs never-there) for the Text cell.

**Method.** The 50 Discriminative/Text positives (median 13 words, e.g. "a vehicle, with number '2500' on its body"; 28 of them ≤ 20 words with 42.9% declines under *removal* `[P]`). Three conditions, expression byte-identical for the image conditions: **N0′** erase the string region on the referent (big-LaMa, in-mask) and render a different plausible string of the same length/colour/size (PIL overlay; a text-editing model only after asking, per repo rule) — answer `null`; **C** the same operation on a *non-referent* text-bearing region — answer: the original box; **N0** no edit, change the quoted string in the expression — answer `null`. Read decline rate and log10 p(null) versus ORIGINAL and versus the existing REMOVE rows for the same items (`tables/gme_remove.jsonl`).

**Closest prior work and delta.** HalluSegBench (GPT-4o class replacement on the referent, "vision-driven hallucinations more prevalent than label-driven") `[VERIFIED: 2506.21546 via survey-A]`; VisMin minimal-change image pairs with a 4-step human verification, 2,084 verified items `[LIKELY: search summary of 2407.16772]`; FineCops-Ref negative images by inpainting. Delta: an *attribute* swap that leaves the object and the category in place, on the benchmark's own images, with a control edit and the removal condition already measured on the same items — a three-way ladder (removed / present-wrong-string / present-right-string) at the decision token.

**Kill experiment.** *Precondition for text edits as training negatives, and a direct D2 test.* Two-sided reading, pre-registered: **N0′ shift < +2 orders and declines ≤ 2/50** → the base does not verify string identity; text-swap negatives are GME-hard (gate 3 passes) and the Text zero is a verification deficit, not a paragraph-integration deficit. **N0′ shift ≥ +5 orders** → the base verifies identity once the string is wrong; text swaps are *easy* negatives (PAM would discard them) and the Text zero on GME must come from the paragraph — this would contradict the sub-axis reading for Text and support the length reading. **Instrument stop rule:** if C moves p(null) by > 1 order or moves the box off the referent on > 5/50, the edit is detectable and N0′ is unusable — fall back to N0. N0 vs N0′ difference measures the text shortcut of the string change. **Cheaper decisive variant:** N0 alone (50 prompts, 2 min, no editing): if N0 shows no rise, N0′ is unlikely to.

**Cost table.**

| dataset | model(s) | GPU RAM | host RAM | GPU-h | where | person-days | USD |
|---|---|---:|---:|---:|---|---:|---:|
| 50 GME Discriminative/Text positives; 100 edits | SAM 3 5 GB; big-LaMa (small); Qwen3-VL-8B-Instruct 20 GB; Gemma4-12B OCR legibility check 24 GB | ≤ 24 GB | ≤ 18 GB | ≈ 1.0 incl. editing | local | 1.5 (edit script; 1 h human review of 100 edits in the existing UI) | 0 |

**Predicted outcome.** N0′ shift +1 to +3 orders, declines ≤ 5%; N0 within 1 order of N0′ `[LIKELY: the three short Text negatives at −16]`. Killed by a ≥ 5-order rise.

**Traps and controls.** *Legibility:* the new string must be read back correctly by the OCR checker ≥ 90%, else "no rise" is "cannot read it". *Original string elsewhere:* run OCR over the whole image; if the original string appears on another object the N0′ expression may still be true — drop the item. *Edit artifact:* C, as in the removal design (0/316 false abstentions on controls `[P]`). *Byte-identical expression* for N0′/C; N0 changes only the quoted token. *Small text:* the 50 items include tiny plates — report by string box area.

---

## I3-4. Support-ratio ladder (clause-count curriculum with the flip-type axis), load-bearing-verified

**Mechanism.** D2 in the paragraph regime: the negative's *support ratio* s = (true clauses fitting the boxed candidate) / (all clauses). GME ≈ (n−1)/n with n ≈ 6–8 (median 54 words); class-label removals are s = 0/1; short mined negatives are s = 0/1 or 1/2.

**Method.** Data only (training is Ideas1's lane; I give the data and the inference-only test). From I3-1's pipeline, build four rungs by the positive's clause count and flip one clause: **L1** head + 1 clause (s=0/1); **L2** 1 true + 1 false (1/2); **L3** 3 true + 1 false (3/4); **L4** 5–7 true + 1 false, 40–60 words, GME-style (≥ 5/6). Cross each rung with flip type (decorative vs load-bearing, from the delete-one listener test = RL-DESIGN gate 4 machinery). Every rung passes the same gates (uniqueness, zero-satisfier on the flipped clause per sibling, hardness gate 3, blind-detectability). Inference-only ladder: base p(null) and box-landing (target vs sibling) per rung × flip type. Training A/B (Ideas1's objective, same count, same seeds): L1-only vs L1–L4 mixed vs L4-only; report GME Rejection, per-rung held-out rejection, positives, and the protocol card (I3-5).

**Closest prior work and delta.** FINER's granularity levels (clause accumulation as the dial; DPO gains +5.5 to +24.2 on InternVL3.5-14B) `[VERIFIED: 2603.17662]`; SAM 3's negatives-per-image sweep 0/5/15/30 (IL_MCC 0.44 → 0.68) `[VERIFIED: 2511.16719]`; GroundingME's own 2:1 mix with an unstated negative construction. Delta: support ratio and flip type as explicit, verified difficulty variables in REC negatives; nobody has separated "candidate matches everything but one detail" from "no candidate matches well".

**Kill experiment.** *Inference-only part is the precondition; the training A/B is the headline.* Pre-registered for the ladder: if base log10 p(null) does not fall monotonically with s (Spearman ρ > −0.3 across L1–L4, n ≥ 100 per rung) **and** decorative-vs-load-bearing shows < 1 order difference, the ratio is not the dial, the curriculum has no axis, and the rung distinction is dropped (still report it). Pre-registered for the A/B: L4-in-mix must beat L1-only on GME Rejection by ≥ 6.2 (the CI) at equal positive cost (protocol card), else "short negatives suffice" and I3-7 is not worth building. **Cheaper decisive variant:** the inference-only ladder on 400 mined items (0.5 GPU-h) before any training. If I3-2 finds F1 ≈ F0, run only the flip-type axis (support ratio is then known not to matter).

**Cost table.**

| dataset | model(s) | GPU RAM | host RAM | GPU-h | where | person-days | USD |
|---|---|---:|---:|---:|---|---:|---:|
| Ladder: 400 mined items (I3-1 pipeline, 100 per rung) | as I3-1 | ≤ 24 GB | ≤ 20 GB | ≈ 2.5 (mining) + 0.2 (policy) | local | 1.5 on top of I3-1 | 0 |
| Training A/B: 3 arms × 1,500 items, 4B LoRA GRPO | Qwen3-VL-4B + co-located vLLM 28–31 GB | 31 GB | ≤ 20 GB | 10–15 per arm (RL-DESIGN §6) → 30–45 | local (2 weeks) or cloud: 3 × 1×80 GB ≈ 12 H100-h ≈ $40 `[EST]` | 3 | 0–40 |

**Predicted outcome.** Base p(null) falls ~1 order per rung from L1 to L4 and decorative flips are 2+ orders lower than load-bearing flips at every rung `[SPECULATION]`; in the A/B, L1–L4 mixed beats L1-only by +8 on GME Rejection at ≤ 2 positive cost `[SPECULATION]`. Killed by a flat ladder or an A/B gap < 6.2.

**Traps and controls.** *Length as a text shortcut:* a model can learn "long expression → reject" if negatives are longer than positives on average — match the clause-count distribution of positives and negatives exactly within each rung (RL-DESIGN's 40/25/15/15/5 mix must be length-balanced). *Easy-negative dominance across rungs:* gate 3 applies per rung; L1 will have the highest discard rate — report it. *Writer fatigue at L4:* 5–7 checkable clauses on OpenImages objects may be unwritable (gate-4 threshold "≥ 3 load-bearing clauses for > 35 words" from RL-DESIGN §6 stands); if L4 yield < 15%, L4 needs richer images (I3-7). *Leakage:* as I3-6.

---

## I3-5. The rejection evaluation protocol (REP) — the card every arm must ship

**Mechanism.** D4 (format-scoring artifact), D5 (over-abstention is visibility), D6 (metric inconsistency across rejection benchmarks); and the over-refusal blind spot survey C names (WebArena 54.9% feasible tasks called impossible under a refusal hint `[VERIFIED: 2307.13854 via survey-C]`).

**Method.** A fixed nine-row card, computed by one script from the decision-token harness plus one free-generation pass, frozen before any training run reads it:

| row | what | why it exists |
|---|---|---|
| 1 | GME Rejection as **three numbers**: literal `{"bbox_2d": null}` rate, parse-failure rate, official score (which is their sum by construction) | official `evaluate.py` counts unparseable as correct `[P: source read]`; every published thinking-mode number lacks the split |
| 2 | Negative-type stratification, one column each: category-absent (class-label removals; gRefCOCO no-target once the adapter exists), other-image (OpenRef noun swap, gated dataset), attribute-false text-only (held-out N0), text-swap (I3-3), edited (N0′/REMOVE with C), GME L4 by sub-axis | the −0.304 correlation between GME Rejection and OpenRef N3R `[P]`; never pool types |
| 3 | Over-refusal on positives, **size-stratified**: GME 804 by `size_bin` (base false-null is 62.5% tiny / 21.7% small / 4.1% medium / 0.8% large `[P]`), RefCOCOg val, and the F4/F5 same-image positives from I3-2 | a method that "improves rejection" by raising p(null) on hard-to-see objects moves Limited, not Rejection |
| 4 | Risk–coverage: AUROC of p(null) absent-vs-*matched* present (same-sub-axis Discriminative positives, and F5 same-image), and correct-abstention at a 5% false-null budget calibrated on positives | a run can improve the ranking without moving greedy; the Limited-driven 0.298 must not be the headline AUROC |
| 5 | Protocol swap: train-format `{"boxes": []}` vs benchmark-native `{"bbox_2d": null}`; gain must survive, Δ(Acc@0.75/0.9) ≥ Δ(Acc@0.5) on positives (CAPABILITY rule 2) | separates format compliance from selection |
| 6 | Blind control: grey image, same prompts; any rejection gain that survives blind is a text-prior gain | the text shortcut is *the* failure mode for rejection training (SugarCrepe) |
| 7 | Suite consistency: rank correlation across GME / Ref-Adv-s / Ref-L4 / OpenRef before vs after | specialization decorrelates (ρ 0.82 → 0.20 `[P: CAPABILITY §5]`) |
| 8 | General regression: DocVQA, AI2D, MathVista, CharXiv (not MME/MMBench/POPE) | the insensitive three move 7× with the eval prompt `[P: CAPABILITY §4]` |
| 9 | **Human ceiling**: 3 annotators on the 201 Rejection + 201 matched Discriminative positives, majority label, per-sub-axis agreement | GME's only human number is 100 items at 91% `[VERIFIED: 2512.17495]`; a Rejection score above ~90 may be unreachable by construction, and CI bins of 6.2 make the ceiling load-bearing |

**Closest prior work and delta.** RC-GRPO (the one paper with P-acc, a regression table and an α trade-off curve) `[VERIFIED: 2608.04698]`; OSWorld-G `eval.py` (refusal = a specific value, parse failure = wrong) `[VERIFIED: repo via survey-C]`; "The Missing I Don't Know" triple scoring and abstention by stratum `[VERIFIED: 2609.17686 via survey-B]`; SAM 3's IL_MCC with a human row. Delta: the first REC rejection card that separates parse failure, stratifies negative type and positive size, reports risk–coverage against matched positives, and carries a blind control and a human ceiling.

**Kill experiment.** *Instrument-only; its validation is the kill.* Run the card on the base and on two cheap known-gaming arms: (a) a permissive prompt ("output null if unsure") — an over-refusal arm; (b) a deliberately broken JSON emitter — a parse-failure arm. Pre-registered acceptance: the card must show (a) as rejection-up with row-3 false-null up (and row 6 blind-gain > 0), and (b) as official-score-up with literal-null flat. If either gaming arm reads as an improvement on the card, the card is rejected and redesigned before any training number is read. **Cheaper variant:** rows 1–4 and 6 only (1,005 × 3 prompts ≈ 1.2 GPU-h); rows 5, 7–9 attach when a trained arm exists.

**Cost table.**

| dataset | model(s) | GPU RAM | host RAM | GPU-h | where | person-days | USD |
|---|---|---:|---:|---:|---|---:|---:|
| GME 1,005 × {harness, permissive, broken-JSON, grey} + class-label removals (exist) + RefCOCOg val | Qwen3-VL-8B-Instruct 20 GB (free generation for row 1, ~10× the 1.4 s) | 20 GB | ≤ 18 GB | ≈ 2.5 `[EST]` | local | 2 (script + freeze) | 0 |
| Human ceiling: 402 items × 3 annotators = 1,206 judgments × ~1.5 min (54-word paragraphs, clause-by-clause) ≈ 30 h + 30-item qualification | — | — | — | 0 | Prolific/UpWork | 1 (task build on the existing local UI; images are research-licensed → in-house or NDA workers, not public crowd `[P: README-INFRA P11]`) | ≈ $550 at $18/h incl. fees `[EST]`; in-house 2 annotators × 2 days = $0 |

**Predicted outcome.** Base card: row 1 = 0 / 0 / 0.0; row 3 false-null 5.1% overall, 26.7% Limited/Small; row 4 AUROC vs matched Discriminative 0.47, vs F5 same-image ≈ 0.55 `[LIKELY: Diagnostician §1]`; human ceiling on the 201 ≈ 88–92 with ~8% items unresolvable `[LIKELY: GME's 91% on 100]`. The permissive-prompt arm reads as Rejection 5–15 with false-null +10–20 on Limited `[SPECULATION]`. Killed (as an instrument) only by failing the gaming-arm validation.

**Traps and controls.** *The card is not a leaderboard:* never sum rows into a scalar; CAPABILITY §5 forbids mixing rejection into a weighted total. *Human ceiling annotators see only the image and the description*, never the model output or the sub-axis (existing P11 rules). *Matched positives* must be same sub-axis and same size bin. *Parse-failure rate on thinking models* must be computed with the official scorer's parser verbatim, not our harness.

---

## I3-6. Contamination and leakage checks for every mined set

**Mechanism.** Evaluation validity (D6's neighbour): in-distribution leakage is how every 50-point in-domain rejection gain in survey A was manufactured, and SA-1B mining (I3-7) can pull GME's own source photographs.

**Method.** Four layers, each a pass/fail with a number. (a) Exact: sha256 and pHash of every mined image against GME (879 images, `image_phash` exists in `items.parquet`), RefCOCOg val/test, Ref-L4, Ref-Adv-s, PR-Bench, OpenRef (gated on HF — request access now; until then OpenRef is *not* a reportable benchmark for any trained arm). `schema.assert_dev_slice_disjoint` already does the pHash half. (b) Near-duplicate: DINOv2-B embeddings, cosine NN, threshold calibrated on known transforms (resize, crop 80%, JPEG q60, flip) so that ≥ 99% of transformed copies are caught and ≤ 0.1% of random pairs; needed because SA-1B ids are not recoverable from our GME snapshot `[P: this session]` — if the HF raw snapshot retains original filenames, add an id match and say so. (c) Expression-level: 8-gram overlap between mined expressions and every benchmark expression; a writer trained after Dec 2025 has plausibly seen GME's expressions `[LIKELY]` — overlap > 1% of items triggers writer re-prompting with GME-style phrases banned. (d) Source-disjoint reporting: every trained arm reports GME on the image-clean subset **and** its train set's image source (OpenImages vs SA-1B) so the OOD claim is explicit.

**Closest prior work and delta.** Ref-L4's COCO-vs-Objects365 rows (8.2/13.6-point gap as a contamination probe) `[P: SATURATION-AUDIT §3]`; ZwZ's image-disjoint MMStar check (+20 in-pipeline → +2.2 disjoint) `[P: CAPABILITY §2]`; SATURATION-AUDIT phase-1 step 1's planned pHash/DINOv2 index. Delta: adds the expression-level layer and the source-disjoint reporting rule; makes it a gate a mined set must pass before it is written to disk as `certified`.

**Kill experiment.** *Instrument-only.* Stop rules: any mined set with > 0.5% near-duplicates of an eval image is rebuilt with hits removed and the removal logged; expression 8-gram overlap > 1% triggers re-prompting; if the DINOv2 threshold cannot reach 99%/0.1% on the calibration transforms, fall back to a stricter pHash radius and report the miss rate. **Cheaper variant:** pHash-only (CPU, minutes) for OpenImages-mined sets; DINOv2 only when SA-1B enters.

**Cost table.**

| dataset | model(s) | GPU RAM | host RAM | GPU-h | where | person-days | USD |
|---|---|---:|---:|---:|---|---:|---:|
| ~10k mined + ~60k eval images | DINOv2-B (~0.4 GB) | 2 GB | ≤ 12 GB (stream) | ≈ 0.4 | local | 1 | 0 |

**Predicted outcome.** OpenImages bank vs GME: < 0.1% near-duplicates (different source) `[LIKELY]`; SA-1B subset vs GME: 0.5–2% near-duplicates if sampled from the same shards `[SPECULATION]`; expression overlap < 0.5%. Nothing to kill; the check either passes or forces a rebuild.

**Traps.** pHash misses crops; DINOv2 threshold false positives on near-identical scenes (same street, different day) — keep both and report counts, not a verdict; a Molmo2 listener trained on PixMo may have seen SA-1B-adjacent images — irrelevant for contamination but note for lineage.

---

## I3-7. A GroundingME-style train split from SA-1B-like images with a cross-family writer — build, or mine?

**Mechanism.** D2 at the full GME regime (real unedited photos, 40–70-word human-edited paragraphs, head noun present, one false clause). Also the only route to a **human-edited vs LLM-flipped** difficulty comparison, which no paper has (GME's synthetic negatives for its 2:1 mix are of unstated construction `[VERIFIED: 2512.17495 via survey-A/D]`).

**Method.** Sample SA-1B (or SA-1B-like: high-resolution, cluttered, licence-checked) images, DINOv2-clean against GME and every eval set (I3-6). SAM 3 instances; pick targets with ≥ 1 same-category sibling and GME-matched area quantiles. **Writer = Gemma4-12B** (GME drafted with Gemini-2.5-Flash, then annotators edited; our writer must not be the policy's family and must not be the checker's — so checker becomes Molmo2 for this arm, listener Qwen3.5-9B; `lineage.py` re-checked). Draft a 40–60-word description of the target (GME style: appearance + component + text/state + a relation). Two negative arms on the same drafts: **(i) LLM-flip** exactly one target-property clause, decorative by default (I3-4), rest verbatim; **(ii) human-edit** on a 300-item subset — annotators "introduce one factual error" as GME's did (exact GME instructions unknown `[LIKELY]`, so ours are written to match the paper's description). All gates from I3-1 (hardness against the base, zero-satisfier per sibling on the flipped clause, blind-detectability, uniqueness of the positive). Then the two comparisons that decide build-vs-mine: (a) base p(null) distribution of (i) vs (ii) vs GME's 201 — if (ii) is not harder than (i) for the base, human editing buys nothing beyond LLM flipping; (b) a training A/B (Ideas1's objective): 1,000 I3-1 OpenImages short negatives vs 1,000 SA-1B GME-style negatives, same positives count, on the I3-5 card.

**Closest prior work and delta.** GroundingME's construction (Gemini draft + human edit, no train split, in-domain 97 / OOD 28 from the same checkpoint); RefBench-PRO's 200k train with 49k reject items by "minimal plausible mutation" (in-domain 58, never tested OOD) `[VERIFIED: 2512.06276]`; HumanRef's 13k rejection train (Qwen rewrite + Molmo verification; RexSeek 0 → 54) `[VERIFIED: 2503.08507]`. Delta: a GME-regime train split with a policy-hardness gate and cross-family verification, on source-disjoint images, plus the human-vs-LLM negative comparison.

**Kill experiment.** *Headline (the A/B) with a precondition (the p(null) comparison).* **Pre-registered:** build only if I3-2 shows F1 ≠ F0 (paragraph form matters) **or** I3-1 yield at the GME regime is < 20% on OpenImages; otherwise the short mined set is in-regime and cheaper and this idea is skipped. For the A/B: GME-style must beat OpenImages-short by ≥ 6.2 on GME Rejection at equal positive cost (card rows 3, 5, 6), else "regime-matched paragraphs are unnecessary" is the finding. For human-vs-LLM: if the base p(null) on (ii) is within 1 order of (i) and the checker's disagreement rate is similar, drop human editing from the scale-up (saves ~$1.5/item). **Cheaper decisive variant:** 300 SA-1B scenes, LLM-flip only, no human arm, inference-only comparison against GME's p(null) distribution (≈ 2 GPU-h).

**Cost table.**

| dataset | model(s), sequential | GPU RAM | host RAM | GPU-h | where | person-days | USD |
|---|---|---:|---:|---:|---|---:|---:|
| SA-1B: 1–2 shards (~10 GB each, research licence — confirm before download, per repo rule) → 3,000 scenes | SAM 3 5 GB (measured 3.6 h / 9.7k scenes → ~1.2 h); Gemma4-12B writer 24 GB (~3 s × 3,000 ≈ 2.5 h); Qwen3-VL-4B hardness gate 9 GB (6,000 × 1 s ≈ 1.7 h); Molmo2 checker on flipped clause × siblings (3,000 × ~3 ≈ 9,000 crop-VQA ≈ 1.5 h) + full-expression listener (0.5 h); DINOv2 dedup 0.4 h | ≤ 24 GB | ≤ 20 GB (8K images: decode-and-downscale streaming, ≤ 8 in flight) | ≈ 8–10 → ~1,200–1,500 certified items at 40–50% yield `[EST]` | local (one weekend) | 5 (SA-1B adapter, writer prompts, human task build) | human arm: 300 items × 5 min = 25 h ≈ $450 `[EST]`; cloud optional |
| A/B training (2 arms) | Qwen3-VL-4B LoRA GRPO 28–31 GB | 31 GB | ≤ 20 GB | 20–30 | local or 2 × 1×80 GB ≈ 8 H100-h ≈ $25 | 2 | 0–25 |

**Predicted outcome.** LLM-flip SA-1B negatives: base median log10 p(null) ≈ −16, boxed 100%, yield ~40% `[LIKELY: GME pattern]`; human-edited ones no harder for the base (both at the floor) but 10–20% more often rejected by the checker as "unclear" `[SPECULATION]`; A/B: +5 on GME Rejection for GME-style over short-mined at equal positive cost `[SPECULATION]` — i.e. *below* the 6.2 bar, which would make "mine short, cheap" the recommendation. Killed by I3-2's F1 ≈ F0 (then not built) or an A/B gap < 6.2.

**Traps and controls.** *Style leakage:* a Gemma writer imitating GME's Gemini-drafted style narrows the OOD claim; report suite consistency (card row 7) and Ref-Adv/OpenRef, not GME alone. *Text shortcut at paragraph scale:* fluency of a flipped paragraph is high by construction, but blind-detectability is still measured. *Image overlap with GME:* mandatory DINOv2 dedup; sample from shards GME is unlikely to have used only if the paper names them (it does not `[LIKELY]`). *Annotator instructions* differ from GME's unknown ones — the human arm's difficulty is a lower bound on GME's. *Cost of the checker* dominates at 6–8 clauses; check only the flipped clause per sibling plus one full-expression listener pass, never every clause per sibling.

---

## Dependencies and order

```
I3-6 (contamination, CPU/0.4 GPU-h)  ─┐
I3-5 rows 1–4,6 (card, 1.2 GPU-h)     ─┼─ instruments: exist before any number is read
                                       │
I3-2 (rewrite GME 201, 0.8 GPU-h) ─────┼─ day 1: settles clause-form vs paragraph
I3-3 (text-swap 50, 1.0 GPU-h)    ─────┘  day 1: settles text-identity cell
        │
I3-1 (PAM mining, 2 GPU-h) ─────────────── day 2–3: yield, regime, flip-type ladder
        │
   F1 ≈ F0 ─────► I3-4 flip-type axis only; skip I3-7; train on OpenImages-short (Ideas1)
   F1 ≠ F0 ─────► I3-4 full ladder + I3-7 (SA-1B, 8–10 GPU-h + $450) ► A/B
I3-5 rows 5,7–9 + human ceiling ($550): order now, attach to the first trained arm
```

Local total before any training: ≈ 6 GPU-h, 6 person-days, $0–$700 (human ceiling + optional human-edit arm). Every item fits the 32 GB card one model at a time and the 23 GB WSL host ceiling with streamed images.

## Ranking 1 — information per person-week (cheapest decisive first)

1. **I3-2** — 0.8 GPU-h, 1 pd: settles the one dispute (clause type vs paragraph) that re-ranks every other data idea, and manufactures the same-image positive pairs GME lacks.
2. **I3-3** — 1 GPU-h, 1.5 pd: settles the text-identity cell and whether attribute edits are usable negatives; N0 variant is 2 minutes.
3. **I3-1** (gate-3-first variant) — 0.5 GPU-h to learn yield and regime; 2 GPU-h for the full gated set; decides whether the existing bank can feed training at all.
4. **I3-5** — 2.5 GPU-h + 2 pd: zero information about the model, but every later number is uninterpretable without it; the gaming-arm validation is the cheapest way to catch a fake rejection gain.
5. **I3-6** — 1 pd: mandatory, low information, blocks SA-1B.
6. **I3-4** — inference ladder cheap (0.7 GPU-h on top of I3-1); the training A/B is 30–45 GPU-h and belongs after Ideas1's objective is fixed.
7. **I3-7** — 5 pd + $450 + 8–10 GPU-h, conditional on I3-2.

## Ranking 2 — paper quality if every kill passes

1. **I3-7 + I3-1** — a released, source-disjoint, policy-adversarial, cross-family-verified GME-regime train split with the first human-edited vs LLM-flipped difficulty comparison; the missing artifact in the whole line (GME has no train split; RefBench-PRO/HumanRef are in-domain only).
2. **I3-5** — a rejection evaluation card with a human ceiling and gaming-arm validation; an evaluation contribution the field is visibly missing (no published method reports over-refusal on GME positives; three scorers count parse failures as rejection).
3. **I3-4** — support ratio and flip type as the difficulty variables of REC negatives, with a curriculum result.
4. **I3-2** — a clean diagnostic (which of clause-form / paragraph carries the zero) with same-image controls; a strong section, not a paper.
5. **I3-3** — one figure in the diagnosis.
6. **I3-6** — a methods paragraph.

## What I could not verify

VisMin's construction and yield (search summary only); GME's exact annotator instruction for introducing errors; whether the HF GroundingME raw snapshot retains SA-1B filenames (our prepared parquet does not); SA-1B licence terms for this use; per-second throughput of Gemma4-12B crop-VQA and the Qwen3.5-9B writer (costs marked `[EST]` from the 8B 1.4 s/item measurement); OpenRef access (gated). All USD and person-day figures are estimates.
