# Innovator — unsolved-problem map and five method proposals for very hard multi-constraint referring expressions (2026-09-16)

Author: `Innovator`. Inputs read in full: `team/methods/CHARTER.md`, `m1-rl-hard-items.md`, `m2-grounding-methods.md`, `m3-data-methods.md`, `m4-verification-tts.md`; skimmed for constraints: `notes/REJECTION-DEEP-DIVE-2026-09-16.md` §1, §4.1, §6; `notes/RL-DESIGN-CANDIDATE-VERIFICATION.md` §0-3, §6, §9; `notes/CAPABILITY-MECHANISMS-2026-09-06.md` §4; `team/rejection/ideas-1` (§0, Ideas 2/3/4/6), `ideas-2` (Ideas 2/7), `ideas-3` (I3-1, I3-4). Every number below is one the surveys mark `[VERIFIED]` or the project marks `[P]`; I add no new web claims. Where I speculate I say so.

**Plain-language glossary used throughout** (one line each, so the rest reads without jargon):
- *pass@k* — of k independent samples, does at least one hit the right answer? "pass@k ≈ 0" means the model essentially never produces the right answer, however many times you ask.
- *prefix* — the first part of a correct answer, placed in front of the model so it only has to finish; *withdrawn* = removed again before the model is deployed.
- *GRPO* — the RL recipe where the model writes 8 answers to the same question and is pushed toward the ones that scored above the group's average; if all 8 score the same, nothing is learned.
- *DPO / pairwise loss* — training on a comparison ("this answer should be more likely than that one") instead of on a target; it has a gradient even when the model never produces the better answer on its own.
- *sibling* — another object of the same category in the same image (the wrong "person" when the expression means a specific person).
- *N0 / N0' / C* — a negative made by changing one detail in the sentence / by editing that detail in the image / a control where a non-referent object is edited and the answer stays the box (`RL-DESIGN` §1.3).
- *false-null* — the model says "none" on an item that has a target.
- *grey-image control* — run the same thing on a blank grey image; anything that still "works" is reading the text, not the picture (GroundingME negatives are text-separable at AUROC 0.92 `[P]`).
- *protocol swap* — train with one output format or scaffold, test under the plain GroundingME prompt without it; a gain that only exists inside the scaffold is format-following, not capability (`CAPABILITY-MECHANISMS` §4 rule 2; Acc@0.9 must move at least as much as Acc@0.5).
- *hint reliance* — of the successes the model has *with* the prefix, how many vanish when the prefix is removed at the same checkpoint (HiLL 2604.00698); low reliance = the model learned to do it, high = it learned to follow.

---

## 1. Unsolved-problem map (what the field can and cannot do on our items)

1. **The failure is one operation, not two.** 338/804 GroundingME positives are boxed on a wrong same-category object at p > 0.999, and 201/201 negatives are boxed at p ≥ 0.9996; Σ p(null) over the 201 is 3.4e-4 `[P]`. The details never enter the choice of object; "none" is the k = 0 case of the same choice (`RL-DESIGN` §0).
2. **Sampling more does not reach them.** GroundingME's own best-of-16 over the 235B thinking model moves Discriminative 66.6 → 66.7 while Spatial moves +5.2 `[V: 2512.17495]`; at k = 16, T = 1 the 8B is expected to produce a null on 0.005 of the 201 negatives `[P]`.
3. **Only one RL mechanism has repeated controlled evidence of solving items the base cannot sample: a partial correct solution placed before the policy, scored on the original task, withdrawn per item.** Six 2025-26 papers, all math on ≤ 8B text models: QuestA (unsolved-at-32: 5 → 2), AdaPrefix-GRPO (17% of pass@16 = 0 items solved unhinted), POPE (k = 128), PrefixRL (pass@64 +28 on pass@512 = 0 items; cross-family prefixes work), MFC (pass@8 at prefix fraction 0 / 0.25 / 1 = 2.2 / 34.0 / 85.4%), 2606.22317 (226/538 base-unsolved at k = 256 become solvable) `[M1 §2]`. Never done for grounding, for boxes, or for a "none" answer.
4. **The same papers say what does not work:** SFT on the full solution collapses (POPE 13.6 → 2.0), always-on hints cost ~10 points (Guide 40.97 vs 51.03), unrelated prefixes give nothing (PrefixRL), easy items in the hard batch stall the hard ones (POPE) `[M1]`. Off-policy trace mixing, advantage tricks, difficulty curricula, self-play (reward peaks at 50% solve rate: VisPlay r = 1 − |2c − 1|) and on-policy distillation never enter the zero rung `[M1 §2, M3 §2]`.
5. **In grounding the unit of decision is the lever.** Rex-Thinker decomposes as: candidate list in the prompt +11.8 DF1 but −18 rejection (given a list, the model picks one); per-candidate check +0.9 DF1 and **+13.8 rejection**; GRPO on top +1; its teacher traces were answer-conditioned (GPT-4o without the answer: 53.2 DF1), so the student learned a format in which its own judgments compose `[V: 2506.04034]`.
6. **Free-form thinking is the wrong lever for our dimension:** Discriminative drops in every Qwen3-VL size under thinking (8B 61.3 → 52.5; 32B 75.0 → 65.7) while Spatial rises +17 to +24 `[V: 2512.17495]`; the model commits to an object early and reasons about it (2604.11025's grounding paradox). Structured enumerate-then-verify helps; free CoT on perceptual REC hurts (Perception-R1 RefCOCO 89.1 → 75.1) `[M2 §6]`.
7. **Rewarding the intermediate step works and needs a guard:** TreeVGR's recall + precision reward on the trace's boxes is +12.4 over the same cold start without it; dropping the precision term gives accuracy 0.0 (the model enumerates and never answers) `[V: 2507.07999]`. Every plain IoU-reward variant is pass@1 sharpening: a group that boxes the same wrong sibling eight times has zero variance whatever the shaping `[M2 §2.3]`.
8. **Pairwise contrast is the only data route with consistent held-out transfer, and it moves below-chance items** (VisMin: Idefics2 spatial 18.6 → 83.0; CounterCurate → SugarCrepe +5-7; FINER-tuning → DASH +2 to +6 at no alignment tax; S-VCO +12.2 vs DPO +7.5 on the same pairs) `[M3 §8]` — **and nobody has applied one to box outputs** `[M3 §5.3]`.
9. **The positive-side cost tracks how much of the output stream the negatives rewrite:** null-in-stream SFT at 2:1 costs −21 Discriminative for +27.9 Rejection; a separate decision (SAM 3 presence head: pmF1 62.4 → 62.8 flat at 0 → 30 negatives/image) or a pairwise loss costs nothing `[M3 §8]`. Mined negatives are one-third not negatives (ARHN relabels/filters 38%) and cleaning pays more out of distribution `[V: 2604.11092]`.
10. **A verifier helps only if it knows more than the policy.** Self-verification of a drawn box correlates r ≈ 0.22 with correctness, and rendering the ground-truth box changes the refined IoU by +0.01 `[V: 2606.13156]`; same-family verifiers at 8-23% rubric accuracy flip self-training to −3 to −11 `[V: 2606.14629]`. The pass@k ≈ 0 → solved numbers (STV 1.5 → 21; R-OPSD 20.3 → 30.5 with an 89.5%-accurate labelled reflector) all used privileged information `[M4 §2]`. No verifier has been measured on category-present, one-clause-false negatives except our own p(null) at AUROC 0.298 `[P]`.
11. **Selection is committed at prefill.** Re-Prefill: attention variance collapses after the first decoded token and the wrong answer's centroid never moves during decoding; a second, attention-guided prefill gives ScreenSpot-Pro 65.8 → 70.1 on Qwen3-VL-8B `[V: 2605.12549]` — the one training-free method aimed at "wrong candidate chosen", never run on natural images.
12. **The other test-time levers are spent for us:** zoom/crop's small-object ceiling is +3.34 and 183 of the 338 wrong boxes are on large objects `[P]`; coordinate refinement is capped at +2.49 `[P]`; consensus-based test-time training reinforces the error when all samples agree on the wrong object `[M4 §2]`.
13. **The only rejection-RL recipe with an over-refusal penalty (RC-GRPO) was never run on GroundingME**, its α knob trades N-acc 19.6-74.5 against P-acc 73.6-60.9, and it forced "None" by constrained beam search because no rollout contained it `[V: 2608.04698]` — an off-policy injection with no direction and no withdrawal.
14. **Nobody reports base pass@k on any hard REC subset, and no VLM RL paper reports "N items with base pass@k = 0, M solved unguided after training"** `[M1 §4, M2 §5, M3 §10]`. The charter's question is unanswerable from the literature as written; any of the runs below can make that table.
15. **Measurement traps that every proposal must carry:** GroundingME negatives text-separable at AUROC 0.92 → grey-image control; unparseable output counted as correct rejection → literal-null and parse-failure columns; false-null lands on Limited/Small first (40/41 of the pilot's false abstentions) → false-null by size; select thresholds on a frozen dev half, never on the 201 (`REJECTION-DEEP-DIVE` §2.5, §6).

**Net:** the answer must (a) change which object is chosen at prefill, (b) be made samplable during training by information the policy lacks and then withdrawn, or be trained through a comparison that needs no sampling, and (c) be measured unhinted, under the plain prompt, with the false-null-by-size row. Nothing published does all three for grounding.

---

## 2. Method proposals

The seven candidate mechanisms named in the brief are mapped as follows: (i) prefix-RL over a verdict block → **P1**; (ii) ExPO self-explanation → **P5**, also P1 arm B; (iii) pairwise box objective on same-image pairs → **P2**; (iv) TreeVGR-style candidate-list reward + clause-verdict reward → P1's reward; (v) crop-plus-clause verifier, selector then reward → **P3**; (vi) Re-Prefill for natural-image REC → **P4**; (vii) presence decision outside the coordinate stream with masked box loss → folded into P1 and P2, rejected as a standalone capability method (§2.6).

P1 and P2 attack the 338 wrong-object positives and the 201 negatives with one mechanism each. P4 is the training-free instrument that changes the sampling distribution rather than selecting among samples.

Shared hardware facts: 4B LoRA GRPO with co-located vLLM fits the local 5090 at 28-31 GB; 8B GRPO does not and goes to RunPod 1×80 GB; WSL host ceiling 23 GB, so images stream and one model is resident at a time (`RL-DESIGN` §9.6). Shared guardrail suite per checkpoint: GME 1,005 free generation + RefCOCOg val + DocVQA 1k + AI2D 1k ≈ 2-3 GPU-h on 4B (`ideas-1` §0).

### 2.1 P1 — Verdict-Prefix RL (withdrawn per-candidate verdict block)

**Problem it solves, plainly.** The model chooses an object before it has checked the details. We make it write a short check-list — one line per same-category candidate, one tick or cross per detail — and only then the answer; during training we start the list for it and take that help away item by item, so the deployed model writes the list itself and, under the plain prompt, at least picks differently.

**Mechanism.**
- *Format.* After the prompt the model emits a fixed short block and an answer:
  ```
  1 [x1,y1,x2,y2] c1✓ c2✓ c3✗
  2 [x1,y1,x2,y2] c1✓ c2✓ c3✓
  3 [x1,y1,x2,y2] c1✗ c2✓ c3✓
  answer: 2          | answer: none
  ```
  c1..cn are the expression's atomic details, decomposed by a text-only LLM with wording kept verbatim (RL-DESIGN `expr_decompose`); symbols only, ≤ 60 tokens for 4 candidates × 6 details. On "none" no coordinates follow — the coordinate stream never sees a negative (the SAM 3 masked-box property for free).
- *Data.* Hard items only: positives the base boxes on a sibling (PAM gate 3: p(box) ≥ 0.99 landing on a same-category instance ≠ target), N0 and N0' negatives passing gate 3, 20% C controls (non-referent edited, answer = target box). Clause counts length-matched between positives and negatives. No easy items in the batch (POPE's ray interference).
- *Prefix source.* The whole block is known by construction for engine items: candidates = SAM 3 instances of the head noun (area ≥ 0.2%, left-to-right), verdicts = construction truth for the flipped clause plus Gemma4-12B per-(candidate, clause) verdicts for the rest (cross-family; `lineage.py`). Arm B: the policy's own answer-conditioned block (P5), kept only if it raises p(answer) ≥ 10× and passes the grey-image filter. PrefixRL says cross-family prefixes transfer, so arm A is the default and arm B is the in-distribution comparison.
- *Prefix schedule (the part that carries the evidence).* Prefix = the first ρ fraction of candidate lines, never the `answer:` line. Per-item ρ_i initialised by BREAD-style search (smallest ρ ∈ {0, .25, .5, .75, 1} giving ≥ 1 success in 8); lowered one step only after the item's group has ≥ 1 success at its current ρ (MFC's monotone frontier); annealed to 0 by 70% of training; items still at ρ = 1 at 60% are dropped and counted. Gradients masked on prefix tokens (PrefixRL / AdaPrefix). Each group of 8 = 4 prefixed + 4 unprefixed copies of the *same* item (POPE 1:1). Optional OC-GRPO ratio π(y|x)/π_old(y|g(x)).
- *Reward.* R = R_answer + 0.3 · R_block. R_answer = `RL-DESIGN` §3.1 set-F1 with the over-refusal penalty (none on a positive = −0.5) and α = 0.5 negative-advantage scaling; forced-None rollouts are **not** used (the prefix replaces them). R_block, computed on the unprefixed part of the block only: coverage (fraction of instance masks ≥ 0.2% area matched by a candidate line at IoU ≥ 0.5) × precision (fraction of candidate lines matching an instance) × (0.5 + 0.5 · verdict), verdict = 1 iff the verdict vector matches truth on the flipped clause and on the target row. Precision term mandatory (TreeVGR: 0.0 without it).
- *Schedule.* Qwen3-VL-4B-Instruct, LoRA r = 32, 2k items, G = 8, 400 steps, low lr (rule 3), 10% format-diverse replay; 8B replication on RunPod. Logged every 25 steps: hint reliance, on-policy (unprefixed) success share on the hard set, over-refusal canary on held-out positives, forced-drop count.
- *Deployment / measurement.* Two protocols always reported: with the block (scoped claim: "the model enumerates and verifies") and under the plain GroundingME prompt without the block (rule 2). Plus the unsolved-set table: items with base pass@8 = 0 → solved unprefixed at the end.

**Why it should work on items the base cannot sample.** (1) Six independent prefix papers with the base-pass@k = 0 subset, unhinted measurement and a GRPO control all transfer (map item 3). (2) The block moves the decision after the verdicts; M2's account of the failure is commitment at prefill, and Rex-Thinker's per-candidate check is the one published REC step with a clean +13.8 elimination gain at +0.9 positives. (3) TreeVGR shows the intermediate step can be rewarded with a large gain when precision is guarded. (4) PrefixRL's Qwen → Llama result says a prefix need not come from the student. (5) The verdict block on a negative ends with a false clause for every candidate, so "none" becomes the only consistent completion — the answer is made samplable by the scaffold, not by a constant.

**Three closest papers and the delta.**
- PrefixRL 2601.18795 / AdaPrefix-GRPO 2607.07674 — output-space prefix with masked gradients, per-item ratio, annealed to zero, math. *Delta:* grounding with boxes and a "none" answer; prefix is a structured verdict block from construction truth and cross-family judges, not a teacher trace; measured under the plain prompt.
- Rex-Thinker 2506.04034 — per-candidate CoT from answer-conditioned GPT-4o traces, detector candidates in the prompt, SFT then GRPO, L2 negatives. *Delta:* candidates enumerated by the policy (no detector at test), the block learned by withdrawn-prefix RL rather than SFT, L4 negatives, over-refusal term, hint reliance and protocol swap reported.
- TreeVGR 2507.07999 — recall + precision reward on the trace's boxes, VQA. *Delta:* REC; coverage/precision over same-category instances plus a verdict-correctness term; prefix withdrawal on top.

**Genuinely new.** Prefix RL for grounding with withdrawal and unhinted measurement; the unlock curve drawn for a referring expression; cross-family verdicts as the prefix source; one mechanism for both failure types; the unsolved-set conversion table for a VLM.

**Kill experiment — the unlock curve (MFC Table 1 for REC).** *Decides:* whether partial scaffolding makes the correct answer samplable at all (the precondition the whole prefix family silently assumes) and what the block costs on Discriminative. *Design:* GME frozen dev half — 100 negatives, 100 wrong-object positives (from the 338), 100 correct positives (false-null watch); ρ ∈ {0, .25, .5, .75, 1} × 8 samples at T = 1 on Qwen3-VL-8B; candidates from SAM 3 head-noun prompts; verdicts from Gemma4-12B, never the policy; grey-image row at ρ = 1. *Pre-registered PASS (all four):* pass@8(none | ρ = 1) ≥ 25% on negatives; pass@8(GT box | ρ = 1) ≥ 40% on wrong-object positives; pass@8 at ρ = 0.5 ≥ 3× pass@8 at ρ = 0 on at least one side (a partial prefix must matter — otherwise the block is an answer leak, not a scaffold); the ρ = 0 block costs ≤ 3 Acc@0.5 on the 804 positives against the plain prompt. Grey-image: pass@8(none | ρ = 1, grey) must be < half of the real-image value. Fail on any → the family is dead for that side (positives and negatives judged separately). *Headline:* 4B LoRA, 2k items, 400 steps, two arms on identical data and seeds — P1 vs `RL-DESIGN` S3 (forced-None set-F1) — stop rule: unsolved-set conversion ≥ 15% on both sides under the plain prompt (AdaPrefix's 17% is the reference) AND GME dev Rejection ≥ 13 AND false-null ≤ 5% by size AND Discriminative ≤ 2 drop; hint reliance ≤ 0.5 at the end.

| step | dataset | model | GPU RAM | host RAM | GPU-h | where | person-days |
|---|---|---|---:|---:|---:|---|---:|
| candidates + verdicts | GME dev 300 items; SAM 3 then Gemma4-12B on ~300 × 4 × 6 (cand, clause) | SAM 3 5 GB; Gemma4-12B 24 GB (sequential) | ≤ 24 GB | 18 GB | 0.7 | local | 1 |
| unlock curve | 300 × 5 ρ × 8 = 12k generations ≈ 80 tokens | Qwen3-VL-8B vLLM | 22 GB | 18 GB | 1.0 | local | 1.5 |
| block-format cost | 804 positives at ρ = 0, greedy | 8B vLLM | 22 GB | 18 GB | 0.3 | local | — |
| headline, 2 arms | 2k S1 hard items; G = 8; 400 steps | 4B LoRA + vLLM | 28-31 GB | 20 GB | 2 × 12-15 | local (2 wk) or RunPod 1×80 GB (1 day) | 4 |
| eval | guardrail suite × 2 arms × 3 ckpts | 4B | 10 GB | 18 GB | 12-18 | local | 1 |
| 8B replication | same | 8B LoRA + vLLM | > 32 GB | — | 20 | RunPod 1×80 GB | 1 |

**Guardrails.** False-null on positives by size (Limited/Small and the ≥ 36-word Discriminative layer separately); Discriminative ≤ 2 drop under the plain prompt; grey-image control on the block (verdicts on grey must be at chance, else the block is read from the sentence); protocol swap with and without the block, `{"boxes": []}` → `{"bbox_2d": null}`, Acc@0.9 ≥ Acc@0.5 movement; hint reliance ≤ 0.5; enumeration recall of the policy's own candidate lines on positives ≥ 85%; over-refusal canary every 50 steps; RefCOCOg ≤ 1; literal-null and parse-failure columns.

**Main risk and its sign.** (1) *Flat curve below ρ = 1:* the model can copy the last verdict line but cannot produce a verdict itself — sign: pass@8 at ρ = 0.5 ≈ ρ = 0 while ρ = 1 is high; then only P2/P4 remain. (2) *The block is a thinking cost:* sign: ρ = 0 on the base drops Discriminative > 3; shorten to symbols-only or scope to the block protocol. (3) *Format learned, check not learned:* sign: hint reliance stays > 0.7 and the gain vanishes under the plain prompt — scope the claim like Rex-Thinker's, do not sell it as capability. (4) *Enumeration collapse:* sign: precision falls while coverage holds and the answer rate drops.

### 2.2 P2 — Sibling-Triplet pairwise objective on box outputs

**Problem it solves, plainly.** The model's confident wrong box on a sibling is never punished relative to the right answer, because the right answer is never produced for RL to reward. A comparison-based loss only needs the model to *score* the right and wrong answers, which it can always do, so it has a learning signal on exactly the items where sampling has none — for both failure types at once.

**Mechanism.**
- *Data (per source scene with ≥ 3 same-category instances).* Target T; sibling S = the instance the base boxes on the target's expression (gate 3 record). The writer (Qwen3.5-9B) writes e_T (n details) and e_S (same template, details re-filled for S); Molmo2 verifies uniqueness of both; e_T⁻ = e_T with one detail flipped so no instance satisfies it (Gemma4 zero-satisfier). Where the B-arm editor exists: I_edit (target's detail edited, e_T byte-identical, answer none) and C (sibling edited, answer b_T). Blind-detectability gate on the writer (text-only judge AUROC ≤ 0.60 for flipped vs original).
- *Responses in the model's own coordinate habit.* b_T and b_S are obtained by prompting the base with an unambiguous outline reference ("the <noun> outlined in red"), so both box strings are the model's own digit patterns (3,856 emitted coordinates were exact 0-1000 integers `[P]`), not GT digits; none = `{"boxes": []}`.
- *Loss (ℓ = log π_θ − log π_ref, β = 0.1), four comparison cells plus an anchor:*
  ```
  −log σ(β[ℓ(b_T | e_T)  − ℓ(b_S | e_T)])      # positive side: the 338 failure
  −log σ(β[ℓ(b_S | e_S)  − ℓ(b_T | e_S)])      # sibling redirect: symmetric
  −log σ(β[ℓ(none | e_T⁻) − ℓ(b_T | e_T⁻)])    # negative side
  −log σ(β[ℓ(none | e_T⁻) − ℓ(none | e_T)])    # none must be conditional, never a prior
  + max(0, ℓ_ref(b_T | e_T) − ℓ(b_T | e_T))     # mDPO anchor: the right box may not get less likely
  ```
  Image-side cells (I vs I_edit, e_T fixed) use the same form (S-VCO's symmetric attend/reject). Cells where the model is indifferent — ℓ(b_T|e_T) ≈ ℓ(b_S|e_T), the signature of the 338 — sit at the sigmoid's steepest point and get the largest gradient; cells it already gets right get ≈ 0. That is SAM 3's "keep only what the current model accepts" rule expressed in the loss rather than in the data.
- *Mix.* 60% triplets, 20% C controls and plain positives (chosen b_T vs rejected none), 20% format-diverse replay (short RefCOCOg-style items, a text-answer task) for rule 3. ORPO-style reference-free variant as a second arm (Visual-Idk: SFT −17.9 vs ORPO −5.5 `[S-B: 2604.26419]`).
- *Schedule.* 4B LoRA, 3k triplets, 2 epochs, lr 5e-7 (rule 3's cheap defence), reference = adapter disabled; no rollouts, no vLLM, ≈ 3 GPU-h. Trained in the `{"boxes": …}` format and evaluated under GME's native format — the protocol swap is built in.

**Why it should work on items the base cannot sample.** The pairwise route needs no sampling of the right answer (M3 route R2), and it is the only data route with a consistent held-out record and below-chance → above-chance cases (map item 8). S-VCO beat DPO by +5 on the same pairs by making the loss symmetric; FINER-tuning transferred with no alignment tax. The positive-side cost of "none" comes from rewriting the output stream (map item 9); here none is one cell of four, conditional on a twin, never a fixed bonus, and the sibling-redirect cell forbids "changed sentence ⇒ none". The redirect cell also directly trains the 338 case: the same image, the same category, the answer moves with the details.

**Three closest papers and the delta.**
- S-VCO 2502.13928 — symmetric attend/reject loss over (image, edited image) pairs, free-text answers. *Delta:* box and none outputs, text-side pairs on one image, sibling-redirect cell, on-policy coordinate strings.
- FINER-tuning 2603.17662 — DPO on clause-level negatives, yes/no answers, held-out DASH +2 to +6. *Delta:* coordinate outputs with two competing same-category objects, negatives derived from the policy's own confident wrong box.
- mDPO 2406.11839 / OViP 2505.15963 — image-side DPO with an anchor; failure-derived rendered negatives. *Delta:* the rejected response is the base's own sibling box on the same real image (a minimal edit in *expression* space), and the anchor protects the right box's likelihood rather than a free-text response.

**Genuinely new.** The first pairwise objective on box outputs (M3 §5.3: none exists); the sibling-redirect cell; hardness weighting that emerges from the loss; on-policy coordinate strings via outline prompting so the likelihood objective compares like with like.

**Kill experiment.** *Precondition (0.3 GPU-h):* on 200 S0 triplets, the base's margins ℓ(b_T|e_T) − ℓ(b_S|e_T), ℓ(b_S|e_S) − ℓ(b_T|e_S) and ℓ(none|e_T⁻) − ℓ(none|e_T). Expect medians near 0 (indifference) or negative (wrong preferred). If > 80% of triplets already show > 2 nats in the right direction, the triplets are too easy and gate 3 must be re-run; if the none margin is undefined because none is unparseable at the base, that is the D8 wall and the none cells are trained anyway (that is the point). *Headline (3 GPU-h train + 3 eval, the cheapest full result in this file):* on held-out hard positives (S1 held-out plus the GME dev wrong-object split) wrong-object rate 41% → ≤ 33% at false-null ≤ 3% by size; GME dev Rejection ≥ 10 (the S3 bar) at Discriminative ≤ 2; Acc@0.9 gain ≥ Acc@0.5 gain under GME's native format. Grey-image: after training, the none margin on grey images must stay ≈ 0. Fail on the positive side alone is still a result worth a paragraph (likelihood training on coordinates cannot re-select).

| step | dataset | model | GPU RAM | host RAM | GPU-h | where | person-days |
|---|---|---|---:|---:|---:|---|---:|
| triplets | S0 A-arm + sibling-expression call + gate 3; 3k triplets ≈ 800 scenes | Qwen3.5-9B 19 GB → Qwen3-VL-4B 9 GB → Molmo2-8B 17 GB → Gemma4-12B 24 GB (sequential) | ≤ 24 GB | ≤ 20 GB (stream) | 6-8 | local | 2 |
| precondition | 200 triplets × 6 log-probs | 4B HF | 10 GB | 18 GB | 0.3 | local | 0.5 |
| DPO | 3k triplets × 2 epochs, LoRA | 4B | ~20 GB | 18 GB | 3 | local | 1 |
| eval | guardrail suite × 2 arms | 4B | 10 GB | 18 GB | 6 | local | 0.5 |
| 8B | same | 8B LoRA | ~30 GB | — | 6 | RunPod 1×80 GB (or local 4-bit, slower) | 0.5 |

**Guardrails.** False-null by size; Discriminative ≤ 2; grey-image on every margin (on grey all four margins must be ≈ 0); protocol swap built in; **Acc@0.9 vs Acc@0.5 tracked per epoch** (coordinate drift is this proposal's specific hazard); Q-3 coordinate-convention probe before and after; over-abstention canary every 200 steps; RefCOCOg ≤ 1; C-control flip rate ≈ 0.

**Main risk and its sign.** (1) *Likelihood training distorts coordinates:* sign: Acc@0.9 falls while Acc@0.5 rises, or the Q-3 probe moves — anchor, low lr, replay are the defences; if it persists, restrict the box cells to the last two coordinate tokens' logits (a "which object" contrast) and keep the none cells. (2) *Text shortcut on e_T⁻:* sign: gain on N0 ≫ N0' — blind-detectability gate; the redirect cell already prices it. (3) *The wrong box is strongly preferred, not indifferent:* the gradient exists but the anchor fights it — sign: ℓ(b_T|e_T) not rising in epoch 1; raise β on the box cells only.

### 2.3 P3 — Ask-Not-Tell critic: a verifier that sees less, trained on edit provenance, used as selector then reward

**Problem it solves, plainly.** Every verifier that is as blind as the policy adds nothing (map item 10). A verifier can only beat the policy on our items if it looks at less (one candidate, one detail) and is asked a question it cannot answer from the sentence. So we turn each detail into a question with the answer withheld, ask it about one outlined candidate, and compare the reply with the withheld value.

**Mechanism.**
- *Stage 0 — zero-shot instrument.* Decompose the expression into details; a text-only LLM turns each into (question, withheld value): "blue shirt" → ("What colour is the outlined person's shirt?", "blue"); "holding a red umbrella" → ("What, if anything, is the outlined person holding?", "a red umbrella"); a text detail → ("What text is written on the outlined object?", "STOP"). Ask the question about the outlined candidate (full image with red outline; crop as ablation; relational details always full image). A text-only judge compares the free reply with the withheld value → entailed / contradicted / unknown. Candidate verdict = min over details; set logic as `ideas-2` Idea 2. The value never enters the image-side prompt, so parroting, yes-bias (POPE-style) and the 0.92 text shortcut are impossible by construction, and the output is graded (k of n details) rather than binary.
- *Stage 1 — train the critic.* Cross-family base (Gemma4-12B or Molmo2-8B, LoRA; `lineage.py`), input = outlined candidate + question, output = value. Labels from provenance the policy never sees: N0' edits give (target, flipped detail, new value from the edit instruction); removals give "absent"; every unflipped (candidate, detail) gives the writer's value verified by Molmo2; C controls enter as *entailed* so "edited ⇒ contradicted" cannot be learned. 3-5k tuples.
- *Stage 2 — selector.* Over SAM 3 or policy-enumerated candidates, run the critic per (candidate, detail); unique all-entailed → its box; none → none; several → highest min score. Reported as a *protocol* (rule 2(c)), never credited as policy capability.
- *Stage 3 — reward.* Replace P1's construction-truth verdict term with the critic's verdicts on **unlabelled** crowded images (SA-1B-like scenes with writer expressions and no flip record), so P1 can train on GME-regime images beyond the engine's labelled items; coverage/precision terms unchanged.

**Why it should work.** M4's conclusion: the only two ways a verifier can know more are a smaller input with an atomic claim and labels the policy never sees — this does both; STV (1.5 → 21) and R-OPSD (20.3 → 30.5) are the pass@k ≈ 0 → solved cases and both used privileged training signal. Vision-SR1's "the answer must be recoverable from the description alone" is the principle behind withholding the value. The project's closed finding that a single detail *can* be judged in isolation (charter) means the question is the right unit; what was never tested is a form that cannot be shortcut.

**Three closest papers and the delta.**
- `ideas-2` Idea 2 / M4 T2 — per-detail yes/no on an outlined candidate, zero-shot; train the critic on edit pairs. *Delta:* value-withheld questions instead of yes/no (kills yes-bias and text parroting; gives a graded signal), cross-family trained critic, use as a label-free reward on unlabelled images.
- Vision-SR1 2508.19652 — describe, then answer from the description without the image. *Delta:* per-detail, per-candidate, value-withheld; the checker is a different model family, not the policy.
- R-OPSD 2608.11191 — a labelled reflector (89.5%) turned into token-level advantages, GUI. *Delta:* natural-image REC, detail-level provenance labels, and a critic whose input is one candidate plus one question.

**Genuinely new.** The value-withheld question as the verifier unit; a provenance-labelled cross-family critic; the crop+detail vs full-image+whole-expression comparison stratified by detail type; label-free RL extension to images without construction truth.

**Kill experiment.** *Stage 0 isolation ladder.* On 200 S0 N0 items (flipped detail known), 100 GME dev Discriminative positives (GT box; all details true) and the 100 dev negatives (own box): contradicted-rate on the flipped detail vs unchanged details, AUROC per sub-axis; decisive Text-only variant first (≈ 0.25 GPU-h). *Pre-registered:* Text ≥ 0.85, Appearance/State ≥ 0.70 (M4 T2's bars); false-contradiction on positives' GT-candidate details ≤ 10%; ask-not-tell must beat the yes/no form on the same items by ≥ 0.05 AUROC (else the extra call is not worth it); cross-family zero-shot row (Molmo2 / Gemma4) ≥ the Qwen row; grey-image contradicted-rate ≈ uniform. *Stage 1 kill:* trained critic held-out AUROC ≥ 0.85 on N0' per sub-axis and C-control AUROC ≈ 0.5 (no artifact learning); below that it is not funded as a reward. Thresholds set on removal controls, never on GME.

| step | dataset | model | GPU RAM | host RAM | GPU-h | where | person-days |
|---|---|---|---:|---:|---:|---|---:|
| questions | 400 expressions × ~6 details, text-only | Qwen3.5-9B or CPU API | 19 GB | 18 GB | 0.2 | local | 0.5 |
| Stage 0 ladder | ≈ 3-10k VQA calls (1-4 candidates × 6 details) | 8B 22 GB / Molmo2 17 GB / Gemma4 24 GB, sequential | ≤ 24 GB | 18 GB | 1.5-3 | local | 1.5 |
| Stage 1 critic | 4k tuples, LoRA | Gemma4-12B (RunPod 1×80 GB) or Molmo2-8B (local ~28 GB) | 28-40 GB | 20 GB | 4 | RunPod / local | 2 |
| Stage 2 deploy eval | 1,005 × K × C ≈ 25k calls | critic | ≤ 24 GB | 18 GB | 8-10 | local | 0.5 |

**Guardrails.** False-null on positives by size, with enumeration misses reported separately from verification misses; Discriminative ≤ 2 under the plain prompt (the selector is scoped anyway; the reward use must show up through P1 under the plain prompt); grey-image; "unknown" treated as entailed for the deployable rule, never as contradicted; relational details never cropped; sibling-swap test on 50 positives (the same reply for the GT candidate and a sibling means the outline is ignored — IVT's failure).

**Main risk and its sign.** (1) *The value is not recoverable at 4-12B on crowded natural images:* sign: false-contradiction > 15% on Appearance/State positives — then the critic is Text-only (≤ 25 points of the 201) and is reported as such. (2) *Artifact learning in Stage 1:* sign: C-control AUROC > 0.6. (3) *Critic hacking in Stage 3:* the policy places candidate lines where the critic says "entailed" — sign: precision falls while coverage holds (TreeVGR's guard is the fix).

### 2.4 P4 — Clause-Conjunction Re-Prefill (training-free; changes the sampling distribution)

**Problem it solves, plainly.** The wrong object is chosen during the very first pass over the picture and the sentence, and nothing later changes it. We run that first pass twice: the second time the model's attention is told to favour the image regions that *every* detail supports, so the distribution of boxes it samples from changes — no training, no picking among samples.

**Mechanism.**
- *Pass 1:* plain prompt; record, at the mid-layer band Re-Prefill uses (cross-layer consistency ρ = 0.8), the attention from each detail's text tokens (spans known from the verbatim decomposition) to the image tokens; also the head-noun map (what the model currently uses).
- *Conjunction:* combine the per-detail maps by a soft AND (product of normalised maps, or min). *Pass 2:* second prefill with key visual tokens = top γ = 0.1 of the conjunction map (Re-Prefill's layer-wise second prefill), optionally down-weighting the tokens of the currently attended sibling (head-noun map minus conjunction map); then decode.
- *Two uses.* (a) Inference, training-free (scoped +x). (b) **Rollout generator:** pass-2 samples are complete generations in the model's own format (rule 3 preserved), injected 2-of-8 into GRPO groups with importance weights π_θ(y|x)/π_2(y|x) (ICPO's regularised weight), hook removed at deployment — the alternative to `ideas-1` Idea 2's probe-steered rollouts that needs no probe, no absence direction, and works on positives.
- *Negatives:* re-prefill cannot create a "none". On a negative the conjunction map should have no strong peak (every candidate fails one detail); its peak height / entropy is logged as an absence *instrument* only, with the grey-image row and against the 0.92 text floor and the 0.298 p(null) baseline.

**Why it should work.** Re-Prefill's diagnosis is our failure verbatim — selection committed at prefill, attention variance collapsed after the first token, the wrong answer's centroid never moves — and its fix moved Qwen3-VL-8B +4.3 on ScreenSpot-Pro while suppressing distractor activations; attention-based scores are the only box selectors above greedy (ACS-Free 63.4 vs greedy 61.4; MTLA AUROC 0.89). `CAPABILITY-MECHANISMS` §3 says perception is not the bottleneck, readout is: if the per-detail attention exists mid-layer, the readout can be redirected without new weights.

**Three closest papers and the delta.**
- Re-Prefill 2605.12549 — one key-token map from cross-layer attention consistency, GUI. *Delta:* natural-image REC; per-detail maps combined by conjunction; used as an RL sampler.
- `ideas-1` Idea 2 — steered rollouts from a probe direction at one layer. *Delta:* no probe, no trained direction; steering is attention re-weighting from the expression's own detail spans; covers positives.
- Illusion of visual re-examination 2605.15864 — 2× attention amplification during reflection (36.6 → 54.8). *Delta:* amplification at prefill, detail-targeted, with a box output, plus a census of whether the target is in the key set at all.

**Genuinely new.** Conjunction-of-details attention as a training-free re-selection; the key-token census on the 338; re-prefill as a group-rollout source; the conjunction-map peak as an absence instrument with the grey-image control.

**Kill experiment.** *Decides:* whether the correct object is reachable by re-weighting at all. (1) *Census* on the dev half of the 338 (169 items): fraction where the GT object's image tokens are inside the conjunction key set (γ = 0.1) — pass ≥ 30% (M4 T1's bar); this is the same forward pass as `CAPABILITY-MECHANISMS` §7 step 1's readout census, run both at once. (2) *Effect:* pass@8 at T = 1 of a GT-hitting box after re-prefill ≥ 10% (from ≈ 0) on those items, with ≤ 2% new errors at greedy on the 466 correct positives. (3) On 100 dev negatives: churn (fraction re-boxing a different sibling) and the peak-height absence AUROC with grey-image. If (1) passes and (2) fails, the attention is right and the readout is not — evidence for P1's block rather than a dead end.

| step | dataset | model | GPU RAM | host RAM | GPU-h | where | person-days |
|---|---|---|---:|---:|---:|---|---:|
| census + re-prefill | 169 + 100 + 466 items, hooks, 8 samples on the 169 | Qwen3-VL-8B HF eager (no vLLM) | 22-24 GB | 20 GB | 1.5 | local | 2-3 |
| as RL sampler (only if P1's curve fails and P4 passes) | 2k items, 2-of-8 re-prefilled | 4B LoRA + HF generation for the 2 | 28-31 GB (+40% wall) | 20 GB | 12-18 | local | 3 |

**Guardrails.** New-error rate on correct positives by size (this proposal cannot produce a false null, so this is its false-null row); Discriminative ≤ 2 at greedy under re-prefill (same plain prompt, so rule 2's protocol condition holds by construction); grey-image: the conjunction map must be flat on grey; Acc@0.9 vs Acc@0.5.

**Main risk and its sign.** Attention is already single-peaked on the sibling at p > 0.999, so nothing is left to promote — sign: census < 30%, and the conjunction map correlates > 0.9 with the head-noun map (the details carry no attention mass of their own, which would also explain D2 and be a diagnostic result in its own right).

### 2.5 P5 — Self-Explain positives (ExPO for boxes and for "none")

**Problem it solves, plainly.** A group with no correct answer needs one correct sample that the model finds natural. A teacher's text is foreign and collapses training (POPE 13.6 → 2.0; LUFFY on Llama below base); ExPO shows the model's *own* explanation, written while being told the answer, is natural and works (MATH L5 pass@4 2% → 23%, base 4% pass@64).

**Mechanism.** For each hard item with a verified answer a (box_T, or none plus the flipped detail), prompt the policy with image + expression + "The correct answer is a. In ≤ 40 words say which same-category candidates exist and why each is or is not the referent, then give the answer." Sample 8. Keep an explanation y only if (i) p_θ(a | image, e, y) ≥ 10 × p_θ(a | image, e) — it raises the answer (ExPO property 2); (ii) p_θ(y | grey, e) / p_θ(y | image, e) ≤ 0.1 — it reads the image; (iii) it parses to P1's block/answer format if that format is adopted. The kept y is inserted as one off-policy member of the GRPO group (ExP-GRPO), advantage on the pooled rewards, LUFFY/ExPO policy-shaping on low-probability tokens; 20% C controls so "explain why none" is priced where it is wrong. The same y is P1's arm-B prefix source and P2's chosen response if the block format is used.

**Why it should work.** ExPO's two properties (likely under the current policy; raises p(answer)) are the diagnosis of every off-policy failure in M1; the "expert" here is a label, which is exactly what our engine produces; the explanation is generated by the policy so the format distribution is preserved (rule 3).

**Three closest papers and the delta.**
- ExPO 2507.02834 — answer-conditioned self-explanation as the positive sample, LLM math. *Delta:* a VLM, a box and a "none" answer, the grey-image filter, C controls.
- LUFFY 2504.14945 — one teacher trace in the group with policy shaping. *Delta:* the off-policy member is the policy's own answer-conditioned generation.
- Rex-Thinker HumanRef-CoT 2506.04034 — GPT-4o answer-conditioned traces, then SFT. *Delta:* same-model traces used as RL group members, not SFT targets; a none answer; nothing in the prompt at test time.

**Genuinely new.** ExPO in a VLM; ExPO for an abstention answer; the grey-image filter on explanations as a rationalisation detector.

**Kill experiment — explanation lift (shares a run with P1's unlock curve).** 100 dev negatives + 100 dev wrong-object positives, 8 answer-conditioned explanations each. *Pre-registered:* median p(none | y) ≥ 0.05 on negatives (from a 3.4e-4 floor; M1's bar) and median p(box_T | y) ≥ 0.2 with box_T rank-1 among candidates on ≥ 40% of wrong-object positives; median grey ratio ≤ 0.1. Fail on the grey ratio = rationalisation → dead. *Headline:* ExP-GRPO vs S3 forced-None vs P1 prefix on the same 2k items, 4B LoRA, 300 steps; stop rule as P1's, plus the on-policy success share must rise over training (else this is SFT in disguise).

| step | dataset | model | GPU RAM | host RAM | GPU-h | where | person-days |
|---|---|---|---:|---:|---:|---|---:|
| explanation lift | 200 items × 8 explanations + p(a|y) and grey scoring | 8B vLLM + HF scoring | 22 GB | 18 GB | 0.6 | local | 1 |
| headline arm | 2k items, 300 steps, 1 explanation per group | 4B LoRA + vLLM | 28-31 GB | 20 GB | 12-15 | local / RunPod | 2 |

**Guardrails.** The six shared ones; the share of group successes that are the injected explanation vs on-policy, per step; p(a | no explanation) per checkpoint (the hint-reliance analogue).

**Main risk and its sign.** The explanation is a sentence-level rationalisation ("no candidate is striped") — sign: grey ratio ≈ 1. Second: p(none | y) stays < 0.05 — the model cannot say none even when told; then P1 at ρ = 1 fails too (same wall), which is why the two kills are run in one pass.

### 2.6 Rejected or folded, and why

- **(vii) Presence decision outside the coordinate stream with masked box loss.** Folded into P1 (the `answer:` line after the verdicts is a separate decision and no coordinates are emitted on negatives) and P2 (none is one conditional cell, never SFT'd at a ratio). As a standalone head (`ideas-1` Idea 1, `ideas-2` Idea 3) it is rejected *as a capability method*: it cannot touch the 338 positives, and rule 2 says an abstention-head gain does not transfer; it stays a deployment safety component measured as risk-coverage.
- **Group Revision with the candidate list** (M1 Idea C; 2605.15951): folded into P1 as arm C (hint = failed first attempt + candidate list, relative-improvement reward), funded only if the unlock curve passes; its recovered fraction of all-fail groups is unpublished and its plain-REC transfer is +1.0.
- **On-policy distillation from Qwen3-VL-32B-Thinking:** deferred to after P1/P5 create support (M1 #5; VOLD: fails without alignment).
- **Zoom/crop, best-of-N selectors, MCC consistency, CANL/GUI-RCPO, LFPR:** rejected (M4 §3: Discriminative +0.1, ceiling +3.34, consensus = the error).
- **Difficulty curricula and self-play as the entry mechanism:** rejected (they aim at 50% success by design); the rung ladder (`ideas-3` I3-4) remains the *climb* after P1/P5 create a non-zero pass rate, run sequentially, never mixed into the hard batch (POPE).
- **A single-detail judgment probe:** not re-proposed (closed by the charter); P3 asks about details only in the value-withheld form and only as a verifier, never as a claim about the policy.

---

## 3. Rankings

### 3.1 Information per person-week (which kill settles the most, soonest)

| rank | item | cost to the decision | what it settles |
|---|---|---|---|
| 1 | **P1 unlock curve + P5 explanation lift** (one shared pass) | ≈ 2.6 GPU-h, 3 person-days | whether the answer can be made samplable by scaffolding at all, for positives and negatives separately; the block's Discriminative cost; whether the base can say "none" when told. Decides the entire "guided-and-withdrawn" family before any training. |
| 2 | **P2 precondition → headline** | 0.3 + 3 + 6 GPU-h, 4 person-days | the first pairwise-on-boxes result; either it moves the 338 or it shows likelihood training on coordinates cannot re-select — both publishable, and it needs no rollouts. |
| 3 | **P3 Stage 0 ladder** (Text-only first) | 0.25-3 GPU-h, 2 person-days | whether any verifier stronger than the policy exists per sub-axis; gates every selector and the label-free reward. |
| 4 | **P4 census** | 1.5 GPU-h, 2-3 person-days | whether the target is attended at all; a training-free +x if yes; diagnostic for P1 if no. |
| 5 | **`RL-DESIGN` S3 (forced-None set-F1 GRPO)** | 12-15 GPU-h | **Sits here as the mandatory control arm for P1/P5, not as a bet.** A forced constant has no direction and no withdrawal (M1 verdict 3), and `ideas-1` §0 predicts Rejection well under 13 on GME. Run it only alongside P1's headline on identical data and seeds, never alone. |
| 6 | P1 / P5 headline training | 25-45 GPU-h, 6 person-days | the capability claim itself; funded only by rank 1. |

### 3.2 Paper quality if every kill passes

1. **P1** — a mechanism paper: guided-and-withdrawn RL extends grounding capability; the unlock curve as a new instrument for REC; the unsolved-set conversion table nobody has published for a VLM; one mechanism for wrong-object positives and negatives; protocol-swap and hint-reliance reported.
2. **P2** — an objective paper: pairwise selection on box outputs with a sibling-redirect cell; cheap to replicate at 8B; a strong out-of-distribution story if held-out transfer appears (the pairwise literature's signature).
3. **P3** — a verifier paper: a verifier that sees less beats the policy; provenance training; label-free RL beyond the engine's items.
4. **P5** — ExPO in a VLM and for abstention; likely a section of P1 rather than a standalone paper.
5. **P4** — a training-free +x on natural-image REC; workshop-grade unless the census itself becomes the story ("the details carry no attention mass").
S3's place in paper terms: the baseline row in every table (RC-GRPO on GroundingME, the missing experiment survey-A named).

---

## 4. Sequence — the first two weeks, concretely

**Week 1 — instruments (all local, ≈ 8 GPU-h, one model resident at a time).**
- Day 1: freeze the GME dev/test split (100/101 negatives; 169/169 of the 338 wrong-object positives; 233/233 correct positives) with `freeze.py`, per `REJECTION-DEEP-DIVE` §6. Build `configs/models/qwen3vl-4b-instruct.yaml` and rerun the Q-3 coordinate probe (`RL-DESIGN` §9.7 step 1, half a day). Draft the block format and the sibling-expression writer prompt (`expr_write_sibling`) as `UNVERIFIED` → verify.
- Day 2: run `RL-DESIGN` S0 A-arm (≈ 50 min GPU) extended with the sibling call — this produces the N0 triplets P2, P3 and P5 need. Read gates 1-3; if gate 3's load-bearing rate fails, switch image source before anything else (`RL-DESIGN` §9.7 step 3).
- Day 3: SAM 3 candidate pass on the GME dev half and the S0 scenes; Gemma4-12B per-(candidate, detail) verdicts and ask-not-tell replies in one residency (≈ 1.5 GPU-h) — this single pass feeds P1's prefixes and P3's Stage 0 ladder.
- Day 4: P1 unlock curve + P5 explanation lift (≈ 1.6 GPU-h) + block-format cost on the 804 (0.3 GPU-h). Read against the pre-registered thresholds, positives and negatives separately.
- Day 5: P4 census and re-prefill effect (1.5 GPU-h, shares the readout-census forward pass); P2 precondition margins (0.3 GPU-h). End of week: a one-page go/no-go table for P1-P5 with the four numbers each kill pre-registered.

**Week 2 — the two cheapest headlines, and the control.**
- Days 6-8: S1 partial build (≈ 1.5-2k certified items with sibling expressions and C controls, ≈ 10 GPU-h local, streaming images, ext4 parquet). Human spot-check of 50 negatives and 50 sibling expressions with the existing `idea91/human/app.py` (1 h).
- Days 8-9: **P2 headline** locally (3 GPU-h train + 6 eval, two arms DPO/ORPO). This is the first number that can move the 338.
- Days 8-10 in parallel on RunPod 1×80 GB (≈ 1 day, only if P1's curve passed on at least one side): **P1 vs S3** on identical data and seeds, 4B LoRA, 400 steps, guardrail suite at 3 checkpoints; hint reliance and unsolved-set table logged. If the curve failed but P4 passed, replace the prefixed copies with re-prefilled rollouts (P4 as sampler) and run that against S3 instead. If both failed, week 2's RunPod budget is not spent and the report says why.
- Day 10: P3 Stage 1 critic training is started only if Stage 0 passed on ≥ 2 sub-axes (RunPod, 4 GPU-h); otherwise P3 is reported as Text-only and the block's verdict reward stays construction-truth.

Deliverable at the end of week 2: the go/no-go table, the P2 result under GME's native format with the false-null-by-size row, and (if funded) the P1-vs-S3 comparison with the unsolved-set conversion column — the table the literature does not have.

---

## 5. What I could not verify and what I assumed

- All paper numbers are taken from M1-M4's `[VERIFIED]` rows; I opened no new sources (the session's search budget is exhausted). Group Revision's id is 2605.15951 per M4; M1 could not recover it.
- I assume the OpenImages instance bank can supply GME-regime negatives (PAM's precondition, `ideas-3` I3-1, unrun); if its survivors' median log10 p(null) is above −8, every proposal's training data moves to an SA-1B-like source and the week-2 schedule slips by the I3-7 build.
- P2's on-policy coordinate strings via outline prompting assume the base follows a red outline; IVT found drawn overlays ignored by a 4B *for self-verification*, not for referring — the P2 precondition checks that b_T and b_S land on the right instances (IoU ≥ 0.5) before any training.
- The 8B thinking checkpoint's parse-failure rate under the block format is unknown; all P1 measurements use the Instruct checkpoint.
- Thresholds (25% / 40% / 3× / ≤ 3 points; 0.85 / 0.70; 30%; 0.05 / 0.1) are inherited from M1 Idea A, M4 T1/T2 and `ideas-1`/`ideas-2` so that the Skeptic can trace each to a survey rather than to me; none was tuned on the 201.
