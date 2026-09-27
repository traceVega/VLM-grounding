# Project-internal evidence (VLM-grounding repo, 2026-09-24) — for the report writer

These are measured facts from this project, not from the literature. Cite them as "[project]" and use them as the constraints the design must satisfy. Paths are repo-relative unless absolute.

## Goal and setting
- Target: Qwen3-VL-4B-Instruct (LoRA) trained locally on one RTX 5090 (32 GB VRAM, 23 GB usable WSL RAM); 8B fits for inference/eval and for LoRA SFT at the memory cliff. Cloud GPUs (RunPod) possible when justified.
- Success criterion set by the user: a *significant* gain on GroundingME Rejection (201 items, human-falsified one clause in a ~54-word description, category present) while every other evaluated dimension stays neutral or improves: GroundingME positives by dimension (Discriminative 204 / Limited 300 / Spatial 300), own held-out scenes, gray-image control, RefCOCO/+/g val samples.
- GroundingME images/text are research-licensed: local evaluation only, never sent to APIs, never used for training.

## What the overnight PoC (2026-09-23/24) established — notes/POC-TRAIN-RESULTS-2026-09-24.md
- Baseline Qwen3-VL-4B: GME rejection 0.000 (Σ p(null) over 201 items = 2.5e-4), positives 0.398 (Disc 0.559 / Lim 0.377 / Spa 0.310), false-null on positives 3.2% (almost all Limited/small objects); RefCOCO/+/g 0.903/0.843/0.870. On a gray image the base answers null for 98.5% of rejection items and 97% of positives, so the gray control must be read as the *gap* between those two rates.
- Training data: 308 reviewed items (110 target positives, 105 sibling positives, 93 one-detail-flipped negatives) from 116 OpenImages scenes; scene-level split, 78 train negatives. Plus 193 Gemini-verified cross-scene negatives (a reviewed description on another same-category image).
- Four recipes, one trade-off line (screening subset = all 201 rejection + 150 stratified positives, baseline recomputed on the same items):

| recipe | GME rejection | GME positives (base 0.447) | GME false-null | RefCOCO mean Δ | p(null) AUROC rej-vs-pos |
|---|---|---|---|---|---|
| SFT lr1e-4 3ep (neg 30%) | 0.308 | 0.300 | 0.367 | −9.3 | 0.378 |
| SFT + 193 cross-scene neg + null-unlikelihood | 0.139 | 0.380 | 0.187 | −7.2 | 0.348 |
| RC-GRPO on SFT (forced null, α=0.5, −1 false-null) | 0.144 | 0.393 | 0.153 | −4.5 | 0.334 |
| ViSurf-style GRPO from base (GT injected, α=0.5, −0.5) @step105 | 0.154 | 0.380 | 0.173 | −2.8 | 0.389 |

- In every recipe GME rejection rate ≈ GME positive false-null rate. p(null) AUROC (rejection vs positives) on GME is 0.26 base and 0.33–0.39 trained, i.e. below chance: the trained null mass is *higher* on hard positives (small objects, spatial) than on rejection items. In-domain (own held-out scenes) the same models discriminate at AUROC 0.83–0.85 (negatives 60–73% null vs positives 9–15% false-null). Conclusion: our negatives unlock the null path and teach in-domain flip detection; clause verification does not transfer to GME. Loss/reward changes only move along the line.
- Unlock speed: SFT reaches 67% null on held-out negatives by optimizer step 20 (lr 1e-4) but with 33% false-null on positives that never recovers; RC-GRPO cut false-null from 0.33 to 0.09 within 15 steps while keeping negatives at 0.73 in-domain.
- Full eval of the RC-GRPO candidate: rejection 0.144, positives 0.398→0.354 (Spatial −7, Disc −3.9), RefCOCO −5 to −9. The screening subset's Discriminative +2.7 was noise (150-item SE ≈ 4 points): compare only paired per-item or on the full set.

## Verify-first prompt probe (zero-shot, not benchmark protocol; same 201+150 items)
- Prompt = GME prompt + "name the most likely candidate, check every attribute/part/text/relation one per line true/false, then final JSON".
- 4B: rejection 0.109 (18 null + 4 unparsable), positives 0.447→0.333 with 15.3% false-null. 8B: rejection 0.144 (24 null), positives 0.347, false-null 0.147. Again rejection ≈ false-null: no discrimination.
- Qualitative: on boxed rejection items the model marks the falsified detail "true" ("Inside the helmet, a red gas mask is visible – true" when the mask is not red); sometimes it writes "true (she is wearing a white cap, not red; this is false)". At global resolution with the whole paragraph, the falsified detail is not perceived/verified.

## The datagen checker runs already measured crop-based clause verification (poc155, 155 scenes; test50c 50 scenes)
Setup: for each candidate instance, the checker sees the full image with the instance outlined in red plus a close-up crop, and answers one isolated clause (prompt `verifier_clause_instance_v2`; Gemini used a batched variant with all clauses of one candidate). Human reviewers (me + subagents with close-up contact sheets; user spot-checked) bucketed each scene; "ok" = the flipped clause is genuinely false for the target and no candidate satisfies the flipped description.

| checker (crop + isolated clause) | original target clauses judged "yes" (n=954) | flipped clause on target judged "no", all scenes (n=150) | flipped clause on target judged "no", human-usable scenes (n=70) |
|---|---|---|---|
| Gemma-4-12B | 0.724 | 0.907 | 0.914 (yes 0.086) |
| Qwen3-VL-8B-Instruct | 0.817 | 0.867 | 0.914 (yes 0.086) |
| Gemini 3.1 Pro | 0.872 | 0.907 | 0.929 (yes 0.029) |

Caveats: Gemini's numbers are inflated because its verdicts were the primary gate for selecting usable scenes; the 8B and Gemma runs were independent of selection. The 4B was not run as a checker (cheap to do: datagen `--checker` option would need a 4B entry). Reading: with a crop and one clause at a time, an 8B open model detects a falsified detail about as well as Gemini (~91%), whereas the same family at global resolution with the full paragraph shows no discrimination. The residual problem is the *false "no" on true clauses* (8B 18%, Gemma 28%, Gemini 13%), i.e. calibration, which is the over-refusal risk of any verification mechanism; we hold ~4–5k (candidate, clause) labels from Gemini plus human-corrected scenes to train it.

## Assets available for training without new API spend
- For 205 scenes: Molmo2 listener candidates (boxes) for the head-noun category; writer clauses (4–7 per description) for the target and the sibling descriptions; the flipped clause; Gemini per-candidate per-clause verdicts (yes/no/unclear); Gemma-4 and Qwen3-VL-8B verdicts on subsets; blind text-only judge results (a text-only judge finds the flip in 56/90 negatives — text-shortcut risk); reviewer buckets and Chinese notes per scene.
- Prompts: `configs/prompts/verifier_clause_instance_v2.txt`, `verifier_clauses_batched.txt`, `expr_decompose.txt` (paragraph → atomic clauses, local Qwen3.5-9B), `grounding_qwen3vl_verify_first.txt` (the probe).
- Code: `train/` (SFT with token weights and null-unlikelihood; hand-written GRPO with forced-null / GT injection, α scaling, over-refusal penalty, optional KL; eval suite with GME gray control and prompt probes; cross-scene negative builder; queue runner). Datagen: `datagen/` stages select → write → check → policy → sibling → blind → listen → reflip → report → export.
- Costs measured: Gemini 3.1 Pro batched checker ≈ $0.03 per scene-candidate set; SFT (260 items × 3 ep) ≈ 10 min alone; screening eval (879 generations) ≈ 20 min; GRPO 260 prompts × G=8 ≈ 30 min/epoch; SFT and eval must not run concurrently (mutual 10× slowdown at 27 GB + 10 GB).

## Earlier project syntheses the design should build on (read them)
- notes/REJECTION-DEEP-DIVE-2026-09-16.md — clusters A (whole-sentence presence channel), B (per-clause verification), C (expression-form ladder), E (hardness-gated negative engine), F (objectives in the coordinate stream incl. T4 paired reward "null never scored alone"), G (thinking/prompt); diagnosis D1–D16; GME text-only floor AUROC 0.92 (negatives detectable from words alone), GME rejection median 54 words, base null channel = "not visible" (over-refusal concentrated on tiny objects).
- notes/HARD-SAMPLE-METHODS-2026-09-16.md — three routes judged viable: withdrawn-prefix RL, per-candidate verification, pairwise on boxes; "pairwise cold start first, then S3".
- notes/RL-DESIGN-CANDIDATE-VERIFICATION.md — the earlier RL design for candidate verification (gates 1–4, set-F1 GRPO, forced empty set, over-refusal penalty).
- notes/CAPABILITY-MECHANISMS-2026-09-06.md — GroundingME loss is object selection among same-category candidates, not localization.

## The user's framing of this request (2026-09-24)
Two mechanisms seen in papers: (1) a designed CoT format enforced in SFT to "activate" pretrained visual reasoning, then amplified by RL; (2) agentic tool calls (crop, zoom) for visual reasoning. The user asks for a survey, careful independent thinking, new proposals welcome, and a judgment on which training mechanism is most reasonable and promising. Earlier in the conversation I sketched a three-call pipeline (propose candidates → per-candidate clause verification on crops → rule-based decision) with clause-level SFT loss; the report should evaluate that against the literature and the evidence above, not assume it.
