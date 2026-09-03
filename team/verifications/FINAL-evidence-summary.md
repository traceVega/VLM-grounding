# Final evidence summary (Verifier, 2026-09-02)

Scope: every idea file in `ideas/` at 2026-09-02 evening. For each idea: the premises it rests on with their verdicts, the A4 claim verdicts it depends on, any REFUTED or UNVERIFIABLE item still present in the file, and an evidence-quality grade. Facts only; no opinion on merit. Verdict vocabulary per CHARTER.md; claim ids refer to `verifications/claims-log.md` (V-001 to V-196); per-idea detail in `verifications/IDEA-xx-verifier.md`.

Grading rule (from the lead): **A** = all premises verified at primary sources; **B** = one or more premises PARTIALLY; **C** = a premise rests on an unverified or refuted claim. "Premise" here means the Section 1 motivating facts plus the three-closest-paper deltas of the novelty test. Details such as licences, costs and effect-size guesses are listed but do not set the grade unless the idea depends on them.

## A4 claim verdicts (anchors, unchanged since 2026-09-01)

| Claim | Verdict | Used by |
|---|---|---|
| (a) RefCOCO is saturated | VERIFIED (val/testA; RefCOCO+ testB not fully; label noise 14/24/5%) | 11, 13, 21, 91 (benchmark choice) |
| (b) Grounding training reduces hallucination | REFUTED as a general claim; PARTIALLY in narrow QA settings (Geigle 2024) | 92 (as the null it tests), 91 (motivation) |
| (c) RL with IoU rewards beats SFT | PARTIALLY (small data / OOD only) | 11 (OOD venue), 13 (difficulty-matched SFT control), 21 (not load-bearing), 22 (reward question only) |
| (d) Attention maps localize the referent | PARTIALLY (few heads, per-sample unreliable) | 11 (starting fact), 21 (secondary probe only), 91 (attention-mask arm) |
| (e) CLIP cannot localize | REFUTED as stated | none |

## Status board

| Idea | Status (file header) | Grade | One-line basis |
|---|---|---|---|
| IDEA-11 | SURVIVING | **B** | six latent-gap papers VERIFIED; starting fact is A4(d) PARTIALLY by design; Stage-4 novelty PARTIALLY; Molmo2 lineage PARTIALLY |
| IDEA-91 | SURVIVING | **A** | every Section 1 premise VERIFIED; three closest papers plus VISE VERIFIED at body level; ancestors added; PARTIALLY items are on non-premise details |
| IDEA-13 | REVISED (awaiting Skeptic round 2) | **B** | motivating facts VERIFIED; novelty PARTIALLY (same loop shape as Vision-Zero / VisPlay; VLM-generated engines are the incumbent); the "incumbents use no listener" delta is REFUTED for GroundingSuite (EVF-SAM filter at IoU 0.5, V-198); one REFUTED citation label (SPARK) in related work |
| IDEA-92 | UNDER REVIEW (revised pre-round 1) | **B** | Geigle null and benchmarks VERIFIED; novelty PARTIALLY (ViGoR, FaithScore, Woodpecker, GROUNDHOG now cited); Ground-What-You-See delta PARTIALLY |
| IDEA-21 | REVISED (awaiting Skeptic round 2) | **B** | bottleneck papers, pitch values, Qwen no-ablation, ScreenSpot-Pro and Ref-L4 facts VERIFIED; ExpVG magnitude premise PARTIALLY; two missing precedents (Shikra §6.2, 2402.07384) |
| IDEA-22 | REVISED (awaiting Skeptic round 2) | **C as written; B after the §1/§4 rewrite** | environment and Jedi facts VERIFIED; the framing premise "the one verified data point is Jedi / no controlled grounding-to-success study exists" is REFUTED by Agent S2 Fig. 6, MMBench-GUI Finding 2 and GroundCUA Table 4; the injection half has a close neighbour (GUI-RobustEval) not yet positioned |
| IDEA-31 | ABANDONED | not graded | abandonment reasons consistent with V-133 (COCO-Search18 target-absent fixations unreleased; data small and COCO-based) |

## IDEA-11: the latent grounding gap (SURVIVING)

Premises and verdicts:
- MLLMs "know where to look" (2502.17422), few-heads localization (2503.06287), attention drift (2605.11559), self-improving small-object grounding (2606.01612), read-then-regenerate for VTG (2605.21954), InnerZoom (2606.30084): all VERIFIED at primary sources (V-050, V-054, V-064, V-101 to V-105).
- Starting fact A4(d) "attention localizes": PARTIALLY, and the idea is built to measure exactly that partial result; this is the reason for grade B, not a defect the file hides.
- GroundingME rejection premise: VERIFIED with caveat (20 of 25 at 0% in non-thinking mode, literal null-box metric, V-111); the file now states it that way.
- Novelty of Stage 4 (read-out-derived reward): PARTIALLY; the attention-reward family (2604.13993, 2602.08241, 2607.01707, 2512.14044, 2606.26387) and SD-RPN / RAL (2509.16944, 2602.04884) are now cited and the delta restated (V-106 to V-110, V-149, V-150).
- Molmo2 "RefCOCO-free": PARTIALLY (no RefCOCO annotations; COCO images present, V-112); wording corrected.
- RLVR forgetting: strong claim REFUTED, comparative claim PARTIALLY, grounding-specific UNVERIFIABLE (V-115); the file keeps the general-capability regression table, which is the correct response.
A4 dependencies: (d) PARTIALLY, (c) PARTIALLY, (a) VERIFIED.
REFUTED items still present: none found (grep of the corrected phrases). UNVERIFIABLE items present: compute tiers; GroundingME vLLM evaluation code; which Qwen3.5 layers keep softmax attention.

## IDEA-91: interventional grounding verification (SURVIVING)

Premises and verdicts:
- Saturation, label noise, contamination of REC benchmarks: VERIFIED (A1, A2, V-015, V-004).
- Cirik 2018 image-only and shuffle results: VERIFIED with detail (Google-Ref; 71.2 / 73.1% top-2; shuffle drop 3.0 to 5.4 points, V-127).
- GroundingME rejection: VERIFIED with the non-thinking caveat (V-111), stated in the file.
- Three closest papers DeFacto, HalluSegBench, PAPO: VERIFIED as described (PAPO mechanism kept at abstract level per action, V-132); VISE (2606.27373) VERIFIED at body level and positioned first (V-153).
- Ancestors RISE, CSS, Elephant in the Room, FP-RefCOCO / SESAME, disagreement-predicts-error: VERIFIED and now cited; the "nobody has run that control" claim narrowed (V-152).
- Editors and data: Qwen-Image-Edit and FLUX.2-klein Apache-2.0 VERIFIED; COCO-Search18 terms and unreleased target-absent fixations VERIFIED and applied; Ref-L4 composition VERIFIED and applied (V-128, V-133, V-134).
A4 dependencies: (a) VERIFIED; (b) used as motivation in its REFUTED-as-general form; (d) PARTIALLY for the attention-mask comparison arm only.
PARTIALLY items on non-premise details: OpenRef "edit-free" (V-131), Molmo2 lineage (V-112), Syn-GRPO details beyond the abstract. UNVERIFIABLE items present: LaMa licence, editor latencies, compute table, expected edit success rates, the 0.6 AUROC gate (a convention). REFUTED items present: none.

## IDEA-13: referring games at MLLM scale (REVISED)

Premises and verdicts:
- GroundingME 45.1 with per-dimension scores, Ref-Adv 2026 shortcut findings, Qwen3-VL pseudo-label recipe: VERIFIED (V-063, V-111, V-017, V-004).
- Three closest papers (Yu et al. 2017, Language Self-Play, Syn-GRPO): VERIFIED; deltas hold (V-119).
- Novelty: PARTIALLY; the VLM-generated expression engines (2407.14563, GroundingSuite) are now the kill-experiment baseline and Vision-Zero / VisPlay are cited (V-116 to V-118), as requested.
- Drift and pragmatics lineage (Strub 2017, SIL, Lazaridou 2020, CLIP listener, Lee et al. 2019): VERIFIED (V-119, V-193).
- New related-work ids in the revision: VERIFIED (V-194) except **SPARK (2605.05546), which is text-only knowledge-graph self-play and is mislabelled as VLM self-play (REFUTED, V-195)**; GREx training detail PARTIALLY.
- Delta against the incumbent engines ("neither conditions on a distractor set nor verifies with a listener"): VERIFIED for 2407.14563 (cropped target, top-5 answers with no post-processing, V-197); **REFUTED for GroundingSuite on the listener half** (stage-3 noise filtering with EVF-SAM removes expressions whose predicted mask has IoU below 0.5, V-198); distractor conditioning not stated there (PARTIALLY). Baseline B3 must include that filter.
A4 dependencies: (a) benchmark choice; (c) PARTIALLY, handled by the difficulty-matched SFT control (V-120).
REFUTED item still present: the SPARK label (related work, not load-bearing). PARTIALLY items present: SPIN "frozen opponent" wording, Molmo2 lineage, GREx training detail, "model answerers" for IC-Seg / 2608.23978. UNVERIFIABLE: SAM 3 concept-prompt behaviour on noun lists; compute.

## IDEA-92: attributed description with a causal attribution-vs-text-correction experiment (UNDER REVIEW)

Premises and verdicts:
- A4(b) in its REFUTED-as-general form, i.e. Geigle 2024's null result, is now the Section 1 premise: VERIFIED at body level (V-148).
- DetailVerifyBench, GAVEL, BICR, Ground What You See: VERIFIED (V-138 to V-141); the Ground-What-You-See "no attribution in output" delta is PARTIALLY (not in its abstract).
- Three closest papers after the reframing (KAWHI, COPO, ClaimDiff-RL): VERIFIED as described (V-154 to V-156); VISE contrast VERIFIED (V-153).
- Missing neighbours from round 1 (FaithScore, Woodpecker, ViGoR, GROUNDHOG, CuRe): VERIFIED and now cited (V-145 to V-147).
- Decoding-time baselines PND and Over-alignment, rubric rewards 2608.12337: VERIFIED (Over-alignment also has a fine-tuning variant, PARTIALLY as a pure decoding baseline; V-196).
A4 dependencies: (b) is the object of study; (a) via POPE saturation for benchmark choice.
REFUTED items still present: none (the "never been isolated" sentence was replaced by the Geigle premise). UNVERIFIABLE: DenseWorld-1M licence (V-143); compute. PARTIALLY: Molmo2 lineage; RLHF-V still labelled [LIKELY] although VERIFIED in V-071.

## IDEA-21: precision-bounds factorial (REVISED)

Premises and verdicts:
- Five bottleneck papers (Hi-Token, LocateAnything, PaDT, VPSG, RULER / I-MRoPE, PGT, UHR-Micro): VERIFIED at abstract level (V-157).
- MolmoPoint "too coarse-grained" (verbatim, 28x28-px pooled patches; about 4.7-px precision claimed): VERIFIED (V-175).
- ExpVG as the only design-space study, and "format choices alone move RefCOCO by several points": VERIFIED for direction, PARTIALLY for magnitude (per-factor deltas only in figures; V-158, V-192). This is the premise that sets grade B.
- Qwen changed convention "with a rationale but no ablation": VERIFIED (V-169).
- Effective token pitch: Qwen3-VL 32 px, Qwen2.5-VL 28 px, InternVL3.5 28 px per 448-px tile, PaliGemma 14 px unmerged: VERIFIED from HF configs and the PaliGemma paper (V-166 to V-168, V-172).
- ScreenSpot-Pro 0.07% targets and 1,581 items; Ref-L4 45,341 items at 30 to 3,767 px: VERIFIED (V-173, V-174).
- SA-Co usable as a non-COCO box benchmark: VERIFIED (HF card ships boxes; images from MetaCLIP and SA-1B; V-170).
- Three closest papers (ExpVG, LocateAnything, Hi-Token): VERIFIED; deltas hold. No matched-data factorial found (V-179). Missing precedents: Shikra §6.2 numerals-vs-bins (V-177) and 2402.07384 small-object factors (V-178).
A4 dependencies: (a) VERIFIED; (c) PARTIALLY, not load-bearing (all arms SFT); (d) PARTIALLY, secondary probe only.
REFUTED items present: none. PARTIALLY items present: Objects365 "non-commercial" (academic-use terms, V-171); SA-Co licence (card says "other"); the float-level drop justification. UNVERIFIABLE: throughput and alignment costs, edge-error model, all predicted effect sizes, Hi-Token / LocateAnything / PaDT details beyond abstracts, Florence-2 and Rex-Omni pitches.

## IDEA-22: grounding accuracy versus task success in computer-use agents (REVISED)

Premises and verdicts:
- Jedi ladder 5.0 to 24.0 to 27.0 (GPT-4o planner) and 51.0 (o3), screenshots only: VERIFIED (V-084, V-161).
- "The one verified data point is Jedi ... three grounders and no control" and the Skeptic's framing that no paper shows the grounding-to-success link in a controlled way: **REFUTED**. Agent S2 Figure 6 swaps four grounders under a fixed planner on OSWorld (V-184); MMBench-GUI Finding 2 fixes a GPT-4o planner and reports 2.8x (Delta=17.25) from grounder improvement versus 1.15x (Delta=3.58) from planner improvement, with the caption "Task success grows roughly linearly with visual-grounding accuracy" (V-185); GroundCUA Table 4 compares grounders under a fixed o3 planner on OSWorld-Verified (V-186).
- Error injection as a first: no parametric grounder-error injection in a closed loop was found, but GUI-RobustEval (2605.29447) replays real root-cause errors (11 types incl. wrong UI element) into live OSWorld and measures post-error success (V-187); OSWorld-Noisy perturbs the environment rather than the grounder (V-188); GUI-Perturbed is offline (V-189). The delta survives but is not yet stated against these (V-190).
- Environment facts: OSWorld 369 tasks, three OS, Apache-2.0, providers; OSWorld-Verified in-place upgrade; AndroidWorld 116 tasks / 20 apps: VERIFIED (V-162, V-180 to V-182).
- Benchmark and grounder facts (ScreenSpot-Pro, OSWorld-G, the ladder numbers): VERIFIED (A1, V-104, V-122, V-085, V-173). GUI-G1 opposite box-size directions: PARTIALLY (V-061). Point-It-Out "where it is vs where to act known to differ": PARTIALLY (V-191). Aguvis stage-wise attribution: PARTIALLY (V-183).
A4 dependencies: none load-bearing; (c) motivates the reward question only.
REFUTED item still present: the §1 / §4 / §8 framing sentences above. UNVERIFIABLE: episode cost, discordant rate, grounding-sensitive subset size, all expected effect sizes; OSWorld-Verified task count (PARTIALLY: not restated on the blog).

## IDEA-31 (ABANDONED)

Abandoned by the author before review. Facts the abandonment rests on: COCO-Search18 is 3,101 + 3,101 COCO images with target-absent fixations not yet released and non-commercial terms (VERIFIED, V-133). Nothing further to grade.

## Cross-cutting items every file must still respect

- Qwen2.5-VL does not document RefCOCO; Qwen3-VL and InternVL2.5/3 do; Molmo and PaliGemma pretraining are RefCOCO-clean; Molmo2 has no RefCOCO annotations but COCO images (A2, V-112).
- ScreenSpot-Pro top: 73.5 in a tech report (zoom), 81 to 83 on the community leaderboard (zoom, self-submitted), 62 single-pass flagship (A1).
- GroundingME "0% rejection" is a non-thinking-mode, literal-null-box result (V-111).
- Ref-L4 images: cleaned RefCOCO/+/g val/test (COCO) plus the Objects365 test set (V-134).
