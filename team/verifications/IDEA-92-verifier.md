# Verifier report: IDEA-92 (attributed description)

Verifier, 2026-09-02 (Phase 2, round 1). File checked: `ideas/IDEA-92-attributed-description.md` (status UNDER REVIEW, author Researcher1, NF-2 contribution by Researcher3). **P** = primary source read; **S** = secondary. Claim ids refer to `verifications/claims-log.md`.

## 0. Summary verdict

- **Motivating benchmarks and papers: VERIFIED** (DetailVerifyBench, Ground What You See, GAVEL, BICR; V-138 to V-141).
- **The sentence "the causal link 'better localization causes less hallucination' ... has never been isolated" is PARTIALLY REFUTED.** Geigle, Timofte, Glavaš 2024 (2406.14492), P (body): added two grounding objectives (referring expressions and grounded captioning) to LVLM training on three backbones (Vicuna-1.5-7B, Llama-3-8B, Phi-3-mini) and measured POPE, CHAIR/CHAIR-MEN and FaithScore; conclusion: "including grounding objectives ... to LVLM training has little to no effect on object hallucination, both in QA-based evaluation and open-ended captioning", and "generating grounded captions at inference slightly reduces object hallucinations but the effect is small and comes at the cost of (slight) reduction in caption detailedness". That is a controlled test of the training-objective form of the claim, with a null result. IDEA-92's arm design (verified attribution reward with necessity vs text-side correction) is a different, sharper test and remains open, but Section 1 must cite Geigle as the prior null and Section 3.5 must say what arm A adds over "grounding objectives" (verification and necessity, not mere box emission). Researcher3 has already added Geigle to r3-history; IDEA-92 has not.
- **Three named closest papers: VERIFIED; deltas hold**, except that Ground What You See's abstract does not say whether outputs carry per-claim regions, so "no explicit attribution in the output" is PARTIALLY supported (Section A).
- **Missing closer prior work (Section B):** the claim-decomposition-plus-verifier design is FaithScore (2311.01477) and Woodpecker (2310.16045); the fine-grained-reward-for-grounded-description design is ViGoR (2402.06118, 15,440 image-text pairs with fine-grained human and automated evaluations); grounded description with entity masks that "reduces object hallucination" is GROUNDHOG (2402.16846); structured claim-level rewards with grounded self-verification exist for video (CuRe 2607.05150; 2604.01460 with an "instance-aware scene-graph reward" and a "video-grounded VQA reward"). None of these tests necessity of the cited region, so the delta survives as "typed claim-region pairs verified for sufficiency *and* necessity, trained with that reward, and the controlled comparison against text-side correction". Novelty: **PARTIALLY** until these are cited and the delta is restated.
- **Section 9 [LIKELY] items: all resolved** (Section C); DenseWorld-1M licence UNVERIFIABLE from the abstract page.

## A. Novelty test (three closest papers)

| Paper | Says what the author says? | Delta holds? |
|---|---|---|
| GLaMM (2311.03356) | Yes, P: Grounded Conversation Generation, "natural language responses seamlessly intertwined with corresponding object segmentation masks"; GranD 7.5M concepts, 810M regions; evaluation protocol with curated grounded conversations (metric not named in abstract). | Yes: noun-phrase masks by SFT, overlap-style evaluation; no attribute/count/relation attribution, no verification reward. |
| Ground What You See (2601.06224) | Mostly, P: three causes (chained visual reasoning anchoring, low exploration diversity, NTK-similar sample conflicts); fix = planning and captioning stages before reasoning with a caption-quality reward, diversity prioritization, InfoNCE regularizer. The abstract does not state whether outputs contain per-claim regions. | Yes on the reward and output design; soften "no explicit attribution in the output" to "not described in the abstract". |
| GAVEL (2606.26923) | Yes, P: verify, explain and localize caption-image misalignment; "even strong closed-source models struggle"; a supervised baseline trained on a "human-annotated training split" improves grounding and explanation metrics; dataset size and GPT-5 are not named in the abstract. | Yes: GAVEL is the verifier side; IDEA-92 is the generator side. The lead's bar (beat SFT on GAVEL train) is well posed. |

## B. Closer or missing prior work

1. **FaithScore (2311.01477, EMNLP Findings 2024), P:** "reference-free and fine-grained evaluation metric" that identifies descriptive sub-sentences, extracts atomic facts and verifies them against the image. IDEA-92's decomposition-plus-judge is FaithScore's design; Geigle used FaithScore as one of the three hallucination measures. Cite it in Sections 3.2 and 5.
2. **Woodpecker (2310.16045), P:** training-free post-hoc correction with key-concept extraction, question formulation, visual-knowledge validation (detectors), visual-claim generation and correction; +30.66 POPE over MiniGPT-4. This is the verifier-ensemble ancestor; the delta is training-time reward and necessity, not post-hoc correction.
3. **ViGoR (2402.06118), P:** "fine-grained reward modeling" from "cheaper human evaluations" and "automated methods" to improve visual grounding of LVLM descriptions; released "15,440 images and generated text pairs with fine-grained evaluations". Whether rewards attach regions per claim is not in the abstract, but this is the closest "fine-grained reward for grounded description" precedent and must be in Section 5 with a one-sentence delta.
4. **GROUNDHOG (2402.16846, CVPR 2024), P:** grounded description with entity masks retrieved from holistic segmentation, M3G2 dataset; the abstract claims reduced object hallucination. Closer than GLaMM for "grounded description reduces hallucination".
5. **Structured claim rewards:** CuRe (2607.05150), P: "decomposes captions into category-aware atomic claims through a structured rubric" for video-caption RL (region scope not stated); Reinforcing Consistency in Video MLLMs with Structured Rewards (2604.01460), S (listing): "instance-aware scene-graph reward" and "video-grounded VQA reward for hierarchical self-verification". Cite both as the reward-design neighbours.
6. **FActScore (2305.14251), P:** "breaks a generation into a series of atomic facts and computes the percentage of atomic facts supported by a reliable knowledge source"; the text-side analogue, correctly labelled in the file.
7. **Geigle 2024 (2406.14492):** see Summary; it is the prior null result for Section 3.5 and must be cited.

## C. Section 9 claims and numbers

| Claim | Verdict | Evidence |
|---|---|---|
| DetailVerifyBench: 1,000 images, 5 domains, >200-word captions, token-level hallucination annotation, no spatial attribution | VERIFIED | P (V-138) |
| Ground What You See: RL can increase hallucination; three causes | VERIFIED | P (V-139) |
| GAVEL: closed models struggle; supervised baseline; training split | VERIFIED (GPT-5 not named in abstract) | P (V-140) |
| BICR: probe with ranking loss on real vs blacked-out hidden states; five LVLMs; 4-18x fewer parameters | VERIFIED | P (V-141) |
| GLaMM details | VERIFIED | P (V-142) |
| DenseWorld-1M: dense grounded captions, three-stage pipeline with two VLMs, 1M images | VERIFIED; licence UNVERIFIABLE | P (V-143) |
| Describe Anything: localized captioning, DLC-Bench, DLC-SDP pipeline | VERIFIED | P (V-144) |
| CapRL: reward = vision-free LLM answering MCQs from the caption; no regions | VERIFIED | P (V-145) |
| Claim-level rubric rewards (CuRe) | VERIFIED (region scope not stated) | P (V-146) |
| RLHF-V: 1.4k segment-level corrections, −34.8% | VERIFIED | V-071 |
| FActScore characterization | VERIFIED | P (V-147) |
| Molmo2 licence / RefCOCO-free | PARTIALLY | V-112 |
| Geigle 2024 as prior test of the causal claim | see Summary | P (V-148) |
| SAM 3 licence | VERIFIED custom | V-114 |
| Compute (250 GPU-h GRPO at 4B etc.) | UNVERIFIABLE | not checked |

## C2. Addendum (2026-09-02): the revised IDEA-92 file, and the Skeptic-cited precedents checked

The file was revised before round 1 (Section 0 reframing, Geigle 2024 as premise, KAWHI/COPO/ClaimDiff-RL novelty test, three-class attribution, arm D). Against the revised text:
- **Geigle is now the premise (Section 1): correct and consistent with V-148** (objectives = referring expressions + grounded captioning; backbones Vicuna-1.5-7B, Llama-3-8B, Phi-3-mini; POPE, CHAIR/CHAIR-MEN, FaithScore; "little to no effect ... both in QA-based evaluation and open-ended captioning"). The file's "grounding objectives added at SFT time" is an accurate paraphrase. One addition worth using: Geigle also found that *generating grounded captions at inference* "slightly reduces object hallucinations ... at the cost of (slight) reduction in caption detailedness", which is the closest existing data point for the file's inference-time typed-claim format and should be cited as such.
- **KAWHI (2603.27375), P (abstract):** "adaptively localizes semantically salient regions through hierarchical geometric aggregation, identifies vision-critical attention heads via structured attribution, and performs paragraph-level credit reallocation"; plug-and-play reward reweighting for GRPO/GSPO on reasoning benchmarks. The revised novelty-test description is accurate; note that KAWHI targets reasoning, not description, which strengthens the delta (V-154).
- **COPO (2508.04182), P (abstract):** "imposes token-level sufficiency and necessity constraints to measure each inference token's causal contribution" with a "causal completeness reward" inside GRPO; motivated by MLLMs attending to background regions. Description accurate (V-155).
- **ClaimDiff-RL (2605.20278), P (abstract):** "reference-conditioned atomic claim differences as the reward unit for caption RL", a multimodal judge "verifies each difference against the image", 160-image human-labelled diagnostic set; no region attribution in the abstract. Description accurate (V-156). The Skeptic's 92-M3 (arm B should be a region-free claim-verification GRPO arm, ClaimDiff-RL/CuRe style) is the right control and I endorse it.
- **VISE (2606.27373), P (body):** listed in Section 5 as "sufficiency and invariance notions for ... predicted regions". Precisely: geometric-invariance GIoU reward under affine/crop/flip, and a semantic-invariance reward that Gaussian-blurs (σ=25) the model's own predicted box and requires it to report the object as no longer visible; evaluated on captioning/VQA only (V-153). Fine as a contrast.
- **Still missing from Section 5 after the revision:** FaithScore (2311.01477), Woodpecker (2310.16045), ViGoR (2402.06118) and GROUNDHOG (2402.16846) (Section B above, V-147). ViGoR is the one that most needs a one-sentence delta ("fine-grained reward modeling" for grounding of LVLM descriptions, 15,440 fine-grained-evaluated pairs).
- **Rubric rewards for long-form hallucination RL (2608.12337)** and **unsupervised self-evolution (2603.21289)**: id checks pending in this round (see claims-log when resolved).

## D. Actions requested of the author

1. Cite Geigle et al. 2024 in Section 1 as the existing controlled null result and restate the causal experiment as "verified attribution with necessity vs text correction", explaining why it could succeed where grounding objectives did not.
2. Add FaithScore, Woodpecker, ViGoR, GROUNDHOG, CuRe and 2604.01460 to Section 5, each with a one-sentence delta; ViGoR is the one a reviewer will name.
3. Soften the Ground What You See delta to what its abstract supports.
4. Reuse IDEA-91's ancestors for the necessity edits (RISE deletion/insertion, CSS critical-object masking), since Section 3.2's necessity test is the same operator.

## C3. Addendum (2026-09-02): pending id checks from C2 resolved; Section D actions confirmed applied

Status of the file at check time: UNDER REVIEW (revised before round 1; a REVISED status has not yet been set). The three ids left pending in C2 and the two decoding-time baselines were resolved through the arXiv API (V-196).

| Claim | Verdict | Evidence |
|---|---|---|
| Rubric rewards for long-form hallucination RL (2608.12337) exist and are claim-level | VERIFIED | P: question-specific key-point rubrics (required and optional information); soft combination of grounding, rubric coverage and relevance rewards; in- and out-of-distribution tasks (V-196). |
| PND (2604.24396) as a decoding-time baseline | VERIFIED | P: training-free Positive-and-Negative Decoding, dual-path contrast with counterfactuals, up to +6.5% on POPE / MME / CHAIR (V-196). |
| Over-alignment (2605.08245) as a decoding-time baseline | PARTIALLY | P: the paper proposes both a training-free inference-time projection and a bias-aware fine-tuning variant; only the former is a decoding-time baseline (V-196). |
| RLHF-V (2312.00849) [LIKELY] | VERIFIED (upgrade the label) | V-071: 1.4k segment-level human corrections, -34.8% hallucination. |
| Unsupervised self-evolution (2603.21289) | VERIFIED | bounded-Judge GRPO for multimodal reasoning (V-194). |

Section D actions from the round-1 report: grep of the current file confirms Geigle 2024 is now the Section 1 premise and Section 5 cites ViGoR, GROUNDHOG, FaithScore and Woodpecker with deltas. Nothing REFUTED remains in the file as far as this check goes.
