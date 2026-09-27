# Extra papers: grounding / visual-reasoning RL and process-reward papers

Scope: main ideas only, fixed template, at most 180 words per paper. Sources are arXiv abstract pages and HTML full texts fetched 2026-09-23; numbers are the papers' own. Two names in the request (VALOR, ViGrit) do not resolve to any arXiv paper; see entries 3 and 4.

## 1. CVSearch

- Paper: arXiv 2605.23655, 2026; plug-in on 2B–32B MLLMs from four families.
- Idea: a training-free "assess-then-search" scheduler that gives an MLLM cognitive visual search over high-resolution images and matches RL-trained searchers.
- SFT/RL: none; it is the training-free baseline the RL papers compete with. Stages: (1) global assessment, the MLLM scores information sufficiency c_q and answers directly if above threshold; (2) expert search, SAM3 proposals for the queried object; (3) on failure, scene-aware scanning: Semantic-Guided Adaptive Patching (feature clusters, not a rigid grid) plus bottom-up search from high-entropy leaves ranked by a complexity-and-evidence priority score. The threshold decays 0.9 to 0.5 over iterations.
- Reward: none; c_q, c_v and the priority score act as inference-time verifiers.
- Failure/fix: proposals miss tiny or occluded objects, so fall back to scanning; grid tiling fragments objects, so SGAP; top-down errors, so bottom-up traversal.
- Results: V* 75.4 to 91.6 (LLaVA-OV-7B); HR-Bench-8K 57.4 to 77.6 (InternVL2.5-8B); beats DeepEyes 90.1/75.1/72.6 with 91.6/75.6/74.8 on V*/HR4K/HR8K.
- Insight: a sufficiency check plus a fallback ladder recovers most of what search-RL buys; RL must beat this baseline.

## 2. PixelEyes

- Paper: arXiv 2607.00115, 2026; Qwen3-VL 4B and 8B.
- Idea: decouple reasoning from perception; the reasoner says what to look for (coarse box + referring expression), a mask tool (SAMTok) says where it is and returns a tight crop.
- SFT: cold start on PixelEyes-6K, 5.8K successful trajectories re-synthesised by Gemini-3-Flash driving the mask tool on about 7K Mini-o3 image–question pairs; failures dropped. RL: vanilla GRPO (no KL, no entropy), max 6 turns, 5.5K prompts filtered by the SFT model to solve rate in (0, 0.9]. RL over SFT: VisualProbe-Hard 50.94 to 54.72, Pinpoint-Bench 52.66 to 54.73; RL sharpens rather than transforms.
- Reward: answer correctness under GRPO; format and tool terms are not itemised. Mini-o3's over-turn masking is dropped to favour short evidence paths.
- Anti-collapse: semantic-region BFS with global coordinate anchoring against redundant zoom loops; fallback to plain box crops for charts and maps where masks are ill-defined.
- Results: Pinpoint-Bench 54.7/55.2 (4B/8B) vs Mini-o3 44.3; V* 91.6/94.2; plugged into Gemini-3-Flash untuned, +26.1 on Pinpoint-Bench.
- Insight: the gain lives in the perception-tool interface, not in RL.

## 3. VALOR (unresolved, skipped)

An arXiv title search for "VALOR" returns only VALOR (2304.08345, vision-audio-language omni-perception pretraining, 2023) and VALOR-EVAL (2404.13874, LVLM coverage/faithfulness evaluation over objects, attributes and relations, 2024). Neither is a grounding-RL or reward paper, and no 2025–2026 "VALOR" grounding paper exists on arXiv or Semantic Scholar under that name. Skipped per instruction 11. If the intended paper was VALOR-EVAL, its one transferable point is that hallucination must be scored jointly on coverage and faithfulness, because models trade one for the other.

## 4. ViGrit (not found; two closest candidates)

No paper named ViGrit, ViGRiT or ViGRIT exists on arXiv or Semantic Scholar. The name overlaps GRIT and ViGoRL, both grounded-reasoning RL papers from May 2025; both are summarised.

### 4a. GRIT

- Paper: arXiv 2505.15879, 2025; Qwen2.5-VL-3B, InternVL3-2B.
- Idea: reasoning chains that interleave text with bounding-box coordinates, learned by RL from only 20 examples.
- SFT: none. RL: GRPO-GR on 20 triplets (10 VSR + 10 TallyQA).
- Reward: format +0.5 for think tags, +0.5 if at least one regex-valid box; counting +0.5 if the number of boxes matches the ground-truth object count; answer = GPT-4o binary + 0.1 x BLEU. No IoU term; boxes are never scored against ground truth.
- Anti-hacking: the box reward is purely syntactic (regex, count), so it cannot be gamed through semantics it does not measure; grounding fidelity emerges from the answer reward.
- Results: VSR 72.9 acc / 0.33 IoU, TallyQA 47.8 / 0.45, GQA 62.8 / 0.49; beats CoT, few-shot FT and ICL baselines.
- Insight: a syntactic "you must emit boxes" reward plus answer correctness is enough for grounding to emerge; dense IoU supervision is optional.

### 4b. ViGoRL

- Paper: arXiv 2505.23678, 2025; Qwen2.5-VL 3B and 7B.
- Idea: every reasoning step anchored to an (x,y) coordinate; the multi-turn variant zooms into the coordinate.
- SFT: about 30K traces from MCTS driven by frozen Qwen2.5-VL-72B over 1.5K prompts, linearised into direct and self-corrected chains. RL: GRPO, plus multi-turn with a crop tool. Removing GRPO costs 4.1 on SAT-2 and 3.5 on BLINK.
- Reward: correctness; format +1 for valid tags and coordinate references; multi-turn adds a grammar reward and a diversity bonus (+0.2 per sufficiently distinct coordinate, up to 4).
- Anti-collapse: without grounding and format constraints "RL collapses onto shortcuts"; the diversity bonus stops repeated zooms on one spot.
- Results: V* 86.4 (multi-turn 7B), SAT-2 62.9 (3B, +12.9 over vanilla GRPO), ScreenSpot-Pro 32.3; 72.8% of predicted coordinates judged correct by humans.
- Insight: grounding is a scaffold; region exploration rises 3.5x and subgoal setting 15x versus the baseline.

## 5. ViGoR

- Paper: arXiv 2402.06118, 2024; LLaVA with Vicuna-7B.
- Idea: fine-grained sentence-level rewards from humans and from a detector, used through rejection sampling plus SFT to cut hallucination.
- SFT: no separate cold start; the base LVLM samples candidate descriptions, the combined reward picks the best, a refinement step deletes sentences containing undetected nouns, then standard autoregressive SFT on the survivors. No policy-gradient RL.
- Reward: (a) a human reward model trained on 15,440 COCO image–text pairs with per-sentence error labels (object hallucination, attribute, relation), dense rather than holistic; (b) an automated reward: nouns extracted with NLTK, existence checked with GroundingDINO, plus or minus per noun. Both are variance-normalised and summed.
- Anti-hacking: fine-grained beats holistic human reward, MME 1309 vs 1027, because holistic scoring rewards fluent hallucination. Stated limitation: the detector cannot judge stuff regions, attributes or layout.
- Results: POPE F1 67.8 to 83.8; MME 960 to 1309; MMHal 1.3 to 1.6; beats LLaVA-RLHF.
- Insight: a claim-level detector check plus sentence deletion is a cheap verifier that transfers directly to any describe-then-ground pipeline.

## 6. Thinking with Visual Grounding

- Paper: arXiv 2606.16122, 2026; Gemma3-4B-IT.
- Idea: interleave thoughts with obj tags holding point or box groundings of the evidence used, and reward those tags directly.
- SFT: 19,909 traces, 107,613 groundings. Qwen3-VL-Plus writes traces, kept if correct; a SAM3 agent masks each referenced object; masks yield boxes and interior points. RL: GRPO on top.
- Reward: answer (w 1.0); grounding (w 0.5): box mode = IoU of unions of generated vs GT boxes, point mode = F1 of one-to-one point-in-mask matching; format 0.1 (think) + 0.1 (tags); truncation −1. Answer and grounding rewards are batch-normalised separately before summing. Unmatched extra objects are not penalised.
- Anti-collapse: non-grounded thinking suffers length collapse under RL and ends below the base model (TallyBench 33.3 to 21.7); grounding tags plus format reward stabilise it; a hard cap on tag count stops over-emission; a router matches generated tags to GT objects before scoring.
- Results: TallyBench 39.3 (point) vs base 33.3; SpatialMQA 25.4 to 38.7 (box), matching Gemma3-27B (39.0).
- Insight: the grounding reward's main value is stabilising RL, not the +1–2 points.

## 7. VL-PRM

- Paper: arXiv 2509.23250, 2025; Qwen-VL-PRM 3B and 7B on Qwen2.5-VL.
- Idea: a systematic study of building vision-language PRMs for test-time scaling, with perception errors labelled separately from reasoning errors.
- Data: hybrid; MCTS rollouts produce candidate steps, an o4-mini judge labels each step in context; the policy is forced into structured traces (perception steps describing the image first, then reasoning), so perception errors (86% of all errors) are localised. Loss: cross-entropy on binary step labels. No policy RL.
- Use: (1) greedy step selection by PRM score, (2) one-shot scoring of N complete solutions (PRM used as an ORM), (3) step-score averaging.
- Findings: PRM-as-ORM beats step selection by 2–3 points; the 3B PRM, trained only on elementary math, gains +16 macro-F1 on VisualProcessBench and matches GPT-4o; perception-level supervision gives consistent gains; larger policies (Gemma3-27B) gain more from test-time scaling.
- Noise: MC score and judge agree on only about 20% of incorrect steps; judge-labelled data clearly beats MC-only labels.
- Insight: label the perception step explicitly and score whole trajectories; cheaper and better than per-step search.

## 8. MM-PRM

- Paper: arXiv 2505.13427, 2025; InternVL2.5-8B for both policy (MM-Policy) and PRM.
- Idea: a fully automatic multimodal PRM whose step values come from MCTS, trained with soft labels.
- Data: MM-K12, 10K K-12 math problems with verifiable answers; OmegaPRM-style MCTS, up to 1,000 rollouts per problem, MC(step) = P(correct final answer | prefix), about 747K step labels. No policy RL.
- Loss: BCE with soft targets (the MC score itself); lr 4e-6, one-tenth of the SFT rate, needed for stability.
- Use: BoN-16 with per-step scores aggregated; MeanOdds is the best aggregator.
- Ablation: hard labels (MC > 0) 34–37 vs soft 42–43 on MM-K12; N from 2 to 16 lifts 38.6 to 42.8.
- Results: MM-K12 33.9 to 42.8, OlympiadBench 15.4 to 24.0, MathVista 62.9 to 67.6; MathVerse on InternVL-78B 50.2 to 54.5.
- Failure noted: hard thresholds discard uncertainty; an aggressive lr erases base knowledge once the model becomes a discriminator.
- Insight: soft MC values plus a conservative lr make an auto-labelled PRM work; thresholding is the easy mistake.

## 9. VisualPRM

- Paper: arXiv 2503.10291, 2025; 8B PRM on InternVL2.5-8B.
- Idea: the first open multimodal PRM plus a human-labelled step benchmark, used for Best-of-N.
- Data: VisualPRM400K; for each step, 16 Monte Carlo continuations give an expected accuracy mc_i, and the step is labelled positive if mc_i > 0, else negative. Trained as multi-turn chat, one step per turn, binary classification. No policy RL.
- Use: BoN-8, averaging step scores (beats min and max).
- Results: InternVL2.5-8B +8.4 over seven benchmarks; 78B +5.9; MiniCPM-V2.6 +8.0, Qwen2.5-VL-7B +3.7; PRM beats ORM by about 1.5 and self-consistency by about 2.4.
- Benchmark: VisualProcessBench, 2,866 samples with 26,950 human step labels.
- Failure noted: open-source MLLMs used as critics "tend to provide positive analysis and label most steps as correct"; model-as-judge is positively biased, MC labels are not.
- Insight: MC labelling with a > 0 threshold is crude but unbiased, which is why it beats prompting a model to judge.

## 10. Perceval

- Paper: arXiv 2604.24583, 2026, CVPR; Qwen2.5-VL 3B and 7B.
- Idea: a perception-centric PRM extracts image-related claims from a response, checks each against the image, and returns the erroneous spans, which drive token-level penalties inside GRPO.
- Data: perception-heavy queries, open VLM rollouts with natural hallucinations; Gemini-2.5-Pro marks hallucinated spans and the visual counter-evidence. PRM SFT'd on this; the policy has no cold start.
- RL: GRPO with per-token advantage A' = A − α·m·|A| (m = 1 on hallucinated tokens): correct tokens keep the sequence advantage, hallucinated ones get A(1 − α) if A > 0, A(1 + α) if A < 0.
- Gains over GRPO (3B): V* 80.1 to 83.3; about +3 on math/chart and general tasks.
- Anti-hacking: the penalty enters at the advantage, not as a scalar reward, so the PRM is harder to game; α = 0.3 over-penalises benign tokens, α = 0.1 is best.
- Inference: truncate the erroneous span and regenerate; k = 16 gives V* 94.8 vs majority vote 92.2.
- Insight: claim-level verification of perception is the highest-leverage place for a PRM.

## Closing: failure modes and standard fixes across these papers

Length collapse of non-grounded chains under RL (Thinking with Visual Grounding): fixed by interleaved grounding tags, a grounding-format reward and separate normalisation of answer and grounding rewards. Shortcut collapse, where the policy stops grounding once it is unrewarded (ViGoRL): format rewards on coordinates plus a diversity bonus for distinct locations. Tag or tool over-emission and redundant zoom loops (TwVG, PixelEyes, ViGoRL): hard caps on tag count, BFS with global anchoring, dropping over-turn masking, filtering prompts to solve rates in (0, 0.9]. Gaming a grounding reward with hallucinated coordinates: score only router-matched objects with IoU-of-unions (TwVG), or keep the box reward purely syntactic and let the answer reward carry semantics (GRIT). PRM over-penalisation (Perceval α = 0.3): small α applied at the advantage, never as a reward. PRM label noise: MC scores and judges agree on about 20% of wrong steps (VL-PRM), model critics label nearly everything correct (VisualPRM), hard thresholds lose nuance (MM-PRM); fixes are soft labels, an external strong judge, perception-first structured steps, a tenth-scale learning rate. Holistic reward rewarding fluent hallucination (ViGoR): sentence-level labels, a detector check, deletion of unsupported sentences. Step selection underperforming whole-trajectory scoring (VL-PRM): use the PRM as an ORM.
