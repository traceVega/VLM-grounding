# Prior work: line-level verdict rewards, free labels from positives, on-policy correction, vote-before-veto (2026-10-02)

Web survey: arXiv abstracts and HTML, plus PDF text for MAttNet, ALFWorld and Speaker-Listener-Reinforcer. Numbers are quoted from the paper text. Anything marked "unverified" was seen only on a third-party page or not checked against the paper.

**Setting (one paragraph).** Qwen3-VL-4B, trained with LoRA SFT and then GRPO, grounds referring expressions or rejects false-premise ones. The model writes the expression's atomic conditions and proposes up to 4 candidate boxes. Each candidate is audited in an isolated context, one line per claim: `claim | seen: ... | match / mismatch / unsure`. A rule returns the first candidate without a named mismatch, else "no object". On correct objects, 2-13 % of audit lines say mismatch (RefCOCO 4 %, RefCOCOg 9 %, PR-Bench relation 13 %), so false vetoes grow with the number of claims. Planned remedies:
- (1) a per-line three-valued verifiable reward (+1 / -1 / 0 unsure) from data-engine labels;
- (2) free line labels from positive grounding data;
- (3) on-policy correction of wrong verdict lines by the label engine, with CE on the corrected tokens only;
- (4) a majority vote over sampled audits before vetoing.

## A. Dense / claim-level rewards for VLM hallucination and verification

| Work | What is rewarded and how | Label source | Difference from ours |
|---|---|---|---|
| Fine-Grained RLHF, Wu et al., NeurIPS 2023, arXiv 2306.01693 | reward after every segment (sentence) from a separate RM per error type (factual, irrelevant, incomplete); PPO | learned RMs on human span labels | text only; scores generated content, not verifier verdicts |
| M-HalDetect / FDPO, Gunjal et al., AAAI 2024, arXiv 2308.06394 | 16k sub-sentence labels on VQA answers; fine-grained DPO; RM for best-of-n | human -> learned RM | preference and rerank, no RL on verdicts |
| RLHF-V, Yu et al., CVPR 2024, arXiv 2312.00849 | humans correct hallucinated segments of the model's own answers; dense DPO on the corrected segments | human | generator side (see also D) |
| FGAIF, Jing & Du 2024, arXiv 2404.05046 (venue unverified) | sub-sentence rewards for object existence / attribute / relation from three RMs inside PPO | AI labels -> learned RMs | closest pre-2025 per-type dense RL reward for VLMs, but learned and applied to descriptions |
| HSA-DPO, Xiao et al., AAAI 2025, arXiv 2404.14233 | sentence-level detector (object / attribute / relation) trained on labels from proprietary models; detect-then-rewrite pairs; severity-weighted DPO | learned detector | DPO; the detector sits outside the policy |
| ViGoR, Yan et al., ECCV 2024, arXiv 2402.06118 | sentence-level human ratings train an RM; object-detector rewards added | learned + detector | descriptions |
| RLAIF-V, Yu et al., arXiv 2405.17220 | an open MLLM splits responses into claims and checks each yes/no; score = -#rejected claims; used to rank responses for DPO | model judge | claim checks collapse into one response score |
| TLDR, Fu et al., ICLR 2025, arXiv 2410.04734 | token-level reward model | learned, on synthetic perturbation labels | an RM, not a policy-side verdict reward |
| SENTINEL, Peng et al. 2025, arXiv 2507.12455 | sentence-level preference pairs where hallucination first appears; context-aware DPO | two open-vocabulary detectors cross-checked | object existence; DPO |
| CHAIR-DPO, Compagnoni et al., BMVC 2025, arXiv 2508.20181 | CHAIR against COCO object annotations picks the winner and loser | annotations (verifiable), no RM | response-level preference |
| SC-Captioner, ICCV 2025, arXiv 2508.06125 | in a self-correction turn, each object / attribute / relation added or removed is rewarded against a scene-graph parse of the reference | references (verifiable) | captions; rewards edits, not verdicts |
| VisualPRM, Wang et al. 2025, arXiv 2503.10291 | 8B PRM; a step is labelled correct if any of 16 Monte Carlo continuations succeeds; used for Best-of-N | learned (MC labels) | test-time reranking only |
| Perceval, Min et al., CVPR 2026, arXiv 2604.24583 | PRM checks image claims one by one and lists erroneous spans; GRPO token advantage = A - alpha*m*abs(A), alpha = 0.1, flagged spans only; test-time truncate-and-regenerate | learned PRM; labels from Gemini-2.5-Pro on Qwen2.5-VL rollouts | penalty only, on generator claims (see Positioning) |
| V-Rubrics, Tian et al. 2026, arXiv 2608.25580 | Gemini-3-Pro scores reference propositions (faithfulness, consistency, instruction following); prefix-localized rubric credit in GRPO | judge | judge-scored |
| ClaimDiff-RL, Li et al. 2026, arXiv 2605.20278 | a multimodal judge verifies each claim difference against the reference caption; separate hallucination and omission terms in GRPO | judge | captions |
| Perception-R1, Xiao et al. 2025, arXiv 2506.07218 | an LLM judge checks responses against visual annotations extracted from correct CoTs | judge | response-level |

**Text-side precedents for rewarding verdicts with verifiable labels**
- "Trust, But Verify" (RISE, arXiv 2505.13445) rewards the model's self-verification verdict with the rule-based outcome verifier's label in the same RL run (response-level, text reasoning).
- ReVeal (arXiv 2506.11442) gives turn-level rewards both to code generation and to the model's self-written tests.

**Three-valued schemes**
- TruthRL (ICML 2026, arXiv 2509.25760) uses +1 correct / 0 abstain / -1 wrong at the answer level.
- PRM800K rates each step -1/0/+1 ([README](https://github.com/openai/prm800k), arXiv 2305.20050), but the ratings are used to train a learned PRM.

**Verifiable vs learned.** Only SENTINEL (detectors), CHAIR-DPO and SC-Captioner (annotations or references) avoid a learned RM or judge, and all three score the generator's content.

**Closest to (1):** Perceval for dense claim-level credit inside GRPO, plus RISE for verdicts rewarded by verifiable labels.

**Not found:** per-claim rewards on a VLM's own verdict lines from label-based supervision, or a 0 reward for "unsure" at line level.

## B. Grounding / REC / visual-search RL

| Work | Reward granularity | Notes |
|---|---|---|
| Rex-Thinker, arXiv 2506.04034 (ICLR 2026 per third-party notes, unverified) | final answer only: 0.9 * F1 over boxes (exact match to the given candidates) + 0.1 * format | CoT = plan -> check each candidate -> summary, abstain if none; 90.8k GPT-4o SFT traces kept only when the final answer matches GT |
| VLM-R1 (arXiv 2504.07615), Visual-RFT (arXiv 2503.01785), UniVG-R1 (arXiv 2505.14231) | final box IoU and format | outcome only |
| Perception-R1, Yu et al., arXiv 2504.07954 | IoU for grounding, point distance for counting, Hungarian-matched F1/IoU for detection | outcome only |
| DeepEyes, arXiv 2505.14362 | accuracy + format + a tool bonus paid only when the answer is correct | the paper says intermediate visual actions lack step-level supervision |
| Ground-R1, arXiv 2505.20272 | answer + format; evidence regions are not annotated | scale-relative policy optimization |
| TreeVGR, ICLR 2026, arXiv 2507.07999 | adds a dual IoU reward (precision + recall) of evidence boxes against annotated boxes | trajectory level |
| SATORI-R1, arXiv 2505.19094 | one reward per stage: caption BLEU/ROUGE, box IoU, answer | stage-level, verifiable |
| ToolsRL, CVPR 2026 Findings, arXiv 2604.19945 | per-state tool rewards (zoom box ModF1 against annotated boxes), followed by an accuracy-reward stage | clearest step-level verifiable reward in visual search |
| PointRL, arXiv 2608.25299 | annotations kept as hidden verifier evidence; deterministic checks of validity, coverage, cardinality, redundancy | checks outputs, not reasoning |
| DRAgent, arXiv 2608.22885 | none (SFT only) | per-candidate yes/no verification; chains filtered by GT; no no-target case |
| RC-GRPO, arXiv 2608.04698 | answer-level +1/0/-1, forced-rejection rollouts, stage-II reason reward | rejection, but no candidates or claims |

**Answer:** no paper found rewards per-claim verification lines.
- Per-candidate verification appears only as SFT supervision (Rex-Thinker, DRAgent).
- The only claim-level credit in visual-search RL is Perceval (learned, penalty only).

**Closest:** Rex-Thinker has the same check-each-candidate-then-abstain structure, but its reward is outcome only. ToolsRL is the closest for per-step verifiable rewards from box annotations.

## C. Positive grounding annotations as free verification labels

- **MAttNet**, Yu et al., CVPR 2018, arXiv 1801.08186. A template parser extracts colour and attribute words from each expression, and these become multi-label BCE targets for the referred object's attribute branch. This is the earliest "the expression's attributes hold on the annotated box" supervision. Unmentioned attributes count as negatives (noisy), and there is no verifier.
- **Speaker-Listener-Reinforcer**, Yu et al., CVPR 2017, arXiv 1612.09542. A discriminative classifier over expression-object pairs serves as the speaker's reward (expression-level).
- **C-REC**, Yu & Li, CVPR 2024 ([CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Revisiting_Counterfactual_Problems_in_Referring_Expression_Comprehension_CVPR_2024_paper.html)). Fine-grained attribute edits build counterfactual RefCOCO/+/g; a counterfactual-label head is trained alongside box prediction (expression-level).
- **Ferret**, ICLR 2024, arXiv 2310.07704. 95k annotation-derived hard negatives (localize categories that are absent).
- **GUI-Actor verifier**, arXiv 2506.03143. A yes/no verifier on one marked candidate, trained on OS-Atlas with annotated targets as positives and other boxes or points as negatives. At inference it returns the first candidate above a threshold, the same shape as our first-candidate rule.
- **REVERSE**, NeurIPS 2025, arXiv 2504.13169. Ground-truth answer phrases from LLaVA-665k are tagged confident; wrong answers made by rules or GPT-4o-mini are tagged unconfident (1.3M samples). The VLM flags its own phrases and backtracks at inference.
- **Whitehead et al.** (Scale AI), arXiv 2409.00238. Grounded spans from a COCO grounded-chat set are kept as non-hallucinated and T5 replacements as hallucinated. Pre-training span detectors on this helps at low data (LLaVA-1.6-13B, 500 samples: 25.3 vs 18.0 F1).
- **Caveat, LRV-Instruction**, ICLR 2024, arXiv 2306.14565: "a balanced ratio of positive and negative instances ... leads to a more robust model." Positive-only labels push toward yes-bias.

**Answer:** treating positive annotations as faithful labels is standard practice for hallucination detectors.

**Not found:** claims derived from the expression, checked on the annotated box, and used as labels against a verifier's false accusations.

**Closest:** MAttNet for the concept, REVERSE for the modern phrase-level form. Both label the generator's own phrases, not a separate verdict line.

## D. On-policy correction with programmatic or privileged teachers

- **DAgger**, Ross et al., AISTATS 2011, arXiv 1011.0686, and the **dynamic oracle**, Goldberg & Nivre, COLING 2012 ([C12-1059](https://aclanthology.org/C12-1059/)). A programmatic oracle labels the optimal action in states the learner itself reaches, including wrong states.
- **OCD**, Sabour et al., ICLR 2019, arXiv 1810.01398. It samples from the student, and an edit-distance DP gives the optimal next tokens toward the ground truth for every prefix; training is CE on those targets. **This is the closest precedent for (3):** a programmatic teacher plus CE on the student's own prefixes.
- **ALFWorld**, ICLR 2021, arXiv 2010.03768. The text agent is trained with DAgger "assisted by a rule-based expert".
- **GKD**, ICLR 2024, arXiv 2306.13649, and **SKD**, ICLR 2025, arXiv 2410.11325 (the teacher replaces student tokens outside its top-K). Both use LLM teachers.
- **OPSD**, Zhao et al., ICML 2026, arXiv 2601.18734. The teacher is the same model conditioned on a verified solution, with per-token divergence on student rollouts. This is the privileged-information analog of a label engine.
- **OEC** (arXiv 2512.14895), **DAgger revisited** (arXiv 2605.12913) and **SCoRe**, Lyu et al., ICML 2026, arXiv 2509.14257. In SCoRe an LLM teacher corrects only the earliest error, followed by SFT and then short-horizon RL from the verified prefix.
- **VLM hallucination**:
  - OPA-DPO, CVPR 2025, arXiv 2501.09695: GPT-4V minimally revises the policy's own responses, then LoRA-SFT on GT and revised responses, then DPO ("on-policy data hold the key").
  - RLHF-V does the same with human corrections and dense DPO.

**Answer:** on-policy correction by a programmatic teacher is classical (DAgger, dynamic oracles, OCD). Recent LLM/VLM versions use model or human teachers and train with KL, DPO or full-response SFT.

**Not found:** a programmatic label engine correcting a VLM's verdict lines with loss on the corrected tokens only. That would be an application of OCD/DAgger, not a new method.

## E. Voting over verification decisions

- **Self-consistency** (ICLR 2023, arXiv 2203.11171) votes over answers. **GenRM** (ICLR 2025, arXiv 2408.15240): a majority vote over CoT verification rationales improves verification.
- **Heimdall**, arXiv 2504.10337. Majority vote over 64 verifications raises AIME2024 verification accuracy from 94.5 to 97.5 %. Both error types (missed errors and rejected correct solutions) fall as votes increase (Fig. 3; no numbers given in the text).
- **Sample, Scrutinize and Scale**, arXiv 2502.01839:
  - It averages k_verif = 50 verification scores per response.
  - Labelling correct responses as wrong "generally" hurts downstream results more than the reverse error.
  - **Split-Context** verification (separate threads for pieces of one response) raises the rate of correct responses labelled wrong from 14 to 19 % (MATH) and from 7 to 11 % (AIME), with Gemini 1.5 Pro (Table 4); the authors attribute this to miscalibration.
  - This is the text analog of our isolated per-claim audits.
- **Stechly et al.**, arXiv 2402.08115. A GPT-4 self-verifier rejects 95.8 % of valid graph colourings, 20.7 % of valid Game-of-24 answers and 15.5 % of valid Blocksworld plans. Self-critique loops degrade results; sound external verifiers fix this.
- **Huang et al.**, ICLR 2024, arXiv 2310.01798: intrinsic self-correction can hurt.
- **Chen et al.**, arXiv 2403.02419. Vote accuracy can first rise and then fall as calls increase: voting helps easy items and hurts hard ones. A vote removes unstable errors, not systematic ones.
- **Aha Moment Revisited**, NeurIPS 2025 MAR workshop, arXiv 2506.17417. VLM self-verification is weak and makes poor use of visual information; a majority vote over answers beats best-of-N with self-verification.
- **Program-Verified Self-Evolution**, arXiv 2609.33855. Human raters judge 24 % of majority-vote labels wrong, against 6 % for labels computed by fixed programs over structured image records.
- **Trust or Escalate**, ICLR 2025, arXiv 2407.18370: sampled judgments estimate judge confidence for abstaining or escalating.
- **SelfCheckGPT**, EMNLP 2023, arXiv 2303.08896: flags sentence-level hallucinations from consistency across samples.
- **VerifySteer**, arXiv 2605.20745: over-critical step verifiers reject correct reasoning; self-consistency is its compute-heavy baseline.
- **Grounding:** True/False verification for REC, arXiv 2509.09958. Each box is checked in isolation. If every box is False, the method falls back to choosing among all boxes and abstains only if that choice is "none", which is a structural guard against false vetoes.

**Answer:** voting over sampled verifications is standard. In the one paper found with an error breakdown (Heimdall), false rejections also fall. Votes from one model are correlated (Chen), so systematic accusations will survive a vote.

**Closest:** Heimdall and GenRM for voting over verifications; arXiv 2509.09958 for handling vetoes in REC.

## Positioning
- **(4) is standard practice** (GenRM 2408.15240, Heimdall 2504.10337, Sample-Scrutinize-Scale 2502.01839). Report it as an inference baseline, not a contribution. Show the vote curve on positives; expect it to remove unstable false vetoes only (2403.02419).
- **Our claim-count effect has a text precedent.** Verifying pieces of a response in separate contexts lowers precision on correct answers (Split-Context, 2502.01839). Our per-line rates in visual grounding (4 / 9 / 13 %) and their growth with claim count are new evidence, not a new phenomenon; cite it.
- **(1) is built from standard parts:** segment-level dense credit (2306.01693, 2404.05046, 2604.24583), ternary rewards (2509.25760; PRM800K -1/0/+1), and verdicts rewarded by verifiable labels (RISE 2505.13445). Not found: label-based (not learned) per-claim rewards on the policy's own verification lines in a VLM, with 0 for unsure. Frame the novelty as this combination.
- **(2) builds on standard practice but looks new in its use.** Positive annotations as faithful labels is common (MAttNet 1801.08186, REVERSE 2504.13169, 2409.00238). Using claims derived from the expression, checked on the annotated box, as free *false-accusation* labels for a verifier looks new. The signal is one-sided, so pair it with data-engine negatives (LRV 2306.14565) to avoid drift toward "match".
- **(3) is OCD/DAgger with a programmatic oracle** (1810.01398, 1011.0686). Its VLM analog with a model teacher is OPA-DPO (2501.09695). Present it as a known recipe, not an invention.
- **No grounding RL paper found rewards per-claim verification lines.** Rex-Thinker (2506.04034) has the same per-candidate structure with outcome-only GRPO. ToolsRL (2604.19945) has per-step verifiable rewards from boxes.
- **Single closest paper to the combination: Perceval** (CVPR 2026, arXiv 2604.24583). It checks image claims one by one, turns flagged spans into token-level GRPO credit on visual-search data, and reuses the checker at test time. It differs from ours on six points:
  - its PRM is learned, with Gemini labels;
  - it only penalizes, and only the generator's claims rather than the policy's verdicts;
  - it has no "unsure" verdict;
  - it uses no free labels from positives;
  - it has no on-policy CE correction;
  - it has no rejection task.
