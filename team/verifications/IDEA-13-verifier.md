# Verifier report: IDEA-13 (referring games at MLLM scale)

Verifier, 2026-09-01 (Phase 2, round 1). File checked: `ideas/IDEA-13-referring-games.md` (status UNDER REVIEW, author Researcher1). **P** = primary source read; **S** = secondary. Claim ids refer to `verifications/claims-log.md`.

## 0. Summary verdict

- **Motivating facts: VERIFIED** (GroundingME 45.1 and its Discriminative/Spatial dimensions, V-063/V-111; Ref-Adv 2026 shortcut findings, V-017; Qwen3-VL pseudo-label data, V-004).
- **Three named closest papers: VERIFIED as described; the stated deltas hold** (Section A).
- **Missed closer prior work: two families the file must confront** (Section B). (1) The incumbent *data engine* is not "Grounding DINO + captioner"; it is VLM-generated referring expressions at scale (2407.14563: 16M model-generated expressions over 1M human-annotated objects, zero-shot RefCOCO SOTA "without using human annotated visual grounding data"; GroundingSuite 2503.10596: multi-VLM-agent pipeline, 9.56M expressions). The kill experiment compares against the weaker baseline. (2) Label-free self-play on unlabeled images for VLMs already exists as a training recipe (Vision-Zero 2509.25541 with Iterative-SPO alternating self-play and RLVR; VisPlay 2511.15661 questioner/reasoner with GRPO), though none targets referring expressions or localization. Verdict on novelty: **PARTIALLY**; the delta survives if it is restated as "a geometric, listener-verified win condition with distractor curricula, benchmarked against VLM-generated-expression data engines at equal size".
- **Section 9 [LIKELY] items: all resolved** (Section C); one wording fix (SPIN), one status change (Molmo2 "RefCOCO-free" → "no RefCOCO annotations, COCO images present", V-112), SAM 3 licence confirmed as a custom "SAM License" with gated checkpoints (V-114).

## A. Novelty test (three closest papers)

| Paper | Says what the author says? | Delta holds? |
|---|---|---|
| A Joint Speaker-Listener-Reinforcer Model for Referring Expressions (1612.09542, Yu, Tan, Bansal, Berg, CVPR 2017) | Yes, P: speaker, listener, reinforcer; "the reinforcer introduces a reward function to guide sampling of more discriminative expressions"; listener-speaker trained jointly end-to-end; three RefCOCO-family datasets. | Yes: RefCOCO human expressions as anchor, task-specific CNN-LSTM modules, no open-domain pool, no curriculum, no rejection mode. |
| Language Self-Play for Data-Free Training (2509.07414) | Yes, P: single model plays against itself in a competitive game; Llama-3.2-3B on instruction following, math, code; no images. | Yes. |
| Syn-GRPO (2511.19343) | Yes, P: asynchronous data server synthesizes images inside GRPO; diversity reward; three perception tasks. | Yes. |

## B. Closer prior work the author missed

1. **VLM-generated referring-expression data engines (the real incumbent baseline).**
   - Learning Visual Grounding from Generative Vision and Language Model (2407.14563), P: prompts a generative VLM with object regions from detection datasets, adds "attribute modeling and spatial relation modeling" to mimic referring-expression language; 500K images, 1M objects, 16M expressions; "the first grounding dataset with purely model-generated queries and human-annotated objects"; zero-shot RefCOCO REC/RES beats prior SOTA without human grounding labels.
   - GroundingSuite (2503.10596, ICCV 2025), P: "automated data annotation framework leveraging multiple VLM agents", 9.56M expressions with masks, GSEval 3,800 images, 4.5x faster than GLaMM's pipeline; training on it gives gRefCOCO cIoU 68.9.
   - Consequence: IDEA-13's kill experiment ("game-generated vs equal-size Grounding-DINO-plus-captioner pseudo-labels") tests the wrong incumbent. The equal-size baseline must be a VLM-generated expression set (GroundingSuite-style or 2407.14563-style, ideally with their attribute/relation prompting), and the claim must be that *listener-verified informativeness with hard distractors* beats *unverified VLM descriptions*. That is a sharper and more defensible delta.
2. **Listener-scored speakers (pragmatics line).** Andreas & Klein 2016 (1604.00562), P: neural listener scores speaker candidates; "In human evaluations on a referring expression game, our approach succeeds 81% of the time, compared to a 69% success rate"; trained from ordinary captions only. Pragmatic Inference with a CLIP Listener for Contrastive Captioning (2306.08818), P (listing): off-the-shelf CLIP as listener in a reference game. Add both to Section 5; the 2017 paper is not the only ancestor.
3. **Label-free self-play for VLMs on arbitrary images.** Vision-Zero (2509.25541), P: "Who Is the Spy"-style games "from arbitrary images", "label-free, domain-agnostic multi-agent self-play", Iterative-SPO "alternates between Self-Play and reinforcement learning with verifiable rewards"; VisPlay (2511.15661), P: "Image-Conditioned Questioner" and "Multimodal Reasoner" trained with GRPO on "large amounts of unlabeled image data" with diversity and difficulty rewards; also Active-Zero (2602.11241) and SPARK (2605.05546) (S, listings). None has a localization win condition, distractor sets, or a listener that outputs boxes, but the training loop (alternating roles, frozen opponent, GRPO, difficulty reward) is the same. The file must cite Vision-Zero and VisPlay in Section 5 and state the delta as the geometric win condition, the segmenter-derived distractor curriculum and the rejection mode.
4. **Drift remedies:** Lee, Cho, Kiela 2019 (1909.04499), P: "a combination of syntactic (language model likelihood) and semantic (visual grounding) constraints gives the best communication performance". The file's KL-to-base and naturalness-judge terms are the modern form; say so explicitly.

## C. Section 9 claims and numbers

| Claim | Verdict | Evidence |
|---|---|---|
| 1612.09542 details | VERIFIED | P (abstract). |
| 2509.07414 details | VERIFIED | P. |
| GuessWhat?! (1611.08481) characterization | VERIFIED | P: two-player guessing game, 150K games, 800K QA, 66K images, CVPR 2017. |
| Lee et al. 2019 (1909.04499) characterization | VERIFIED | P (see B.4). |
| SPIN (2401.01335) "frozen opponent copy" | PARTIALLY | P: abstract says the model "generates its own training data from its previous iterations" and discriminates them from human data; the "frozen previous-iteration opponent" reading is fair but the abstract does not use those words. |
| CoT Referring (2510.06243) | VERIFIED | P: structured CoT referring steps, unified detection+segmentation MLLM, +2.5% on RefCOCO/+/g and a curated complex-referring benchmark. |
| 2602.10815 (data-centric RL vs SFT) | VERIFIED | P: "RL's generalization advantage arises from an implicit data filtering mechanism that inherently prioritizes medium-difficulty training samples"; difficulty-curated SFT "surpasses the performance of RL-based training". Implication for the listener phase: the matched-data SFT control must also be difficulty-matched, otherwise the RL-vs-SFT comparison is confounded. |
| Molmo2 licence and RefCOCO-free status | PARTIALLY | V-112: no RefCOCO in the named pointing sets; COCO images present via VQA-style sets in the Molmo lineage; licence per Ai2 blog Apache-2.0 (S). |
| SAM 3 concept-prompt behaviour on the noun list | UNVERIFIABLE | not testable from documents. |
| SAM 3 licence "custom; check redistribution" | VERIFIED as custom | P: repo states "SAM License"; checkpoints gated behind a request on Hugging Face; derived-data terms not read (V-114). |
| GroundingME 45.1 driven by Discriminative/Spatial | PARTIALLY | P: the top model's dimension scores are Discriminative 69.6, Spatial 49.7, Limited 54.0, Rejection 0.0; Rejection and Spatial, not Discriminative, are the weakest dimensions for the top model (V-111). Adjust the sentence. |
| Ref-Adv 2026 shortcut findings | VERIFIED | V-017. |
| Qwen3-VL data recipe cited as "Verifier V-006" | wrong row id | it is V-004 (V-006 is the GUI numbers). |
| RSC / ScenGround benchmark sizes (31k / 4k / 3k) | VERIFIED (S) | Skeptic's abstract read (2604.02323); not re-read by me. |

## D. Actions requested of the author

1. Replace the kill-experiment baseline with an equal-size VLM-generated expression set (GroundingSuite pipeline or 2407.14563 prompting) and add both papers to Section 5.
2. Add Vision-Zero and VisPlay to Section 5 and restate the delta (geometric win condition, distractor curriculum, rejection mode).
3. Add Andreas & Klein 2016 and the CLIP-listener paper to the pragmatics lineage.
4. Make the listener-phase SFT control difficulty-matched (2602.10815).
5. Fix the GroundingME sentence (Spatial and Rejection are the weakest dimensions), the V-006 → V-004 reference, and the Molmo2 wording.

## C3. Addendum (2026-09-02): the REVISED file (round 1 answered), new ids in Section 5 checked

Status of the file at check time: REVISED. All ids added in the revision were resolved through the arXiv API (V-193 to V-195).

| Claim in the revised Section 5 | Verdict | Evidence |
|---|---|---|
| Strub et al. 2017 (1703.05423): RL for visually grounded goal-driven dialogue | VERIFIED | P: policy gradients over GuessWhat-style dialogues, 120k dialogues (V-193). Drop the [LIKELY]. |
| Seeded Iterated Learning (2003.12694) and Lazaridou et al. 2020 (2005.07064) as drift countermeasures | VERIFIED | P: SIL re-seeds from teacher agents to counter drift; Lazaridou combines multi-agent self-play with a pretrained LM for visual referential communication and gives a drift taxonomy plus reranking (V-193). Drop the [LIKELY]. |
| CLIP as off-the-shelf listener (2306.08818) | VERIFIED | P: +11 to 15% human-evaluated accuracy for contrastive captions (V-193). Now primary, not "listing". |
| GREx (2601.05244): joint generation, comprehension, segmentation, "trained against references rather than a listener" | VERIFIED (title and scope) / PARTIALLY (training detail) | P: Generalized REG/REC/RES with multi-target and no-target expressions on gRefCOCO, ReLA baseline; how the generator is supervised is not in the abstract (V-194). |
| Active Zero (2602.11241), self-evolution with self-judging (2603.21289) as VLM self-play loops | VERIFIED | P: Active Zero has Searcher / Questioner / Solver co-evolving VLM agents; 2603.21289 uses a bounded Judge under GRPO for multimodal reasoning (V-194). Neither has a localization win condition, as the file says. |
| SPARK (2605.05546) listed under "Self-play for VLMs" | **REFUTED** | P: "Self-Play with Asymmetric Reward from Knowledge Graphs" is text-only self-play over knowledge graphs built from scientific literature, for multi-hop relational reasoning (V-195). Remove it from that line. |
| RSC / ScenGround (2604.02323): difficulty-aware curriculum RL for grounding; benchmark | VERIFIED | P: ~31k train / 7k test; SFT warm start plus difficulty-aware RL (V-194). Consistent with the 31k / 4k / 3k split quoted earlier if the two test sets sum to 7k. |
| IC-Seg (2605.17531) and 2608.23978 as interactive-grounding benchmarks with model answerers | VERIFIED (existence and scope) / PARTIALLY ("model answerers" not in either abstract) | P (V-194). 2608.23978 finds LVLMs far below humans when the target must be acquired by asking, which supports the file's speaker-listener framing. |

Earlier actions (Section D of this report): the revised file names GroundingSuite as an incumbent, cites Vision-Zero and VisPlay, and makes the listener SFT control difficulty-matched (grep confirmed the terms are present); the GroundingME "weakest dimension" wording and the V-006 to V-004 reference were not re-audited line by line.

## C4. Addendum (2026-09-02): Researcher1's question on the incumbent data engines (Section 1 and 13-M5)

Claim checked: "neither 2407.14563 nor GroundingSuite conditions its generated expressions on a controlled distractor set or verifies each expression with a listener before use" [LIKELY in the file].

| Paper | Distractor conditioning? | Listener verification? | Verdict |
|---|---|---|---|
| Learning Visual Grounding from Generative VLM (2407.14563), P (body) | No. Objects are cropped to "isolate the objects from the influence of the context" and PaLI-3 is prompted "Describe the major object in the image, ignore the background". | No. "We choose the top-5 answers with highest confidence ... without any further post-processing." | Claim VERIFIED for this paper (V-197). |
| GroundingSuite / GSSculpt (2503.10596), P (body) | Not stated. Prompt templates "emphasize spatial relationships, distinctive visual features, and contextual cues" to make references "distinct and unambiguous", but no controlled distractor set is described (PARTIALLY). | **Yes.** Stage 3 noise filtering "employ[s] instruction-based segmentation models, i.e., EVF-SAM, to identify potentially ambiguous referring expressions by measuring consistency between the generated expression and the corresponding mask"; pairs with IoU below 0.5 are removed. | Claim **REFUTED** for the listener half (V-198). |

Consequences for the file: the delta against GroundingSuite is not "no listener"; it is (i) a controlled, similarity-tiered distractor set that the speaker must defeat, (ii) an RL speaker trained on the listener signal rather than a filter applied after generation, (iii) the curriculum and rejection mode, and (iv) a listener-independent uniqueness check. Baseline B3 must therefore include an EVF-SAM-style (or same-family grounder) consistency filter at IoU 0.5, otherwise it is weaker than the incumbent it stands for. GroundingSuite's images are 2M from SA-1B and its VLM agents are InternVL2.5, Florence-2 and SAM2; 2407.14563's objects come from COCO 2017 and Objects365 v1 with PaLI-3 as the generator.
