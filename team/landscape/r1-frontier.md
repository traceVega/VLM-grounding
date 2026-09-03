# R1 Frontier survey: VLM grounding, late-2024 to 2026-09

Author: Researcher1 (Frontier Explorer). Last updated: 2026-09-01.
Coverage boundary agreed with Researcher3 (proposed, see message log): I cover late-2024 onward. Foundations (Kosmos-2, Shikra, Ferret-v1, GLaMM, Groma, PaliGemma-1, Molmo-1, GLIP/Grounding-DINO lineage, RefCOCO history, POPE/CHAIR) are Researcher3's; I only reference them.

Evidence labels follow the charter. `[VERIFIED: id]` means I read the arXiv abstract/HTML of that paper in this session (WebFetch). `[LIKELY]` is recollection or a search-snippet number I did not open. `[SPECULATION]` is my own conjecture. Unless stated otherwise, numbers come from the paper's own abstract and are self-reported.

How to read this file: Sections 1 to 9 are the survey (per paper: key insight / contribution / limitation, in my words). Section 10 is the opportunity map. Section 11 has six candidate directions. Section 12 lists the claims I want the Verifier to check. Section 13 is the id index.

---

## 1. What current MLLMs can do (capability snapshot)

**Qwen3-VL (Nov 2025)** `[VERIFIED: 2511.21631]`
- Insight: grounding is now a first-class pretraining task, not a fine-tune. Boxes, points and counting (direct, box-based, point-based) are all trained; 3D grounding on Omni3D too.
- Contribution: switched from Qwen2.5-VL's absolute pixel coordinates back to normalized [0,1000] coordinates, justified as robustness to resolution/aspect ratio and simpler post-processing. Reported ODinW-13 48.6 mAP for the 235B-A22B flagship; claims SOTA on CountBench and ScreenSpot-Pro; SUN RGB-D 3D grounding +5.2 over Gemini-2.5-Pro (Thinking variant).
- Limitation (important for us): the box grounding data are "aggregated open-source datasets plus synthetic annotations from Grounding DINO and Qwen2.5-VL". The teacher detectors set the ceiling and inject their biases; the field is training on its own pseudo-labels. `[VERIFIED: 2511.21631]` for the data statement; the ceiling consequence is `[SPECULATION]`.
- Note the flip-flop on coordinates: Qwen2.5-VL used absolute pixels `[VERIFIED: 2502.13923 via search snippet, LIKELY]`, Qwen3-VL normalized. Nobody has published a controlled comparison at Qwen scale; the report gives a rationale, not an ablation.

**InternVL3.5 (Aug 2025)** `[VERIFIED: 2508.18265 via blog/search snippet, LIKELY for exact numbers]`
- RefCOCO-family average about 90 for 14B (90.1) and 92.4 for 241B-A28B. This is the saturation regime: RefCOCO gains are now within 1 point per generation. GUI and embodied grounding added as capabilities.

**Molmo2 (Jan 2026)** `[VERIFIED: 2601.10611]`
- Insight: pointing is the universal grounding primitive; extend it to video (points + timestamps) and tracking, with fully open data and no distillation from closed models.
- Contribution: 7 new video datasets, 2 multi-image datasets; 8B model reports video pointing 38.4 F1 vs Gemini 3 Pro 20.0, video tracking 56.2 J&F vs Gemini 3 Pro 41.1, video counting 35.5 vs Qwen3-VL 29.6. Bidirectional attention over vision tokens.
- Limitation: "competitive" rather than leading on long video; the video-pointing numbers show that even the best proprietary model is far from solved on dense spatio-temporal grounding. The absolute F1 of 38.4 is low.

**MolmoPoint (Mar 2026)** `[VERIFIED: 2603.28069]`
- Insight: text coordinates force the LM to learn a coordinate system and cost many tokens; instead a special pointing token cross-attends to visual tokens and *selects* one, hierarchically (region, sub-patch, sub-sub-patch), with a no-more-points class token.
- Contribution: PointBench 70.7 (claimed SOTA), +5.7 J&F overall on Molmo2-Track `[VERIFIED by Verifier V-059: Ai2 blog; my earlier "+6.3" was wrong]`, higher sample efficiency than text coordinates. The ScreenSpot-Pro 61.1 (and OSWorld-G 70.0) numbers belong to the separate MolmoPoint-GUI-8B variant, not the base MolmoPoint-8B `[VERIFIED by Verifier V-059]`.
- Limitation: the paper itself says the 28x28-pixel image token (4 ViT patches merged) is "too coarse-grained for precise grounding", which is a statement about all Qwen/Molmo-style merged-patch encoders, not only theirs. Hierarchy partially fixes it; precision at the sub-patch level is still bounded by what the merged token encodes.

**Gemini 2.5/3 (proprietary)** `[LIKELY: Google developer blog on conversational segmentation; search snippets]`
- Gemini 2.5 outputs box_2d + mask + label JSON; Gemini 3 Pro boxes are described as tighter than competitors. But Molmo2 reports Gemini 3 Pro at 20.0 F1 on video pointing and 41.1 J&F on tracking, so proprietary dense grounding is weak relative to open specialists. `[VERIFIED: 2601.10611]` for the numbers.

**GPT-5 / o3 "thinking with images"** `[VERIFIED: 2507.07999]` for TreeBench: OpenAI-o3 scores 54.87 on TreeBench (405 QA over SA-1B images with traceable box evidence); no model reaches 60. `[VERIFIED: 2604.19697]` StepSTEM: Gemini 3.1 Pro and Claude Opus 4.6 at 38.29% on interleaved STEM reasoning, "still rely heavily on textual reasoning". Also GPT-4o scored 0.9 on ScreenSpot-Pro at release `[LIKELY: 2504.07981 snippet; Verifier V-060 could not confirm from the abstract, needs the paper table]`; GPT-4o underperforms open models on Point-It-Out `[VERIFIED: 2509.25794]`. Takeaway: proprietary models do not dominate grounding; open 4B to 8B specialists do.

**Segmentation-capable MLLMs** (Sa2VA `[LIKELY: 2501.04001]`, Seg-Zero `[VERIFIED: 2503.06520 abstract via snippet]`, Qwen3-VL-Seg `[LIKELY: 2605.07141]`, STAMP `[VERIFIED: 2512.00395]`, Dr. Seg `[LIKELY: 2603.00152]`)
- Seg-Zero's decoupled design (MLLM emits box/points, frozen SAM produces mask, RL on the MLLM) is now the default; its known limitation is that mask quality is fully determined by the MLLM's box/point and errors propagate.
- STAMP's insight: the "trilemma" (dialogue ability, mask quality, speed) comes from forcing masks through either embedding prediction (conflicting objectives) or sequential tokens (slow). Its fix is non-autoregressive all-mask prediction as a parallel fill-in-the-blank over image patches after the text. This is one of several 2026 papers saying autoregression is the wrong decoder for spatial outputs (see Section 8).

**Other 2025 open models** (Kimi-VL `[LIKELY: 2504.07491]`, GLM-4.1V/4.5V-Thinking `[LIKELY: 2507.01006]`, MiMo-VL `[LIKELY: 2506.13642]`, Llama 4, PaliGemma 2 `[LIKELY: 2412.03555]`, Florence-2 `[LIKELY: 2311.06242]`, STEP3-VL-10B `[LIKELY: 2601.09668]`): all ship box/point grounding; none introduces a new grounding mechanism that I could find. MiMo-VL-7B and Qwen3-VL-8B are used as backbones in the temporal grounding attention paper `[VERIFIED: 2605.21954]`. I did not read these reports; Researcher2 may cover their numbers.

---

## 2. Training methods: what is actually known about RL vs SFT for grounding

**The canonical claim** ("SFT memorizes, RL generalizes", `[LIKELY: 2501.17161]`) was made on rule-based card/navigation tasks, not grounding. The grounding-specific evidence is thinner than the citation count suggests.

**VLM-R1 (Apr 2025)** `[VERIFIED: 2504.07615 via snippet, LIKELY for details]`: R1-style GRPO with IoU reward on REC; RL beats SFT out-of-domain (LISA-Grounding). Visual-RFT `[LIKELY: 2503.01785]`, Vision-R1 `[LIKELY: 2503.06749]`, UniVG-R1 `[LIKELY: 2505.14231]` are variations (IoU/format rewards, difficulty weighting). These are the "GRPO + reward" papers the Skeptic will flag as saturated.

**RL makes MLLMs see better than SFT (Oct 2025)** `[VERIFIED: 2510.16333]`
- Insight: RL changes the *vision encoder's* representations, producing "stronger and precisely localized" features; SFT does not. They propose PIVOT (preference-instructed vision optimization) at under 1% of standard vision pretraining compute.
- Contribution: evaluates the encoder directly (ImageNet, segmentation, gradient visualization) rather than only downstream VQA.
- Limitation: abstract-level claims; mechanism ("why" RL localizes) not explained; no grounding-precision (IoU threshold) analysis.

**GUI-G1 (May 2025)** `[VERIFIED: 2505.15810]`
- Insight: IoU and hit rewards are *size-sensitive*. The abstract says hit and area rewards "allow models to exploit box size"; the specific directions (hit rewards shrink boxes over training, IoU rewards inflate them) are body-level and must be cited to the paper's analysis section `[PARTIALLY per Verifier V-061]`. GRPO's group normalization also over-weights easy samples. Longer reasoning chains *hurt* grounding ("fast thinking" template wins).
- Contribution: 3B model with box-size constraint and difficulty-aware objective: ScreenSpot 90.3, ScreenSpot-Pro 37.1.
- Limitation: GUI only; but the size-drift finding almost certainly applies to REC IoU rewards `[SPECULATION]`.

**GUI-G2 (Jul 2025)** `[VERIFIED: 2507.15846]`: replaces binary hit rewards with Gaussian point and coverage rewards with adaptive variance; +24.7% on ScreenSpot-Pro over UI-TARS-72B. Reward shaping, not a new capability.

**TimeLens (Dec 2025)** `[VERIFIED: 2512.14698]`
- Insight: existing VTG benchmarks are noisy enough that re-annotation re-ranks models; "thinking-free RLVR" with interleaved textual timestamps is the best recipe; they explicitly call the result an incremental baseline, and claim to beat GPT-5 and Gemini-2.5-Flash.
- Lesson for us: two independent lines (GUI-G1, TimeLens) find that verbal chain-of-thought does not help grounding-type outputs. This contradicts the "thinking" narrative.

**Vero (Apr 2026)** `[VERIFIED: 2604.04917]`: 600K RL samples from 59 datasets with task-routed rewards; gains of 2.9 to 5.4 average points across 5 base models; finds that categories must be trained jointly. Grounding is one of the categories but not analyzed separately.

**Faithful GRPO (Apr 2026)** `[VERIFIED: 2604.08476]`: RL-trained reasoning traces are inconsistent with answers 24.5% of the time; constrained policy optimization (Lagrangian) drops that to 1.7% and raises visual-grounding scores by 13% on spatial reasoning. Grounding here is judged by an LVLM, not by boxes.

**Multimodal reward hacking / MO-GRPO** `[LIKELY: 2607.09492, 2509.22047]`: multi-objective rewards (format + IoU + answer) are dominated by the highest-variance term under group normalization.

**Syn-GRPO (Nov 2025)** `[VERIFIED: 2511.19343]`: an *online image generator* inside the GRPO loop synthesizes new training images from descriptions, with a diversity reward; asynchronous data server makes this practical. Relevant because it shows generative models can sit inside an RL loop for perception tasks (used in direction D-R1b).

**What is NOT known** (gaps I could not find papers on):
- Whether RL improves *precision* (P@0.9, small objects) or only coarse hit rate. Hi-Token and LocateAnything (Section 8) claim high-IoU gains from representation changes; no RL paper reports IoU-threshold curves that I saw. `[SPECULATION]`
- Whether RL's OOD gains survive at 32B+ scale or vanish when the base is already grounding-pretrained (Qwen3-VL). Most RL grounding papers use 3B to 7B Qwen2.5-VL. `[LIKELY]`
- Any label-free reward for grounding (all use GT boxes or pseudo-labels from detectors).

---

## 3. Grounded reasoning ("thinking with images") and whether the grounding is faithful

**The tool-use generation** (DeepEyes `[LIKELY: 2505.14362]`, Chain-of-Focus `[LIKELY: 2505.15436]`, Pixel Reasoner `[LIKELY: 2505.15966]`, Mini-o3 `[LIKELY: 2509.07969]`, ZoomEye `[LIKELY: 2411.16044]`, Thyme `[LIKELY: 2508.11630]`): RL teaches the model to emit crop/zoom calls; gains on V*-style high-resolution benchmarks. Survey: `[LIKELY: 2506.23918]`.

**On the Faithfulness of Visual Thinking (Oct 2025)** `[VERIFIED: 2510.23482]`
- Insight: intervene on the visual steps vs the textual steps of a multimodal CoT. Predictions barely change under visual intervention and change a lot under textual intervention: the crops are largely decorative.
- Contribution: reliability and sufficiency metrics via an LVLM judge; SCCM (sufficient-component-cause) learning that pushes traces toward minimal sufficient visual components; plug-in for RFT.
- Limitation: judge-based metric; the fix is a regularizer, not a guarantee; does not address *where* the model should look before it has looked.

**Beyond Accuracy / ViEBench (Jan 2026)** `[VERIFIED: 2601.11633]`: 200 high-resolution images with expert-annotated evidence; a 2x2 diagnosis shows models reach correct answers from irrelevant regions and, conversely, find the right evidence and misuse it. Small benchmark; useful protocol.

**TreeBench / TreeVGR (Jul 2025)** `[VERIFIED: 2507.07999]`: 405 QA with traceable box evidence; o3 at 54.87. TreeVGR (Qwen2.5-VL-7B, RL with joint localization + answer reward) gains +16.8 on V*, +13.4 on TreeBench. Note: supervises boxes with labels; does not test causal use.

**DeFacto (Sep 2025)** `[VERIFIED: 2509.20912]`
- Insight: evidence-grounded reasoning can be enforced by *counterfactual* images: positive (evidence intact), counterfactual (evidence altered), random-masked. GRPO with answer, format and evidence-consistency rewards; DeFacto-100K built by an automatic localize-and-alter pipeline; DeFacto-1.5K human benchmark.
- Limitation: masking/altering produces unnatural images; the pipeline decides the evidence region offline, so the policy never has to *discover* it; evaluation is answer-level.

**Test-time Scaling over Perception / the grounding paradox (Apr 2026)** `[VERIFIED: 2604.11025]`
- Insight: to inspect the right region you must already know which detail matters, which requires having inspected it; grounding errors early in a trace are irreversible because everything downstream conditions on the wrong crop; answer-level voting throws away the trajectory information that would reveal this.
- Contribution: entropy-gated exploration over perception trajectories plus an evidence ledger for iterative refinement.
- Limitation: test-time cost; no training signal.

**Thinking with Visual Grounding (Jun 2026)** `[VERIFIED: 2606.16122]`: interleave points/boxes with reasoning text; synthesize supervision by distilling correct traces, extracting needed objects, grounding them with a SAM3-based agent; RL with answer + dense grounding rewards. Gemma3-4B matches 27B on spatial reasoning; points suit counting, boxes suit spatial tasks. Limitation: grounding supervision is again pipeline-derived; faithfulness is assumed, not tested.

**Zooming without Zooming (Feb 2026)** `[VERIFIED: 2602.11858]`: distill region-level supervision into full-image training so a single forward pass gets most of the zoom gain; ZoomBench (845 VQA) quantifies the global-vs-regional "zooming gap" and identifies when agentic zooming is still necessary. Lesson: much of "thinking with images" is a resolution/attention problem that can be internalized.

**InnerZoom (Jun 2026)** `[VERIFIED: 2606.30084]`: GUI grounding; intermediate decoder layers already carry target-region awareness that is lost by the time coordinates are emitted; bridging that evidence across layers beats two-pass zoom-in by 1.3 points at 31.8% lower latency (OSWorld-G 64.7 for a 4B model). This is direct evidence for the "latent grounding gap" (Section 8, direction D-R1a).

**Other evaluation signals**: StepSTEM `[VERIFIED: 2604.19697]` (models rely on text even when images are necessary); Faithful GRPO (above); ReGround `[LIKELY: 2608.04385]` (self-diagnosis and visual re-examination in multi-step reasoning); "Thinking with Programming Vision" `[LIKELY: 2512.03746]`; UHR-Micro `[VERIFIED: 2605.12237]` ("resolution illusion": higher input resolution gives the appearance of detail without reliable perception of small evidence, not fixed by scale).

**Bottom line for this area**: three independent measurements (2510.23482, 2601.11633, 2604.19697) say the visual steps in current traces are weakly causal. Training fixes so far (DeFacto, SCCM, Faithful GRPO, TreeVGR, 2606.16122) either need offline evidence annotations or use judge-based rewards. A verifiable, label-free causal reward is missing.

---

## 4. Hallucination, attribution and grounded description

**Ground What You See (Jan 2026)** `[VERIFIED: 2601.06224]`: RL-induced hallucination has three causes (over-reliance on chained visual reasoning, low exploration diversity, conflicting samples); fix: localize relevant regions early and describe them concisely, plus caption feedback. Shows that RL for reasoning can *increase* hallucination unless grounding is enforced.

**When Language Overwrites Vision (May 2026)** `[VERIFIED: 2605.08245]`: decoder VLMs over-align visual embeddings with the text manifold; the linguistic bias lives in the top principal components of a universal text subspace; projecting it out (training-free) reduces POPE/CHAIR/AMBER hallucination. A geometric explanation of "text prior beats vision" that is also relevant to grounding shortcuts (Section 9).

**Positive-and-Negative Decoding (ACL Findings 2025)** `[VERIFIED: 2604.24396]`: contrastive decoding that amplifies attended visual evidence and penalizes generation from degraded object features; up to +6.5% on POPE/MME/CHAIR. One of many decoding tricks; not a grounding mechanism.

**DetailVerifyBench (Apr 2026)** `[VERIFIED: 2604.05623]`: 1,000 images, 5 domains, captions over 200 words with token-level annotation of multiple hallucination types; the task is to *localize* the erroneous spans. There is no spatial attribution of claims to regions in it, as far as the abstract says.

**DenseWorld-1M (Jun 2025)** `[VERIFIED: 2506.24102 via snippet, LIKELY]`: dense grounded captions at scale via segmentation-model-mediated region captioning. Data, not a training objective.

**BICR: Grounded or Guessing (May 2026)** `[VERIFIED: 2605.10893]`: confidence probe trained with a ranking loss between hidden states on the real image vs a blacked-out image; best calibration and discrimination across five LVLMs with 4 to 18x fewer probe parameters. Directly measures "did the image matter", which is the question behind every grounding-faithfulness claim.

**Transcoders trace visual grounding (May 2026)** `[VERIFIED: 2605.22902]`: transcoder attributions from image patches to output tokens are more stable under patch ablation than SAE attributions; graph features over these circuits predict hallucination at AUC 0.68 (Gemma 3-4B only). Early, but it is the first circuit-level account of grounding vs prior.

**Gap**: grounded *description* stalled after GLaMM `[LIKELY: 2311.03356]` at "noun phrases get boxes". Attributes, counts and relations, which are what DetailVerifyBench-style captions hallucinate, are not attributed, and there is no reward that scores attributed claims. See direction D-R1d.

---

## 5. GUI / screen grounding

**State of play**: ScreenSpot-Pro `[VERIFIED: 2504.07981 abstract via snippet]` launched with the best model at 18.9 and GPT-4o at 0.9. Since then: GUI-Actor-7B 44.6 `[VERIFIED: 2506.03143]`, GTA1-7B 50.1 `[VERIFIED by Verifier V-067: GUI-C2 Table 3, 2605.30884]` (my earlier "OSWorld-G 72.2" for GTA1 came from a search snippet and is unsupported; dropped. MAI-UI, 2512.22047, lists GTA1-32B at 65.2 on OSWorld-G), GUI-CURSOR 56.5 `[LIKELY: snippet]`, MolmoPoint-8B 61.1 `[VERIFIED: 2603.28069]`, InnerZoom-4B OSWorld-G 64.7 `[VERIFIED: 2606.30084]`, UI-Venus-1.5 `[LIKELY: 2602.09082]`, Qwen3-VL claims SOTA `[VERIFIED: 2511.21631]`. Numbers are not directly comparable across papers (model sizes, thinking modes, and some report test-time scaling).

**GUI-Actor (Jun 2025)** `[VERIFIED: 2506.03143]`
- Insight: text-coordinate generation has weak spatial-semantic alignment, cannot represent ambiguous supervision (any point in the element is correct), and mismatches the patch granularity of the encoder. An <ACTOR> token attends over patch tokens and proposes regions in one pass; a verifier picks among them.
- Contribution: 7B beats UI-TARS-72B on ScreenSpot-Pro; training only the 100M-parameter head on a frozen backbone is competitive; better generalization to unseen resolutions.
- Limitation: needs GUI labels; the attention head is task-specific; the verifier is a separate model.

**Attention-based training-free GUI grounding**: TAG `[VERIFIED: 2412.10840 abstract via snippet]` aggregates attention from query tokens without fine-tuning; Trifuse `[VERIFIED: 2602.06351]` fuses MLLM attention, OCR anchors and icon captions with a consensus-single-peak rule because attention alone is unreliable on GUIs.

**Regression-free layout-aware matching (Aug 2026)** `[VERIFIED: 2608.09654]`: a frozen MLLM writes a structured description with layout cues; a small model matches it against layout-prior candidates trained only on text/icon binary labels; +20% on ScreenSpot-Pro vs end-to-end and +15% Mind2Web. Frames end-to-end coordinate output as "coordinate hallucination".

**RULER tokens + I-MRoPE (Oct 2025)** `[VERIFIED: 2510.03230]`: explicit coordinate gridline tokens the model can reference, plus symmetric interleaved MRoPE; largest gains on high-resolution screens, i.e. exactly where implicit patch-to-pixel mapping breaks.

**Uncertainty**: SafeGround `[VERIFIED: 2602.02419]` uses spatial dispersion of stochastic samples as uncertainty and a Learn-Then-Test threshold with FDR control; +5.38 points system accuracy over Gemini-only on ScreenSpot-Pro. UI-Zoomer `[LIKELY: 2604.14113]` uses uncertainty to decide when to zoom. Zoom-in Click-out `[LIKELY: 2512.05941]`, MEGA-GUI `[LIKELY: 2511.13087]`, R-VLM `[LIKELY: 2507.05673]`: multi-stage crop pipelines.

**What remains hard** (from ScreenSpot-Pro analysis and follow-ups, `[LIKELY: snippets]`): icons over text (icon errors were 76.9% of failures on ScreenSpot-V2 in one analysis), dense professional UIs at 4K, functional/semantic instructions ("the tool that would do X") rather than visible labels, and unseen resolutions. My reading: GUI grounding is now the field's best-instrumented testbed for the *representation* question (Section 8) because targets are rectangles, labels are cheap, and the ambiguity of "any point inside" is explicit.

**Learning active perception via self-evolving preference optimization** `[LIKELY: 2509.04243]`: GUI models learn when to crop via preference optimization; another "zoom as action" paper.

---

## 6. Video temporal and spatio-temporal grounding

- **RL for VTG** (Time-R1 `[LIKELY: 2503.13377]`, Tempo-R0 `[LIKELY: 2507.04702]`, Video-in-the-Loop `[LIKELY: 2510.04022]`, "Datasets and Recipes for VTG via RL" `[LIKELY: 2507.18100]`, TimeLens `[VERIFIED: 2512.14698]`): converged on GRPO with tIoU reward; TimeLens says thinking-free is best and the benchmarks were noisy.
- **MLLMs Know When Before Speaking (May 2026)** `[VERIFIED: 2605.21954]`
  - Insight: a sparse set of "temporal grounding heads" attend to the ground-truth interval during *prefill*; during decoding, answer tokens drift to salient but irrelevant segments. The model knows when, then loses it while speaking.
  - Contribution: read the heads, extract the interval, re-run with restricted visual context; up to +3.5 mIoU on MiMo-VL-7B, Qwen3-VL-8B, TimeLens-8B without training.
  - Limitation: relies on finding the heads per model; inference-only.
- **Open-o3 Video (Oct 2025)** `[VERIFIED: 2510.20579]`: reasoning with explicit timestamps + boxes; STGR data; cold-start RL with answer, temporal and spatial rewards; V-STAR mAM +14.4, mLGM +24.2 over Qwen2.5-VL; grounded traces enable confidence-aware test-time scaling.
- **STVG-R1 (Feb 2026)** `[VERIFIED: 2602.11730]`: sidesteps coordinates entirely by painting temporally consistent instance IDs onto frames as visual prompts and having the model name IDs; +20.9 mIoU on HCSTVG-v2 over Qwen2.5-VL-7B; zero-shot 47.3 J&F on MeViS. Insight worth generalizing: identity tokens beat coordinate strings whenever the same object recurs.
- **VideoMolmo** `[LIKELY: 2506.05336]`, **Molmo2** `[VERIFIED: 2601.10611]`, **MolmoMotion** `[LIKELY: 2606.18558]` (3D point-trajectory forecasting from language), **EvoGround** `[LIKELY: 2605.13803]`, **GIRL-DETR** `[LIKELY: 2606.00775]`, **EVIDENT** `[LIKELY: 2605.26104]`.
- **Open**: dense video pointing is at 38 F1 for the best open model and 20 for Gemini 3 Pro; identity persistence across frames, turns and multiple images has no clean formulation inside an MLLM except STVG-R1's painted IDs; long-video grounding evaluation is unreliable (TimeLens).

---

## 7. 3D, embodied, spatial grounding and pointing for robotics

- **RoboRefer / RefSpatial (NeurIPS 2025)** `[VERIFIED: 2506.04308 abstract via snippet, LIKELY for numbers]`: disentangled depth encoder + SFT then RFT; 20M QA, 31 spatial relations, multi-step reasoning; RFT beats Gemini-2.5-Pro by 12.4 on RefSpatial-Bench. Standard recipe; the interesting part is the claim that depth must be a *separate* encoder.
- **N3D-VLM (Dec 2025)** `[VERIFIED: 2512.16561]`: native 3D localization from text with RGB or RGB-D; lifts 2D annotations to 3D with depth estimation (6x the largest single-image 3D detection set).
- **SpatialPoint** `[LIKELY: 2603.26690]`, **G2VLM** `[LIKELY: 2511.21688]` (geometry-grounded VLM with 3D reconstruction), **PointVG-R** `[LIKELY: 2606.24539]` (visual CoT for precise pointing).
- **Point-It-Out (Sep 2025)** `[VERIFIED: 2509.25794]`: three-stage protocol (referred-object localization, task-driven pointing, visual trace prediction) over indoor/kitchen/driving/manipulation; GPT-4o underperforms open models; Molmo is good at S1/S2 and fails S3 (trace planning). "Where is it" and "where to act" are different skills.
- **Embodied3DBench** `[LIKELY: 2605.29074]`, **EPIC-Bench** `[LIKELY: 2605.17070]`, **ReVSI** `[LIKELY: 2604.24300]`, **MultihopSpatial** `[LIKELY: 2603.18892]`, spatial survey `[LIKELY: 2510.25760]`: benchmark proliferation; most report large gaps.
- **Rethinking VLM Representation for VLA Initialization (May 2026)** `[VERIFIED: 2605.25802]`: the pretrained VLM representation is itself a major source of action performance; embodied-VQA adaptation helps inconsistently and non-additively; LoRA beats full fine-tuning because reshaping the representation hurts. Implication: grounding pretraining's value for control is unproven and possibly negative if it distorts the representation. `[SPECULATION]` on the last clause.
- **PGT (May 2026)** `[VERIFIED: 2605.23883]`: procedurally overlaid geometric primitives give dense, unambiguous supervision for relations, counting and depth; +20% What'sUp, +13.3% CV-Bench-2D on LLaVA-1.5, +5.5/+8.3 on SOTA models. Their conclusion: deficits are supervision problems, not architecture/resolution problems. Also describes current MLLMs as "bag-of-words" spatial classifiers.

---

## 8. Representation: coordinates, decoders, attention read-outs, interpretability

This is the most active 2026 cluster and, in my view, the one with the clearest scientific question.

**Evidence that text-coordinate autoregression is the bottleneck**
- Hi-Token (Aug 2026) `[VERIFIED: 2608.03471]`: digit tokens lack numerical order and axis semantics; axis-specific hundreds/tens/ones tokens improve IoU across the whole threshold range on three backbones under matched training; Hi-GAR adds geometry rewards in GRPO.
- LocateAnything (May 2026) `[VERIFIED: 2605.27365]`: boxes decoded token-by-token are geometrically misaligned and slow; decode each box/point as an atomic unit in parallel; 138M-sample dataset; better high-IoU quality and throughput.
- GETok (Dec 2025) `[VERIFIED: 2512.10554]`: grid tokens as spatial anchors + offset tokens for iterative refinement, no architecture change, works in SFT and RL.
- MolmoPoint `[VERIFIED: 2603.28069]`, GUI-Actor `[VERIFIED: 2506.03143]`, STAMP `[VERIFIED: 2512.00395]`, RULER/I-MRoPE `[VERIFIED: 2510.03230]`, regression-free matching `[VERIFIED: 2608.09654]`, Motto/Mixture-of-Thought-Tokens `[VERIFIED: 2607.24407]` (spatially grounded thought tokens that switch grounding modes inside a reasoning chain; PR-Bench for the "perception-reasoning gap"), STVG-R1's painted IDs `[VERIFIED: 2602.11730]`.
- ExpVG design-space study `[VERIFIED: 2508.08066]`: LLaVA-1.5 + RefCOCO only; useful ablations but pre-dates native-resolution encoders and is on a saturated benchmark.
- Qwen went absolute (2.5) then normalized (3) `[VERIFIED: 2511.21631]`; Molmo uses percentage coordinates `[LIKELY]`; no controlled study at scale with high-resolution, small-object evaluation exists that I found.

**Evidence that the model localizes internally better than it verbalizes** (the "latent grounding gap")
- Temporal heads `[VERIFIED: 2605.21954]`; InnerZoom's intermediate-layer awareness `[VERIFIED: 2606.30084]`; Self-Improving Small Object Grounding `[VERIFIED: 2606.01612]` (an IoU regressor trained only on attention maps reaches Pearson r>0.67; attention-entropy ranking of candidate boxes improves small-object localization by up to 19% with no training); GUI-Actor's frozen-backbone result; TAG/Trifuse; Entropy-Gradient Grounding `[VERIFIED: 2604.08456]` (backprop next-token entropy to visual embeddings gives relevance maps that beat attention heuristics across four VLMs and seven benchmarks); Where Does Vision Meet Language `[VERIFIED: 2601.08151]` (fusion at specific layers, a late "review" re-activation of vision before output, persistent attention noise on irrelevant regions); PlaM `[LIKELY: 2601.07645]` (model merging that concentrates attention); Transcoders `[VERIFIED: 2605.22902]`.
- Nobody has (a) measured this gap as a general property across models/tasks/scales, (b) determined whether it is a decoding-format artifact or a perception limit, or (c) used the internal read-out as a training signal rather than a test-time selector. See D-R1a.

**Resolution and tokenization**: MolmoPoint's 28-px token remark; UHR-Micro's resolution illusion `[VERIFIED: 2605.12237]`; Robust grounding against occlusion/small objects `[LIKELY: 2604.24036]`; PixelPrune `[LIKELY: 2604.00886]`; Beyond Static Cropping `[LIKELY: 2602.04304]`. Precision at P@0.9 for small objects is basically unreported by MLLM papers `[LIKELY]`.

---

## 9. Under-explored settings

**Rejection / absent targets** (the single most striking number in this survey)
- GroundingME (Dec 2025) `[VERIFIED: 2512.17495]`: 1,005 human-verified hard cases in four dimensions (discriminative, spatial, limited/occluded-tiny, rejection). Best of 25 MLLMs: 45.1%. Most models: 0% on rejection. Test-time trajectory selection +4.5; data-mixture training lifts rejection from 0% to 27.9%.
- Teaching MLLMs to Say No / RC-GRPO (Aug 2026) `[VERIFIED: 2608.04698]`: models "succumb to compulsory grounding prompts" and hallucinate boxes; RC-GRPO forces None rollouts for advantage estimation on negatives and penalizes over-refusal; notes that SFT and plain RL degrade positives. Three GREC benchmarks.
- PostAlign `[LIKELY: 2506.17901]` (REJ token + rejection loss), OpenRef `[VERIFIED: 2605.25706]` (multi-target and none-target, adverse conditions, N3R metric, training-free consistency checker), RefBench-PRO `[LIKELY: 2512.06276]` (reject sub-task), HKVLM `[VERIFIED: 2606.28862]` (abstaining verifier over a frozen detector; decomposes error into see-error and say-error; monotone faithfulness-recall trade-off; admits lower raw accuracy than end-to-end models), Ref-Adv `[VERIFIED: 2602.23898]` (negation facets).
- Open: *why* rejection is at 0% (format? prior? data?), and selective grounding with risk control outside GUIs.

**Shortcuts and language priors**: Ref-Adv `[VERIFIED: 2602.23898]` (RefCOCO expressions are short, few distractors, redundant descriptors; models degrade sharply when only necessary information is given); RSC scenario grounding `[VERIFIED: 2604.02323]` (paragraph-length goal-based queries with misleading distractors; 31K train, 4K/3K test; curriculum RL); fine-grained compositional REC `[LIKELY: 2502.20104]`; over-alignment geometry `[VERIFIED: 2605.08245]`. Missing: a "blind grounding" baseline (how far does text + category prior get you) across benchmarks. `[SPECULATION]` that it explains a large share of RefCOCO.

**Multi-instance, counting, set outputs**: GroundingME and OpenRef multi-target; Molmo2 counting via points; MolmoPoint no-more-points token; LocateAnything/STAMP parallel decoding; LMM-Det `[LIKELY: 2507.18300]`; aerial MLLM detection baseline `[LIKELY: 2501.09720]` (notes autoregression mismatches detection outputs). No study of order effects, duplicates and omissions as instance count grows. `[LIKELY]`

**Cross-image / long context**: MC-Bench `[LIKELY: 2410.12332]`, GeM-VG `[LIKELY: 2601.04777]` (MG-Data-240K, hybrid RL), multi-image grounding RL `[LIKELY: 2507.00748]`, CrossView Suite `[LIKELY: 2605.18621]`. Thin.

**Uncertainty / calibration of grounding**: SafeGround (GUI), BICR (answers), "Calibrating UQ using grounding" `[LIKELY: 2505.03788]`, MMBoundary `[LIKELY: 2505.23224]`, training-free uncertainty guidance `[LIKELY: 2510.00705]`. No general REC/VTG risk-coverage protocol.

**Benchmarks that replace RefCOCO**: GroundingME, Ref-Adv, OpenRef, RefBench-PRO, RSC, TreeBench, ViEBench, ZoomBench, PR-Bench, PointBench, V-STAR, TimeLens-Bench, ScreenSpot-Pro, OSWorld-G, UI-Vision, Point-It-Out, EPIC-Bench, Embodied3DBench. The bottleneck is no longer benchmarks.

---

## 10. Opportunity map

### Crowded (do not enter without a sharply different angle)
1. GRPO + IoU/point/tIoU reward on an existing REC/GUI/VTG dataset with a 3B-7B Qwen2.5-VL (dozens of papers; reward-shaping variants GUI-G1/G2, Hi-GAR, MO-GRPO).
2. Crop/zoom as an RL action, and its internalization (DeepEyes through InnerZoom, Zooming-without-Zooming, TTSP, UI-Zoomer).
3. Alternative coordinate representations for boxes/points (Hi-Token, GETok, MolmoPoint, LocateAnything, GUI-Actor, RULER, Motto, STAMP): a 2026 wave; one more token scheme will not stand out unless it answers *why*.
4. Training-free attention/gradient grounding (TAG, Trifuse, ACS-Free, Entropy-Gradient, PlaM).
5. "Harder REC benchmark" papers.
6. Spatial-reasoning benchmarks and depth-encoder VLMs.
7. Decoding-time hallucination mitigation (PND and kin).

### Open (evidence exists that the problem is real; no satisfactory method)
A. The latent-vs-verbalized grounding gap as a general property, and label-free exploitation of it (Section 8).
B. Rejection at 0% and selective grounding with guarantees beyond GUI (Section 9).
C. Causal faithfulness of grounding in reasoning traces, with a verifiable (non-judge, non-offline-annotation) reward (Section 3).
D. Attributed long description at the claim level with verifiable rewards (Section 4).
E. Discriminative grounding among near-duplicates and informative language (GroundingME-Discriminative, Ref-Adv), where human data are structurally short.
F. Identity-persistent grounding across frames/images/turns (STVG-R1's painted IDs are a hack pointing at a real need); dense video pointing at 38 F1.
   Stub (per lead decision 2026-09-01, not a Phase 2 direction yet): the best open model reports 38.4 F1 on video pointing and 56.2 J&F on tracking, against 20.0 F1 and 41.1 J&F for Gemini 3 Pro `[VERIFIED: 2601.10611]`, so dense spatio-temporal grounding is unsolved for everyone. STVG-R1 gets +20.9 mIoU on HCSTVG-v2 by painting temporally consistent instance IDs onto frames and letting the model name IDs instead of emitting coordinates `[VERIFIED: 2602.11730]`; MolmoPoint's grounding token selects visual tokens directly and gains +5.7 J&F on Molmo2-Track `[VERIFIED by Verifier V-059]`. Both point at the same missing primitive: a persistent entity handle inside the MLLM that can be referred to, re-grounded and tracked across frames, images and dialogue turns without re-describing the object. The open questions are whether such handles can be learned without painted prompts, whether they survive long videos where SAM 3's memory tracker drifts `[LIKELY: 2511.16719]`, and how to evaluate identity errors separately from localization errors. Researcher3 covers the video lineage; revisit in Phase 2 if fewer than three strong ideas survive.
G. Set-valued grounding and counting at scale inside autoregressive decoders (order, duplicates, omissions).
H. What bounds localization precision: patch merge, positional encoding, coordinate format or decoding; evaluated at high IoU and small objects, at matched data.
I. Task-driven pointing ("where to act") vs object localization (Point-It-Out S3).
J. Data circularity: grounding pretraining data are pseudo-labels from Grounding DINO / Qwen2.5-VL; the ceiling and biases of teachers, and benchmark contamination via those teachers, are unstudied.

### Shared assumptions that may be wrong
- A1 "Emitting coordinates as text measures what the model knows about location." Contradicted by the latent-gap evidence; evaluation conflates decoding with perception.
- A2 "RL generalizes better than SFT for grounding." Evidence is small-scale and coarse-IoU; size-drift reward hacking (GUI-G1) means part of the RL gain may be box-size calibration; no precision curves.
- A3 "Longer reasoning helps grounding." GUI-G1 and TimeLens find the opposite for the localization step; faithfulness studies find the visual steps are decorative.
- A4 "Higher resolution fixes small targets." UHR-Micro's resolution illusion; MolmoPoint's 28-px floor.
- A5 "A box in the trace is evidence the model used that region." Refuted by intervention studies.
- A6 "RefCOCO accuracy is grounding ability." Ref-Adv shows shortcut reliance; the top models are within 1 point of each other at 90-92.
- A7 "Adding negatives fixes rejection." Trades off positives (RC-GRPO); the mechanism of the 0% is undiagnosed.
- A8 "Pseudo-labels from detectors are adequate grounding supervision." Nobody has measured the inherited ceiling.
- A9 "Points/boxes are the right interface to action." Point-It-Out S3.

---

## 11. Candidate directions

Numbering reserved per charter: IDEA-11 to IDEA-16 map to D-R1a to D-R1f if promoted in Phase 2.

### D-R1a (IDEA-11 candidate). The latent grounding gap: MLLMs localize better than they say, and the difference is a free training signal

**Research problem.** Five independent 2026 results show that internal signals of an MLLM (attention heads, entropy gradients, intermediate-layer states) localize the target better than the coordinates the model emits: temporal heads recover +3.5 mIoU `[VERIFIED: 2605.21954]`; InnerZoom's intermediate-layer evidence beats explicit zooming `[VERIFIED: 2606.30084]`; an IoU predictor from attention alone reaches r>0.67 and attention-ranked candidates gain up to 19% on small objects `[VERIFIED: 2606.01612]`; a 100M attention head on a frozen backbone matches SOTA GUI grounding `[VERIFIED: 2506.03143]`; entropy-gradient maps beat attention baselines `[VERIFIED: 2604.08456]`. Each paper treats its finding as a task-specific trick. Nobody has asked whether "the model knows where but cannot say where" is a general property of coordinate-emitting MLLMs, how large it is as a function of object size, task (box/point/mask/interval) and model scale, whether it is a decoding-format artifact or a perception limit, or whether it can supervise the verbalized output on unlabeled images.

**Core insight.** An MLLM contains two grounding systems: an implicit one (where the visual tokens that causally drive the answer are, measurable by ablation, attention or gradient) and an explicit one (coordinate generation). Their disagreement is measurable per example. When the implicit system is more accurate, it can teach the explicit one without labels (self-distillation from the model's own read-out, with a frozen copy as teacher); when they agree, that agreement is a confidence signal; when they disagree, the example is a candidate for abstention or zooming. This unifies three literatures (attention-based grounding, coordinate representation, uncertainty) under one measurement.

**Why now.** Read-out quality crossed a threshold in 2026 (ACS, Entropy-Gradient, Trifuse, temporal heads); open models with strong native grounding and accessible internals exist (Qwen3-VL-8B, Molmo2-8B, InternVL3.5); RLVR infrastructure makes label-free reward training cheap; benchmarks that expose failures at high IoU and small scale exist (GroundingME, OpenRef, Ref-Adv, ScreenSpot-Pro, ZoomBench).

**Method sketch.** Stage 1, audit: for 6 to 10 models x {REC boxes, pointing, GUI, VTG intervals}, compute a read-out localization R (attention aggregation over the referent's answer tokens, entropy-gradient relevance, and a causal variant via patch ablation) and the verbalized localization V; report IoU(R,GT) vs IoU(V,GT) stratified by object size, difficulty and rejection cases, and test whether the gap shrinks with scale or with grounding-token architectures (MolmoPoint) which would indicate a decoding artifact. Stage 2, exploit at test time: constrain coordinate decoding to the read-out region, or select among sampled V by agreement with R (generalizing ACS from small objects to all settings). Stage 3, exploit in training: RL on *unlabeled* images where the reward is agreement between V and R computed by a frozen teacher copy (prevents attention collapse), plus a causal term: masking region V must change the model's answer to a question about the referent (necessity) and keeping only V must preserve it (sufficiency). Stage 4: V-R disagreement as an abstention/zoom trigger, evaluated as risk-coverage on GroundingME-Rejection and OpenRef.

**Three closest papers and the delta.**
- Self-Improving Small Object Grounding in LVLMs (2606.01612): attention-based selection among candidate boxes for small objects on COCO/O365. This direction measures the gap across tasks and scales, tests its cause, and uses the read-out as a training reward rather than a test-time selector.
- MLLMs Know When Before Speaking (2605.21954): temporal only, inference only. This direction generalizes to spatial outputs and closes the gap with training.
- InnerZoom (2606.30084): an architectural bridge trained with GUI labels. This direction needs no labels and no architecture change, and explains when the bridge is needed.
(Also GUI-Actor 2506.03143 and Entropy-Gradient Grounding 2604.08456 as baselines.)

**Main risk.** Both systems share the same perception, so on the hardest cases both fail and the gap is small; the gap may exist only for weaker or older models. Reward hacking: the policy could learn to make V and R agree trivially; the frozen teacher and the causal term are the mitigation. Kill experiment: on Qwen3-VL-8B failures on GroundingME and OpenRef, compute IoU(R,GT); if R beats V by a meaningful margin on fewer than about 20% of failures, stop.

### D-R1b (MERGED with Researcher3's NF-1 into IDEA-91 by lead decision 2026-09-01; Researcher3 is author of record; I contribute the online edit-server design, the control-edit protocol and the Syn-GRPO precedent as a section). Edit-to-verify: generative counterfactual images as verifiable rewards for faithful and rejecting grounding

**Research problem.** Two of the field's open problems are counterfactual questions. Faithfulness: would the answer change if the cited region were different? Rejection: does the model say "none" when the referent is absent? Current methods approximate the counterfactual with gray masks (DeFacto `[VERIFIED: 2509.20912]`), text-only negatives (RC-GRPO `[VERIFIED: 2608.04698]`, PostAlign), or LVLM judges (SCCM `[VERIFIED: 2510.23482]`, Faithful GRPO). Masks are unnatural and detectable, text negatives never test the image, judges can be gamed, and all of them fix the evidence region offline so the policy never has to find it.

**Core insight.** Instruction-following image editors can now remove a referent, swap its attribute, or insert a look-alike distractor with realistic results `[LIKELY: 2025-26 editors such as Qwen-Image-Edit, FLUX Kontext, Gemini image editing]`. An edited pair (I, I') with a known edited region gives labels by construction: the same expression must ground to the edited region in I and be rejected (removal, attribute swap) or split into two targets (distractor insertion) in I'. Cross-pair consistency is a verifiable reward that needs no human boxes and trains causal reliance on the referent region. For thinking-with-images traces, a cited region is faithful only if editing it flips the answer, which gives a non-judge faithfulness metric.

**Why now.** Editor quality; Syn-GRPO `[VERIFIED: 2511.19343]` demonstrated an asynchronous generative data server inside a GRPO loop; SAM3 gives clean edit masks; GroundingME/OpenRef/RC-GRPO give evaluation with negatives and multi-targets.

**Method sketch.** Edit server: for (image, expression, initial box from the policy or a detector) produce removal, attribute-swap and distractor-insertion variants; a verifier VLM checks edit success and discards failures; also produce *control edits* on non-referent regions so that "image was edited" is not itself a cue. Rewards for the policy: on I, IoU with the (self- or teacher-)box; on removal/swap variants, None required; on insertion variants, two boxes required; consistency term: the box on I must overlap the edit mask (otherwise the policy grounded something else). For reasoning traces: reward the cited region if the answer flips under its edit and does not flip under control edits. Train Qwen3-VL-4B/8B with GRPO; evaluate on GroundingME (all four dimensions), OpenRef N3R, gRefCOCO, Ref-Adv negation, DeFacto-1.5K and ViEBench.

**Three closest papers and the delta.**
- DeFacto (2509.20912): masking-based counterfactuals prepared offline for VQA answers. This direction uses realistic edits, generates them online for the policy's own proposals, targets grounding outputs (boxes/None) not only answers, and adds control edits to block artifact shortcuts.
- Teaching MLLMs to Say No / RC-GRPO (2608.04698): refusal via text negatives and reward calibration. This direction produces image-level negatives with paired positives, so the model must use the pixels, and measures faithfulness as well as rejection.
- Syn-GRPO (2511.19343): online image synthesis for response diversity. This direction uses synthesis for counterfactual *supervision* with known edit masks.

**Main risk.** Editor artifacts leak (a detector of "edited" images becomes a shortcut): mitigated by control edits and artifact-matched positives. Small referents are hard to edit convincingly. Cost per rollout. Kill experiment: build 2k edit pairs on RefCOCO/GroundingME images; check that (a) a simple classifier cannot tell edited from control-edited images, and (b) Qwen3-VL and Molmo2 still hallucinate boxes on removal edits at a rate near their GroundingME rejection failure; if either fails, stop.

### D-R1c (IDEA-13 candidate). Referring games at MLLM scale: self-play speaker-listener training for discriminative grounding

**Research problem.** Models fail when the target has near-duplicates and the language must be informative (GroundingME-Discriminative and Spatial `[VERIFIED: 2512.17495]`, Ref-Adv `[VERIFIED: 2602.23898]`, RSC `[VERIFIED: 2604.02323]`). Human referring data are structurally short and under-informative (the RefCOCO shortcut analysis in Ref-Adv), and pseudo-labels from detectors carry no language at all. There is no scalable source of *informative* expressions paired with hard distractors.

**Core insight.** Informativeness of a referring expression is verifiable by a listener: an expression is good if and only if a listener picks the target uniquely among distractors. One MLLM can play both roles. With object proposals and categories from SAM3 (no language labels), the speaker is rewarded when the listener localizes the target among same-category instances, and the listener is trained on the resulting expressions. Difficulty is controlled by the number and similarity of distractors, giving a curriculum for exactly the dimensions where GroundingME says models are weakest. The classic pitfall of emergent communication (private codes) is controlled by a frozen listener or a KL/naturalness term; the payoff is Gricean, discriminative language and a listener robust to hard distractors, with no human annotation.

**Why now.** The 2017 speaker-listener-reinforcer work `[LIKELY: 1612.09542, Yu et al. CVPR 2017]` used small CNN/LSTM models on RefCOCO only. Now one model can speak and ground at open vocabulary, SAM3 supplies instances at scale, RL infrastructure exists, and the failure mode is documented by benchmarks that did not exist before 2025.

**Method sketch.** Image pool: SA-1B or OpenImages, filtered to images with k>=3 same-category instances (SAM3 masks + labels). Round t: speaker (policy) describes target r; listener (frozen copy at round t-1, later the same policy in listener mode) grounds; speaker reward = success(IoU>0.5 and no other instance selected) minus length penalty minus KL to base model (prevents drift); listener reward = IoU against the target under expressions from the current speaker, mixed with human data replay. Add a negative mode where the speaker describes a nonexistent object and the listener must reject (feeds rejection). Evaluate listener transfer on GroundingME (Discriminative, Spatial, Rejection), Ref-Adv, OpenRef, RefCOCO+ (no location words), and speaker quality via human listeners.

**Three closest papers and the delta.**
- A Joint Speaker-Listener-Reinforcer Model for Referring Expressions (1612.09542): joint training of small task-specific modules on RefCOCO. This direction is self-play of a single generalist MLLM over an open, unlabeled image pool with distractor-controlled curricula and explicit drift control.
- Language Self-Play for Data-Free Training (2509.07414): text-only self-play. This direction grounds the game in images with a verifiable geometric win condition.
- Syn-GRPO (2511.19343): synthesizes images to diversify responses. This direction synthesizes *language* against fixed images, targeting discriminative grounding.

**Main risk.** Language drift to codes; collapse to easy distractor sets; the listener overfits speaker idiosyncrasies; contamination if RefCOCO images are in the pool. Kill experiment: one round of speaker RL against a frozen listener on 10k images; train a listener on the produced expressions and compare against a listener trained on equal-size Grounding-DINO pseudo-labels; if GroundingME-Discriminative and Ref-Adv do not improve after two rounds, stop.

### D-R1d (MERGED with Researcher3's NF-2 into IDEA-92 by lead decision 2026-09-01; I am author of record; Researcher3 contributes the necessity-and-sufficiency formalism and the causal experiment of attribution reward vs text-correction at equal data). Attributed description: every claim carries a region, every region verifies its claim

**Research problem.** Long captions hallucinate at the span level `[VERIFIED: 2604.05623]`, and RL for reasoning can raise hallucination `[VERIFIED: 2601.06224]`. Grounded captioning stopped at GLaMM-style noun-phrase boxes `[LIKELY: 2311.03356]`; attributes, counts and relations, which are what long captions get wrong, are never attributed, and no reward scores attributed claims.

**Core insight.** Attribution converts hallucination detection into grounding verification. A claim-region pair can be checked by a verifier that sees only the crop (attributes), the two crops and their layout (relations) or the point set (counts). "Attributed precision x coverage" is computable without caption annotations and gives users a verifiable interface (click a claim, see its evidence). Adding a necessity term (the claim must fail on a counterfactually edited region, from D-R1b) makes the attribution causal rather than decorative.

**Why now.** Fluent box/point emission in Qwen3-VL and Molmo2; SAM3 for mask-level attribution; DetailVerifyBench for span-level evaluation; DenseWorld-1M as seed data; the BICR blind-image probe as an "image mattered" signal.

**Method sketch.** Output format: interleaved text with typed claim tags (entity, attribute, count, relation) each carrying region(s). Verifier ensemble (frozen VLM on crops; detector/pointing model for counts; the base model with blind-image contrast for "was the image used"). Reward = verified-claim precision minus unattributed-content penalty plus coverage against a held-out reference judged coarsely; GRPO on Qwen3-VL-8B starting from a small SFT on auto-attributed DenseWorld-1M captions. Evaluate on DetailVerifyBench (span localization of hallucinations), CHAIR/AMBER, and a new attributed-precision metric with a human audit of 1k claims.

**Three closest papers and the delta.**
- GLaMM (2311.03356): noun-phrase-to-mask grounded conversation via SFT. This direction attributes attributes, counts and relations, and trains with a verification reward.
- Ground What You See (2601.06224): caption feedback and localization to reduce RL hallucination. This direction makes attribution explicit in the output and scores each claim.
- Thinking with Visual Grounding (2606.16122): step-level grounding for QA with pipeline supervision. This direction targets open-ended description with verifier rewards and no per-claim labels.

**Main risk.** The verifier hallucinates and gets gamed (reward hacking on the judge), especially on relations; long outputs make RL expensive; reviewers may read it as "GLaMM + RL". Kill experiment: attribute existing Qwen3-VL captions post hoc with SAM3, verify with a crop-only judge, and compare to DetailVerifyBench human labels; if judge AUROC for attribute and relation claims is below about 0.75, the reward is too noisy.

### D-R1e (MERGED into D-R1a / IDEA-11 by lead decision 2026-09-01; becomes its evaluation half and Stage 4, together with Researcher3's NF-4 scoring-rule confidence reward). Selective grounding: risk-coverage evaluation and calibrated abstention across REC, GUI and video

**Research problem.** Most MLLMs score 0% on rejection `[VERIFIED: 2512.17495]`; the only guarantee-bearing method is GUI-specific `[VERIFIED: 2602.02419]`; REC and VTG are evaluated by accuracy alone, which hides the "guess a box anyway" behaviour.

**Core insight.** Rejection is a decision under uncertainty, so the right metric is a risk-coverage curve and the right method is calibration of a few internal evidence statistics (dispersion of sampled boxes, attention concentration, blind-image contrast, V-R agreement from D-R1a) with distribution-free risk control, plus a third action (zoom/ask) when the evidence is insufficient rather than absent.

**Why now.** SafeGround's recipe, BICR's probe, the read-out signals of D-R1a and benchmarks with negatives (GroundingME, OpenRef, gRefCOCO) all appeared in the last 12 months.

**Method sketch.** Define selective grounding: output a box, "multiple", "none", or "inspect (zoom)"; fit a conformal threshold per action with FDR control on a calibration split; report risk-coverage AUC across REC, GUI and VTG for 8 to 10 models. Then train a lightweight abstention head on the evidence statistics and show it transfers across benchmarks without retraining the backbone.

**Three closest papers and the delta.** SafeGround (2602.02419): GUI only, dispersion only. HKVLM (2606.28862): abstention through a frozen detector, lower raw accuracy. RC-GRPO (2608.04698): trains refusal into the policy; this direction adds guarantees and a cross-task protocol without touching the backbone.

**Main risk.** Incremental in the Skeptic's sense ("conformal + X"); most valuable as the evaluation half of D-R1a. I list it so the team can decide whether to merge.

### D-R1f (IDEA-16 candidate; analysis paper; suggest hand-off to Researcher2). What bounds localization precision in MLLMs: tokens, positions, or decoding?

**Research problem.** Three 2026 papers blame text-coordinate decoding `[VERIFIED: 2608.03471, 2605.27365, 2512.10554]`, one blames patch merging `[VERIFIED: 2603.28069]`, one blames implicit patch-to-pixel mapping/RoPE asymmetry `[VERIFIED: 2510.03230]`, and PGT blames supervision `[VERIFIED: 2605.23883]`. They cannot all be the binding constraint. There is no matched-data study on native-resolution models evaluated at high IoU and small object scale.

**Core insight.** Precision should follow a measurable "floor" set by the effective token pitch (patch merge x downsampling) unless the decoder can interpolate within a token; the different fixes should separate cleanly on a P@0.9-vs-object-size curve. A controlled factorial (coordinate format x position encoding x decoder type x patch merge) on one backbone with fixed data would produce a precision law and settle which assumption (A1/A4) is right.

**Closest papers.** ExpVG (2508.08066; LLaVA-1.5 and RefCOCO only), Hi-Token (2608.03471), RULER/I-MRoPE (2510.03230). Delta: native-resolution backbone, small-object and high-IoU evaluation, all factors in one design.

**Main risk.** Compute (many training runs); could be judged "engineering". Fits Researcher2's practical remit better than mine.

---

## 12. Claims I am least sure about (for the Verifier)

1. GroundingME: "best of 25 MLLMs 45.1%, most 0% on rejection, data mixture raises rejection to 27.9%" (Section 9; drives D-R1b/e). Source read: abstract of 2512.17495. Needs: which models were the 25 (does it include Qwen3-VL/Gemini 3?), and how "rejection" is scored.
2. Molmo2 vs Gemini 3 Pro video pointing 38.4 vs 20.0 F1 and tracking 56.2 vs 41.1 J&F (Section 1/6). Source: abstract of 2601.10611. Needs: protocol; whether Gemini was prompted comparably.
3. Faithfulness of Visual Thinking: "predictions nearly unchanged under visual intervention, change under textual intervention" (Section 3; premise of D-R1b). Source: abstract of 2510.23482. Needs: which models, effect sizes.
4. Self-Improving Small Object Grounding: "attention-only IoU regressor r>0.67; up to 19% gain" (Section 8; premise of D-R1a). Source: abstract of 2606.01612. Needs: which models, whether gains hold on non-small objects.
5. GUI-G1 size-drift claim (hit reward shrinks boxes, IoU reward inflates them) and "longer chains hurt grounding" (Section 2; supports A2/A3). Source: abstract of 2505.15810. Needs: magnitude and whether shown outside GUI.
6. Qwen3-VL coordinate system is normalized [0,1000] and its box data include Grounding-DINO/Qwen2.5-VL pseudo-labels (Section 1; supports A8/J). Source: HTML of 2511.21631 via WebFetch summary.

---

## 12b. Addendum (2026-09-01, after reading landscape/skeptic-crowded-map.md)

I read eight papers the Skeptic's map surfaced that bear on my directions. Effects on the survey and on each direction:

**New evidence that changes the survey**
- "MLLMs know where to look" (ICLR 2025) `[VERIFIED: 2502.17422]`: attention and gradient maps point at the right region even when the answer is wrong; training-free crop intervention on two models and seven VQA benchmarks; the maps are used only for post-hoc intervention, never as supervision. The "latent gap" premise is therefore a 2025 finding for VQA answers; what is new in 2026 is its appearance for *coordinate outputs* (temporal heads, InnerZoom, ACS).
- "What does RL improve for visual reasoning?" `[VERIFIED: 2602.12395]`: causal probing, parameter comparison and merging show RL's gain is a consistent inference-time shift in mid-to-late layers, transferable by merging and necessary by freezing; not a perception change. This contradicts PIVOT's encoder claim `[VERIFIED: 2510.16333]` at face value; Section 2 should read "two papers disagree on whether RL touches perception". It also supplies the hypothesis for D-R1a: the gap between internal localization and emitted coordinates lives in late layers/decoding, so RL-for-grounding may work mostly by teaching the decoder to read a map that already exists. That is a testable, explanatory claim.
- "The illusion of visual tool use" `[VERIFIED: 2608.06270]`: causal-graph audit with trajectory-level observation corruption and step-level counterfactual replacement; a Visual Evidence Gain metric; two failure modes ("calling without looking", "looking without planning"); accuracy gains concentrate in a calibrated minority of rollouts; six models, five benchmarks. This is the measurement instrument D-R1b needs and it does not propose a training signal.
- IVT self-correction mirage `[VERIFIED: 2606.13156]`: the +2.4 Acc@0.5 from iterative box refinement was an oracle artefact (best step chosen with ground truth); every label-free stopping rule is at or below step 0; self-verification confidence correlates with correctness at r about 0.22; the loop reacts to box presence, not accuracy. Direct warning for any test-time selection stage (D-R1a stage 2, D-R1e): report step-0 and label-free selection only, and treat "V-R agreement correlates with correctness far above 0.22" as a kill metric.
- VISTA `[VERIFIED: 2606.14579]`: label-free GRPO for GUI grounding from target-preserving multi-view crops with a self-verified cross-view anchor; Qwen3-VL 4B/8B/30B-A3B on ScreenSpot-Pro from 55.5/52.7/53.7 to 63.4/65.8/67.0. Two consequences: (i) closest label-free reward to D-R1a stage 3 (VISTA enforces consistency across input views; D-R1a enforces consistency between internal read-out and output; they compose); (ii) Qwen3-VL-8B is at 52.7 on ScreenSpot-Pro, so my Section 1 line "Qwen3-VL claims SOTA on ScreenSpot-Pro" holds at most for the flagship.
- ViPSy `[VERIFIED: 2606.28401]`: preference pairs from "semantically aligned image variants" with object-level visual cues; AMBER hallucination down 35.7%. Together with mDPO/POVID/OViP/P2-DPO (Skeptic item 23) this is the corrupted-image-preference family closest to D-R1b; none uses targeted edits with known masks, none produces grounding outputs, none tests faithfulness of cited regions.
- GAVEL `[VERIFIED: 2606.26923]`: joint verification, explanation and localization of caption-image misalignment with human annotations; closed models do poorly; a supervised baseline on the train split improves. D-R1d's *evaluation* partly exists; D-R1d must be positioned on the generation side (typed claim attribution and a verifier reward) and must beat SFT on GAVEL's own train split.
- Rex-Omni `[VERIFIED: 2510.12798]`: quantized 0-999 coordinate tokens, 22M SFT then GRPO with geometry-aware rewards fixing low recall, duplicates and misalignment; 3B matches or exceeds Grounding DINO zero-shot on COCO/LVIS. Mandatory baseline for D-R1f and for any multi-instance claim; my Section 9 "duplicates/omissions unstudied" is partly answered by its RL stage, though order effects remain open.

**Other Skeptic-listed work I must cite per direction** (not read; ids from the Skeptic's map): label-free rewards SSL4RL 2510.16416, SSL-R1 2604.20705, SR-MCR 2512.22545, GUICrafter 2606.29705 (D-R1a); attention regularization toward labeled regions Faithful-MR1 2605.22072, attention-guided fine-tuning 2606.01558, visual-inertia breaking 2604.01989 (D-R1a: these need GT regions, D-R1a does not); rejection GSVA 2312.10103, GREx 2601.05244, SAM 3 presence head 2511.16719 (D-R1b/e: a presence head trained on edit pairs is a natural component); attention grounding V2P 2508.13634, DeepScan 2603.03857; VPSG directional coordinate errors 2510.22102 (D-R1f); RH-AUC 2505.21523 (supports A3); data-centric reading of RL-vs-SFT 2602.10815 (must accompany 2501.17161 in Section 2); contamination 2411.03823 and Ref-L4 label errors 2406.16866 (evaluation protocol: use post-2024 image sets and a RefCOCO-free base such as Molmo, pending its data card).

**Revised standing of my directions**
- D-R1a: still my top pick, but the paper must be the triple "measure the gap against emitted coordinates across tasks and scales; explain it (late-layer/decoder, using 2602.12395's tools); close it label-free with a frozen read-out teacher and a causal term", with 2502.17422 replacing GUI-Actor in the closest-three. Evaluate on Ref-Adv, GroundingME, RSC-OOD and ScreenSpot-Pro, step-0 baselines, no oracle selection.
- D-R1b: strengthened. The Skeptic's own "what would surprise me" asks for an evidence-dependence training signal and for rejection above 70% without losing positives; D-R1b targets both. Add 2608.06270 as the audit instrument and ViPSy/OViP/mDPO as the closest family; control edits against artefact shortcuts are non-negotiable.
- D-R1c: absent from the Skeptic's map; the risk is emergent-communication drift, not grounding prior work. Keep.
- D-R1d: weakened by GAVEL and Describe Anything (2504.16072). Keep only if the verifier reward beats SFT on GAVEL's train split; otherwise fold into D-R1b as an application.
- D-R1e: merge into D-R1a as its confidence/abstention component; the r about 0.22 result and the Skeptic's AUROC-above-0.9 target define the bar.
- D-R1f: the Skeptic independently lists the format factorial as under-explored; Rex-Omni and VPSG anchor it. Recommend Researcher2 owns it.

## 12c. Addendum (2026-09-01, after Researcher2's feasibility notes in landscape/r2-practical.md Section 8)

Accepted as constraints on the Phase 2 IDEA files. Attribution: all items below are Researcher2's findings unless marked as my response.

**Corrections to the survey**
- Section 5 "best open" baselines are stale: Qwen3.5-27B reports RefCOCO avg 90.9, ScreenSpot-Pro 70.3 and RefSpatial-Bench 67.7 on its HF card, and MolmoPoint-GUI-8B reports OSWorld-G 70.0 (blog) `[LIKELY: R2, model cards/blog, not arXiv]`. These supersede GTA1/GUI-CURSOR/InnerZoom as the numbers to beat; Verifier should confirm since they come from cards, not papers.
- Qwen3.5/3.6/3.8 use Gated-DeltaNet hybrid layers `[VERIFIED by R2: HF Qwen3.5-4B card]`; most layers have no softmax attention matrix. Any claim in Section 8 about "attention read-outs" applies to all-softmax models only (Qwen3-VL-4B/8B, Molmo2-8B, InternVL3.5-8B, Gemma4-E4B). Gradient-based read-outs (entropy-gradient, 2604.08456) still apply to hybrids.

**IDEA-11 (latent gap) constraints**
- Audit on all-softmax backbones, with MolmoPoint as the non-text-decoder contrast and a Qwen3.5 hybrid as the gradient-only contrast (my response: this turns the architecture blocker into a useful axis, since the hypothesis that the gap lives in decoding predicts it should also appear in hybrids when measured by gradient read-outs).
- Attention needs eager mode with per-layer hooks; one H100 for up to 8B. Fix which token's attention is read (yes/no presence probe vs first coordinate token) and report both.
- RL stage: precompute the frozen teacher's read-out mask per prompt so the reward is a lookup; EasyR1 GRPO at 7B fits 4xH100 with no extra model copy; the causal term doubles forward passes, batch them. Use Ref-L4 Acc@0.5/0.75/0.9 for the high-IoU analysis. Budget: audit 20 GPU-h, test-time 10, RL 150 (4B) to 300 (8B). GroundingME ships vLLM evaluation code that already supports Qwen3-VL-8B-Thinking.

**IDEA-91 (edit-to-verify, R3 author; my section) constraints**
- Precompute an edit bank per image over SAM 3 instances (remove/swap each of 5 to 15 instances) and select the edit whose mask best matches the predicted box at reward time; fast inpainter only for off-bank boxes. Editors: LaMa-class inpainting for removal, Qwen-Image-Lightning (Apache-2.0) or Qwen-Image-Edit / FLUX.2 klein 4B for swaps and insertions; 100k edits about 30 to 60 GPU-h.
- SAM 3 is under a custom licence, not Apache/MIT; check redistribution before releasing derived data.
- First kill experiment is the artifact gate: an edited-vs-control ResNet-18 must stay below about 0.6 AUROC or the reward is exploitable (about 2 GPU-h). Keep COCO out of the pool (SA-1B / OpenImages / Objects365). Crop judge on its own vLLM GPU. Full study about 170 GPU-h.

**IDEA-13 (referring games) constraints**
- Alternating phases, not simultaneous self-play: speaker GRPO against a frozen listener served in vLLM inside the reward function; then listener SFT/RL on the produced expressions. Evaluate the listener on human benchmarks every round to catch co-adaptation. Small human study for speaker quality (about 500 items). About 150 GPU-h for two rounds at 4B.

**IDEA-92 (attributed description) constraints**
- Do it at 4B with a dedicated judge GPU, precomputed sufficiency and necessity on a claim subsample (about 250 GPU-h; 8B would be about 500). Judge-AUROC-vs-DetailVerifyBench kill test first (about 5 GPU-h). The P12 causal experiment (attribution reward vs RLHF-V-style text correction at equal data) is about 60 GPU-h.

**Hand-offs and cross-cutting**
- D-R1f accepted by Researcher2 as IDEA-21: 2B backbone, fixed 200k non-COCO mix, fractional factorial over coordinate format x decoder x patch merge x RoPE, with connector re-alignment budgeted per arm.
- Every RL run adds a 2 GPU-h forgetting check (MMMU / MMBench / POPE before and after), the free baselines (sampling + NMS voting, max_pixels sweep, token-likelihood confidence), and open judges only.

## 12d. Verifier spot-check applied (2026-09-01; verifications/claims-log.md V-058 to V-067, V-100)

Corrected in place above: MolmoPoint tracking gain (+5.7 J&F, not +6.3) and the GUI-variant attribution (Section 1, Section 10 F stub); GTA1 OSWorld-G 72.2 dropped as unsupported (Section 5); GPT-4o 0.9 kept as LIKELY (Section 1); GUI-G1 size-direction claim marked body-level (Section 2).

Labels upgraded to VERIFIED by the Verifier's primary-source reads (I leave the inline labels as written and record the upgrades here): Qwen2.5-VL absolute-pixel coordinates (V-002); InternVL3.5 RefCOCO avg 90.1 (14B) / 92.4 (241B-A28B) (V-078); Molmo2 38.4 vs 20.0 F1, 56.2 vs 41.1 J&F, 35.5 vs 29.6 counting (V-058); GUI-G1 90.3 / 37.1 and "longer chains hurt" (V-061); GUI-G2 +24.7% (V-062); GroundingME 1,005 items / 25 models / 45.1 / 0% / +4.5 / 27.9 (V-063; model list not in abstract); self-improving small objects r>0.67 and up to 19% (V-064; COCO/Objects365; non-small generalization not stated); faithfulness-of-visual-thinking intervention asymmetry (V-065); TreeBench 405 / o3 54.87 / +16.8 / +13.4 (V-066); Molmo 0-100 percentage coordinates (V-011); Qwen3-VL normalized [0,1000] and pseudo-labelled box data (V-006, from the PDF text).

New anchor facts from the Verifier: VLM-R1 RL vs SFT is RefCOCO val 90.55 vs 88.7 and LISA-Grounding 63.16 vs 54.82, with SFT falling below the 56.51 base (V-052), so the "RL generalizes" evidence for grounding is one OOD set with SFT *hurting*; Qwen3-VL-235B-A22B RefCOCO avg 92.1 (Thinking) / 91.9 (Instruct) with GPT-5 (high) 66.8 and Gemini 2.5 Pro 74.6 in the same table; Qwen3-VL SUN RGB-D +5.2 over Gemini is the Thinking variant (34.9 vs 29.7), Instruct is +9.7 (39.4). Name collision: "Ref-Adv" 2026 (2602.23898) is a different dataset from Akula et al. ACL 2020 Ref-Adv (2005.01655) that Researcher3 cites (V-100); I mean the 2026 one throughout.

## 13. Index of ids cited (read this session unless marked LIKELY in text)

2511.21631 Qwen3-VL; 2502.13923 Qwen2.5-VL; 2508.18265 InternVL3.5; 2601.10611 Molmo2; 2603.28069 MolmoPoint; 2506.05336 VideoMolmo; 2606.18558 MolmoMotion; 2507.07999 TreeBench; 2604.19697 StepSTEM; 2509.25794 Point-It-Out; 2503.06520 Seg-Zero; 2501.04001 Sa2VA; 2605.07141 Qwen3-VL-Seg; 2512.00395 STAMP; 2603.00152 Dr. Seg; 2501.17161 SFT-vs-RL; 2504.07615 VLM-R1; 2503.01785 Visual-RFT; 2503.06749 Vision-R1; 2505.14231 UniVG-R1; 2506.04034 Rex-Thinker; 2510.16333 PIVOT; 2505.15810 GUI-G1; 2507.15846 GUI-G2; 2512.14698 TimeLens; 2604.04917 Vero; 2604.08476 Faithful GRPO; 2607.09492 multimodal reward hacking; 2509.22047 MO-GRPO; 2511.19343 Syn-GRPO; 2505.14362 DeepEyes; 2505.15436 Chain-of-Focus; 2505.15966 Pixel Reasoner; 2509.07969 Mini-o3; 2411.16044 ZoomEye; 2508.11630 Thyme; 2506.23918 survey; 2510.23482 SCCM; 2601.11633 ViEBench; 2509.20912 DeFacto; 2604.11025 TTSP; 2606.16122 Thinking with Visual Grounding; 2602.11858 Zooming without Zooming; 2606.30084 InnerZoom; 2608.04385 ReGround; 2512.03746 programming vision; 2605.12237 UHR-Micro; 2601.06224 Ground What You See; 2605.08245 over-alignment; 2604.24396 PND; 2604.05623 DetailVerifyBench; 2506.24102 DenseWorld-1M; 2605.10893 BICR; 2605.22902 transcoders; 2311.03356 GLaMM; 2504.07981 ScreenSpot-Pro; 2506.03143 GUI-Actor; 2507.05791 GTA1; 2412.10840 TAG; 2602.06351 Trifuse; 2608.09654 regression-free matching; 2510.03230 RULER; 2602.02419 SafeGround; 2604.14113 UI-Zoomer; 2512.05941 Zoom-in Click-out; 2511.13087 MEGA-GUI; 2507.05673 R-VLM; 2509.04243 active perception GUI; 2602.09082 UI-Venus-1.5; 2503.13377 Time-R1; 2507.04702 Tempo-R0; 2510.04022 Video-in-the-Loop; 2507.18100 VTG RL recipes; 2605.21954 temporal heads; 2510.20579 Open-o3 Video; 2602.11730 STVG-R1; 2605.13803 EvoGround; 2606.00775 GIRL-DETR; 2605.26104 EVIDENT; 2506.04308 RoboRefer; 2512.16561 N3D-VLM; 2603.26690 SpatialPoint; 2511.21688 G2VLM; 2606.24539 PointVG-R; 2605.29074 Embodied3DBench; 2605.17070 EPIC-Bench; 2604.24300 ReVSI; 2603.18892 MultihopSpatial; 2510.25760 spatial survey; 2605.25802 VLA initialization; 2605.23883 PGT; 2608.03471 Hi-Token; 2605.27365 LocateAnything; 2512.10554 GETok; 2607.24407 Motto; 2508.08066 ExpVG; 2606.01612 self-improving small objects; 2604.08456 entropy-gradient; 2601.08151 contrastive attention; 2601.07645 PlaM; 2604.24036 occlusion/small; 2604.00886 PixelPrune; 2602.04304 layer-adaptive; 2512.17495 GroundingME; 2608.04698 RC-GRPO; 2506.17901 PostAlign; 2605.25706 OpenRef; 2512.06276 RefBench-PRO; 2606.28862 HKVLM; 2602.23898 Ref-Adv; 2604.02323 RSC; 2502.20104 compositional REC; 2507.18300 LMM-Det; 2501.09720 aerial MLLM detection; 2410.12332 MC-Bench; 2601.04777 GeM-VG; 2507.00748 multi-image grounding RL; 2605.18621 CrossView; 2505.03788 UQ via grounding; 2505.23224 MMBoundary; 2510.00705 uncertainty guidance; 1612.09542 speaker-listener-reinforcer; 2509.07414 language self-play.
