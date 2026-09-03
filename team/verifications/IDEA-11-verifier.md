# Verifier report: IDEA-11 (latent grounding gap)

Verifier, 2026-09-01 (Phase 2, round 1). File checked: `ideas/IDEA-11-latent-grounding-gap.md` (status UNDER REVIEW, author Researcher1). Evidence marks: **P** = I read the primary source (abstract page, HTML body, or official repo/blog); **S** = secondary. Claim ids refer to `verifications/claims-log.md`.

## 0. Summary verdict

- **Premises (six "latent gap" results): VERIFIED.** All six papers exist and say what the file says (V-050, V-054, V-064, and rows V-101 to V-105 below). None of the five 2025-26 papers uses the internal read-out as a *training* reward: 2605.21954 is inference-only; 2606.30084 (InnerZoom) is trained with GUI click labels (SFT + GRPO with an inside-box reward) and reads intermediate layers only at inference to propose regions; 2606.01612 is test-time candidate selection with no LVLM update; 2506.03143 trains a supervised head; 2604.08456 is training-free.
- **Novelty of Stage 4 (label-free reward from the model's own read-out): PARTIALLY.** The *concept* of an attention-derived, annotation-free RL reward already exists in at least five 2025-26 papers aimed at reasoning/VQA (rows V-106 to V-110). What remains unclaimed, as far as I could find, is (i) applying it to *emitted coordinates* (agreement between verbalized box and read-out), (ii) the necessity/sufficiency terms on the box, (iii) the frozen-teacher construction, and (iv) the cross-task gap audit and the probe-vs-GRPO explanation. The file must cite these papers and rewrite the delta accordingly, otherwise a reviewer will type "attention reward" into arXiv and find them.
- **GroundingME rejection premise: VERIFIED with an important caveat.** 20 of 25 models score exactly 0% on Rejection *in non-thinking mode*; the paper states that with thinking enabled "all models successfully demonstrate some level of rejection behavior". The scoring requires the literal output `{"bbox_2d": null}`. GPT-5, Claude Sonnet 4.5 and Grok-4 were excluded for producing "substantial displacement and distortion". So "0%" is partly an output-format/mode effect, which is consistent with the idea's decoding-artefact hypothesis but weakens "most models cannot reject" as a capability claim (V-111).
- **"RefCOCO-free lineage" (Molmo2): PARTIALLY.** Molmo2's image-pointing sets are PixMo-Points, PixMo-Count, CoSyn-Point and Molmo2-MultiImagePoint (no RefCOCO), but its 32 image-QA sets are listed only by citation, and Molmo 1's academic mix includes VQA v2, A-OKVQA and TallyQA, which are COCO images. Say "no RefCOCO expressions or boxes", never "no COCO images" (V-112).
- **RLVR-grounding forgetting (lead's question): the strong claim "RLVR grounding does not forget" is UNVERIFIABLE for grounding and REFUTED for RLVR in general** (Section D). Keeping the general-capability regression table in Section 6 is correct and should be mandatory.

## A. Novelty test (three closest papers)

| Paper named by author | Says what the author says? | Does the claimed delta hold? |
|---|---|---|
| MLLMs Know Where to Look (2502.17422, ICLR 2025) | Yes, P: "consistently know where to look, even when they provide the wrong answer"; training-free crops from attention/gradient maps; two MLLMs, seven VQA benchmarks. | Yes. It never compares the read-out with emitted coordinates, never trains on it, and targets VQA accuracy, not localization output. |
| Self-Improving Small Object Grounding (2606.01612) | Yes, P (body read): Qwen2.5-VL-7B and InternVL-3.5-8B; IoU regressor on attention maps, Pearson r>0.67; ACS-Learned / ACS-Free; "requires no modification, fine-tuning, or auxiliary supervision to LVLMs"; gains on COCO small objects +6.35 (Qwen) and +19.35 (InternVL); on >5% objects Qwen is already at 87% with no headroom while InternVL still gains 64→70. | Yes. Test-time selection among sampled boxes; no RL, no teacher, no abstention. Correction: the file's "small objects only" understates their medium/large result for InternVL; cite both. |
| MLLMs Know When Before Speaking (2605.21954) | Yes, P: temporal-grounding heads concentrate on the GT interval during prefill, decoding drifts, read-then-regenerate gives up to +3.5 mIoU on MiMo-VL-7B, Qwen3-VL-8B, TimeLens-8B, training-free. | Yes. Inference-only, VTG only. |

## B. Closer prior work the author missed (add to Section 5 and to the delta)

1. **Attention-derived, annotation-free RL rewards already exist (for reasoning):**
   - Reward Design for Physical Reasoning in VLMs (2604.13993), P: "a novel internal reward derived from model attention weights over input image regions", "requires no spatial annotations", Granite Vision 3.3-2B, PhyX spatial-relation accuracy 0.27→0.50.
   - SAYO, Do MLLMs Really See It (2602.08241), P: "region-level visual attention-based reward" inside RL; label requirement not stated in the abstract.
   - LASER (2607.01707), P: post-training with a "Visual Grounding Reward" that regulates the attention trajectory during reasoning.
   - OmniDrive-R1 (2512.14044), P: "annotation-free, process-based grounding reward" enforcing consistency between visual focus and textual reasoning.
   - VIGIL (2606.26387), P: RL post-training that penalizes "blind confidence" under an attention-masked counterfactual; claims "emergent spatial grounding capabilities without explicit bounding box supervision".
   - Also related: SRPO (2605.07274, credit from visual dependency under corrupted inputs, close to R_caus) and Credit Where It Is Due (2602.11455, attention-topology anchors in RLVR credit assignment).
   None of these makes the *coordinate output* the policy action or scores the emitted box against the read-out, so IDEA-11's Stage 4 is not killed, but the sentence "a label-free reward built from the model's own frozen read-out" is no longer a first. Rewrite the delta as: first read-out-vs-coordinate agreement reward with causal necessity/sufficiency on the emitted box, evaluated on localization benchmarks.
2. **Read-out-guided decoding is partially anticipated on GUI:** InnerZoom (2606.30084), P (body): "Given the hidden states before decoder layer ℓ, we extract the instruction-token states and decoder-side visual-token states" to propose the target region at inference, on Qwen3-VL-2B/4B trained on OS-Atlas, OmniAct, AndroidControl, AMEX, AgentNet (283K SFT + 100K RL samples). Stage 3's "read-out-constrained decoding" should cite this as the labelled GUI precursor.
3. **Teacher-student transfer of regional perception:** Vision-OPD (2605.18740, crop-conditioned teacher, full-image student) and Zooming without Zooming (2602.11858). Different signal (privileged crops, not internal read-outs) but the same "frozen teacher supervises the single-pass policy" pattern.
4. **Attention-supervised training with labels** (contrast, already in the file): Faithful-MR1 (2605.22072), P: supervises a `<Focus>` token's attention "directly against image regions" and then rewards "trajectories that concentrate visual attention where vision causally matters"; GUI-AIMA (2511.00810), P: supervised alignment of intrinsic attention with patch-wise grounding signals, ScreenSpot-Pro 61.5 with 509k samples. Att-CoT (2606.01558), P: an objective that "encourages CoT trajectories to delay answer commitment while maintaining sustained visual-token access"; the abstract does not describe labelled attention targets, so the file's grouping of it with "regularize attention toward labeled regions" is PARTIALLY right.
5. **Stage 2 probes:** I could not find a prior layer-wise linear-probe study of box coordinates in MLLM hidden states via arXiv search (no results for the query); UNVERIFIABLE that it is new, but no killer found. 2602.12395 (P) supports the mid-to-late-layer hypothesis.

## C. Claims in Section 9 and numbers used

| Claim | Verdict | Evidence |
|---|---|---|
| RLCR: binary correctness + Brier term; calibration improves with no accuracy loss; ordinary RL hurts calibration (2507.16806) | VERIFIED | P, V-113. Transfer to boxes is untested (the file labels it correctly). |
| Faithful-MR1 / Att-CoT / visual-inertia descriptions | PARTIALLY | see B.4; visual-inertia (2604.01989) not read. |
| Molmo2 licence | PARTIALLY | paper page is CC BY 4.0 (arXiv licence); Ai2 blog says Apache-2.0 with third-party dataset caveats (per Researcher2's read, S for me). |
| Molmo2 RefCOCO-free | PARTIALLY | V-112: no RefCOCO in the named pointing sets; QA sets not enumerated; COCO images certainly present via VQA-style sets in the Molmo lineage. |
| GroundingME's 25 include Qwen3-VL | VERIFIED | P: Qwen3-VL 2B/4B/8B/32B/30B-A3B/235B-A22B are in the list; Gemini 2.5 Pro/Flash and Seed-1.6-Vision are the only commercial entries; no Gemini 3, GPT-5 excluded (V-111). |
| GUI-G1 size direction | PARTIALLY | V-061. |
| r about 0.22 (2606.13156) | VERIFIED | V-089. |
| +3.5 mIoU (2605.21954) | VERIFIED | V-101. |
| OSWorld-G 64.7 at 4B (2606.30084) | VERIFIED | V-102. |
| r>0.67, up to 19% (2606.01612) | VERIFIED, with nuance | V-064/V-103: 19% is InternVL-3.5-8B on COCO small objects; Qwen2.5-VL-7B gains 6.35. |
| GUI-Actor 100M head "matches coordinate-generating SOTA on ScreenSpot-Pro" | VERIFIED as of mid-2025 (44.6 vs UI-TARS-72B 38.1) | P, V-104. Dated: 2026 numbers are 61 to 83; say "matched the 2025 SOTA". |
| Qwen3.5 hybrid layers lack attention matrices | VERIFIED (architecture) | P: card states "Gated Delta Networks combined with sparse Mixture-of-Experts"; which layers keep softmax attention is not stated on the card (UNVERIFIABLE detail). |
| GroundingME ships vLLM eval code for Qwen3-VL-8B-Thinking | UNVERIFIABLE | not checked. |
| Compute tiers (220 to 370 GPU-h) | UNVERIFIABLE | Researcher2's estimates; not checked. |

## D. Lead's question: does RLVR grounding training forget?

What I found (all P unless marked):
- Grounding-RL papers do **not** report general-capability tables: VLM-R1 (2504.07615) evaluates only RefCOCO/+/g, LISA-Grounding, COCO and OVDEval; Visual-RFT (2503.01785) only its perception tasks; Perception-R1 (2504.07954) abstract silent. So "RLVR grounding does not forget" is **UNVERIFIABLE** for grounding specifically.
- Evidence that RL forgets less than SFT: RL's Razor (2509.04259): "despite similar performance at a new task, RL preserves prior knowledge and capabilities significantly better"; forgetting tracks the KL to the base policy; tested on LLMs and robotic foundation models, not VLMs. S-GRPO (2604.16557) abstract: SFT "often induc[es] catastrophic forgetting of general multimodal capabilities". The data-centric rebuttal (2602.10815): RL's OOD advantage comes from implicit filtering to medium-difficulty samples; difficulty-curated SFT matches or beats RL.
- Evidence that RLVR itself regresses capabilities in VLMs: Beyond Reasoning Gains (2510.21978): RLVR "introduces a significant risk of capability regression", including "perception and faithfulness", on Qwen2.5-VL-3B/7B, fixed by replay (RECAP); ViSurf (2510.10606): sequential SFT→RLVR "suffers from catastrophic forgetting"; RL Forgets! (2607.04364): continual RL post-training on Qwen3-VL-8B shows "severe catastrophic forgetting"; More Thought, Less Accuracy (2509.25848): reasoning RL "may gradually impair perceptual grounding" ("visual forgetting").
- Verdict: the strong claim is **REFUTED** (RLVR can and does regress perception/faithfulness); the comparative claim "RL forgets less than SFT at matched new-task accuracy" is **PARTIALLY** supported (LLM/robotics evidence, contested by 2602.10815); nobody has measured it for IoU-reward grounding RL, which makes IDEA-11's planned before/after table on MMMU, MMBench and POPE a genuine (small) contribution rather than a formality.

## D2. Addendum (2026-09-02): the Skeptic's 11-N1 precedents, checked

- **SD-RPN (2509.16944), P (abstract):** transforms "noisy attention maps from the MLLM's middle layers into high-quality pseudo-RoI labels by explicitly denoising the signal and resolving ambiguity", trains a lightweight RPN "predicting the RoI in a single forward pass using features from the MLLM's middle layers", no human labels, "over a 10% absolute accuracy improvement on unseen benchmarks, including TextVQA, DocVQA, and V-Star" from ~10K QA pairs. This is the closest existing instance of "the model's own read-out becomes a label-free training target for a localizer". It differs from Stage 4 in target (an RoI/crop for VQA, not REC coordinates) and in mechanism (pseudo-labels for a new head, not a reward on the text decoder), but the Skeptic's question stands: the file must say why training the text decoder toward the read-out beats SD-RPN's head, given the idea's own hypothesis that the gap lives in decoding. Verdict on the Stage-4 novelty claim: **PARTIALLY** (V-149).
- **Reinforced Attention Learning (2602.04884), P (abstract):** "directly optimizes internal attention distributions rather than output token sequences" with policy gradient, plus on-policy attention distillation; image and video benchmarks; label requirements not stated. Related to Stage 4's read-out-as-teacher (it distills attention behaviour) but not to coordinate outputs (V-150).
- **GUI uncertainty benchmark (2606.25760), P (abstract):** 27 open-weight UQ methods across 4 VLM agents and 4 GUI datasets (plus 8 for closed vendors); hidden-state and density methods most stable; rankings stable within a model (Spearman up to 0.969) but not across model classes; calibrated conformal disks shrink 40-60% with coverage loss under mismatch. Confirms the Skeptic's 11-E1: Stage 5 must include these families and report their metrics (V-151).
- **Ref-L4 sources (Skeptic 11-E2):** 6,502 images from cleaned RefCOCO/+/g val/test (COCO) + 3,233 from the Objects365 *test* set (V-134). So Objects365-train pools do not share images with Ref-L4, but they share the domain, and the COCO half is inside every Qwen/InternVL mixture.

## E. Actions requested of the author

1. Add the papers in B.1 to Section 5 and rewrite the Section 4 delta as stated above.
2. Cite InnerZoom's inference-time intermediate-layer proposal in Stage 3.
3. Replace "most of 25 MLLMs score 0% on rejecting" with "20 of 25 score 0% in non-thinking mode; thinking modes show some rejection; the metric requires a literal null box" and use this as evidence for the decoding-artefact hypothesis.
4. Change "RefCOCO-free lineage" to "no RefCOCO annotations in the documented mixture (COCO images present)".
5. Date the GUI-Actor claim.
