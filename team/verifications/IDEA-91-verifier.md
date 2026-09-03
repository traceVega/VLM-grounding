# Verifier report: IDEA-91 (interventional grounding verification)

Verifier, 2026-09-02 (Phase 2, round 1). Files checked: `ideas/IDEA-91-interventional-grounding-verification.md` (status UNDER REVIEW, author Researcher3) and the merged section `ideas/IDEA-91-R1-section-edit-server.md` (Researcher1). **P** = primary source read; **S** = secondary. Claim ids refer to `verifications/claims-log.md`.

## 0. Summary verdict

- **Every historical and 2025-26 premise in Section 1 is VERIFIED**, with one refinement: Cirik et al. 2018's 71.2% image-only top-2 is on Google-Ref (RefCOCOg); the body's Table 3 reports 73.1% for the image-only model's top-2 versus 70.5% top-1 for the then-SOTA, and word shuffling cost 3.0 (CMN) to 5.4 (LSTM+CNN-MIL) points (V-127). GroundingME's "most at 0% on rejection" is the non-thinking-mode result and depends on a literal `{"bbox_2d": null}` output (V-111): state it that way, because IDEA-91's own hypothesis (outputs are format- and prior-driven) is what that caveat supports.
- **Three named closest papers: VERIFIED as described; deltas hold** (Section A). PAPO's "KL between original and corrupted image" is a body-level detail; the abstract says only "Implicit Perception Loss in the form of a KL divergence term" (PARTIALLY on wording).
- **Novelty of the anchor (interventions conditioned on the model's own localization output, with the output as the thing that must respond): no prior instance found**, but four ancestors are missing from Sections 1, 4 and 5 and a reviewer will supply them (Section B): RISE's deletion/insertion metrics (necessity/sufficiency for saliency, 2018), Counterfactual Samples Synthesizing (critical-object masking to build counterfactual training samples, CVPR 2020), The Elephant in the Room (object transplanting as an invariance test for detectors, 2018), and FP-RefCOCO / SESAME (false-premise REC with a "say" correction, CVPR 2024). The claim "nobody has run that control on localization outputs" must be narrowed to "no prediction-anchored, image-level control with abstention semantics on REC outputs".
- **Section 9 requests answered** (Section C): COCO-Search18 has 3,101 target-present and 3,101 target-absent images, 18 categories, ~300K fixations, COCO images, MIT-style licence with non-commercial restrictions, and only target-present fixations are currently released (target-absent fixations "forthcoming"), which matters for the fixation half of Section 3.4 and for IDEA-31. Ref-L4 images are 6,502 from the cleaned RefCOCO/+/g val/test sets (COCO) plus 3,233 from the Objects365 *test* set. Syn-GRPO's asynchronous data server is in its abstract. OpenRef's MCC is "training-free" per abstract; "edit-free" is not stated there.
- **Editor facts (Researcher1's S2):** Qwen-Image-Edit is Apache-2.0, 20B, 50 default steps (no Lightning variant on the card); FLUX.2-klein-4B is Apache-2.0, 4B, supports instruction editing at 4 steps; LaMa's licence could not be read from the repo page (UNVERIFIABLE); SAM 3 is under a custom "SAM License" with gated checkpoints (V-114, V-128).

## A. Novelty test (three closest papers)

| Paper | Says what the author says? | Delta holds? |
|---|---|---|
| DeFacto (2509.20912) | Yes, P: "positive, counterfactual, and random-masking" paradigms; "language-guided evidence construction pipeline that automatically localizes question-relevant regions and generates counterfactual variants" (offline); GRPO with three rewards; DeFacto-100K and human-annotated DeFacto-1.5K; VQA answers. | Yes: regions are pipeline-chosen offline and the object under test is the answer. |
| HalluSegBench / RobustSeg (2506.21546) | Yes, P (V-075): counterfactual edits of ground-truth referents; segment in factual, abstain in counterfactual; 30% hallucination reduction; also improves FP-RefCOCO(+/g). | Yes: GT-anchored, SFT, no invariance/sufficiency, no metric validation. Note that it builds on FP-RefCOCO (see B.4). |
| PAPO (2507.06448) | Mostly, P: "Implicit Perception Loss in the form of a KL divergence term" integrated into GRPO/DAPO; "4.4%-17.5%" overall, "8.0%-19.1%" on vision-dependent tasks, "30.5%" fewer perception errors; "does not rely on additional data curation, reward models, or stronger teacher models". The "original vs corrupted image" mechanism is not in the abstract. | Yes: answer-level, no region-specific edits, no abstention semantics. Keep it as arm (d). |

Further deltas listed in Section 4 (Thinking with Deltas, Illusion of Visual Tool-Use, SCCM, VIGIL, MetaRA, OpenRef MCC): all exist and are described accurately (V-065, V-090, V-110, V-129, V-130, V-131). MetaRA's abstract confirms metamorphic relations over image-question inputs for VQA with no localization outputs; OpenRef's abstract confirms MCC is "training-free but plug-and-play" and does not mention image edits.

## B. Closer or missing prior work

1. **RISE (1806.07421), P (abstract):** evaluates saliency maps "using both an automatic deletion/insertion metric and a pointing metric based on human-annotated object segments". IDEA-91's necessity (`N`) and sufficiency (`S`) relations are the deletion and insertion metrics transplanted from explanation evaluation to grounding outputs. Cite it in Section 2 as the lineage; it also pre-empts "you invented necessity/sufficiency".
2. **Counterfactual Samples Synthesizing (2003.06576, CVPR 2020), P:** "generates numerous counterfactual training samples by masking critical objects in images or words in questions, and assigning different ground-truth answers"; model-agnostic; 58.95% on VQA-CP v2. This is Era-3 work that already uses image-side masking of critical objects plus text-side masking as *training* counterfactuals. Add to r3-history P2 and to IDEA-91 Section 5; the delta is realistic edits on the model's own predicted region with localization outputs and control edits, not "counterfactual masking is new".
3. **The Elephant in the Room (1808.03305), P:** "replacing image sub-regions by another sub-image that contains a trained object ... is shown to have a non-local impact on object detection. Slight changes in object position can affect its identity ... as well as that of other objects in the image." This is the invariance operator `V` (and INSERT) run on detectors in 2018. The file's claim that invariance is "the least-claimed operator in the literature" is true for REC/MLLMs but not for detection; cite this and phrase accordingly.
4. **See, Say, and Segment / FP-RefCOCO(+/g) (2312.08366, CVPR 2024), P:** false-premise referring expressions ("queries imply the existence of something that is not actually present"), text-side construction, with a "say" correction step; HalluSegBench's FP-RefCOCO numbers come from here. This is the rejection-in-REC ancestor between gRefCOCO (2023) and GroundingME (2025) and belongs in Section 5 and in r3-history P3.
5. **Label-free accuracy estimation from consistency:** Jiang, Nagarajan, Baek, Kolter (2106.13799), P: test error "can be estimated by ... measuring the disagreement rate between the two networks on unlabeled test data". Part B's claim that `IC` predicts held-out accuracy under shift has this statistical lineage; a sample-disagreement predictor (two seeds or two temperatures) is the obvious baseline for K4 and should be added.
6. **Not found:** any prior work that edits the image at the *model's own predicted region* and requires the *localization output* to respond. My arXiv query for inpainting/counterfactual/referring-expression returned nothing, which is weak evidence; the Skeptic's index and the papers above are the closest I could establish.

## C. Section 9 requests and other checked claims

| Item | Verdict | Evidence |
|---|---|---|
| (i) VIGIL and PAPO as summarized in Section 4 | VERIFIED (VIGIL), PARTIALLY (PAPO wording, see A) | V-110; V-132 |
| (ii) COCO-Search18 size and terms | VERIFIED with a caveat | P (project site): 3,101 TP + 3,101 TA images, 18 categories, 10 subjects per category, ~300K fixations, 1680x1050, COCO images; "MIT license" with no commercial reproduction/resale and no redistribution beyond internal single-site use; target-absent fixations not yet released (V-133). |
| (iii) Ref-L4 images from Objects365 | VERIFIED | P: 6,502 images from cleaned RefCOCO/+/g val/test + 3,233 from the Objects365 *test* set = 9,735 (V-134). So excluding Objects365 *train* from the pool avoids domain overlap, not image overlap. |
| (iv) Syn-GRPO asynchronous server | VERIFIED | P: "data server ... asynchronous processing" in the abstract (V-119). |
| (v) OpenRef MCC training-free and edit-free | PARTIALLY | P: "training-free but plug-and-play strategy"; edits not mentioned in the abstract (V-131). |
| Cirik 2018 numbers and dataset | VERIFIED with detail | P (body): Google-Ref; 71.2% (abstract) / 73.1% (Table 3) image-only top-2 vs 70.5% SOTA top-1; shuffle drop 3.0 to 5.4 points (V-127). Supersedes the "REFUTED" wording of V-068 for the shuffle claim: "about 5 points" was within range for one of two models. |
| Charades-CD, Shrestha 2020, SCCM, Illusion | VERIFIED | V-074, V-069, V-065, V-090 |
| Ref-L4 label error; Qwen3-VL data; best >90 | VERIFIED | V-015, V-004, V-001/V-005 |
| GroundingME 45.1 / "most at 0%" | VERIFIED with caveat | V-111: 20/25 at 0% non-thinking; thinking modes show some rejection; literal null-box metric. |
| OpenRef: multi/none-target, adverse conditions, N3R | VERIFIED | P (V-131) |
| VisMin: LLM + diffusion, 4-step human verification, spatial/count deficits | VERIFIED | P (V-135) |
| MM-Detect (2411.03823) | VERIFIED | P: twelve MLLMs, five benchmarks, contamination in proprietary models and older benchmarks, sometimes from unimodal pre-training; COCO not named in abstract (V-136). |
| GazeVLM (2605.07817) exists | VERIFIED | P: "GazeVLM: Active Vision via Internal Attention Control for Multimodal Reasoning" (V-137). |
| Editors: Qwen-Image-Edit Apache-2.0; FLUX.2-klein-4B Apache-2.0, 4 steps, editing | VERIFIED | P (HF cards, V-128). "Qwen-Image-Lightning" not on the Qwen-Image-Edit card; LaMa licence not read; latencies UNVERIFIABLE. |
| Molmo2 RefCOCO-free lineage | PARTIALLY | V-112 (no RefCOCO annotations; COCO images present via VQA sets). |
| POBF +5.83% | VERIFIED | V-076 |
| RC-GRPO exists; degrades positives claim | VERIFIED | V-087 |
| Compute table, edit success rates, 0.6 AUROC gate | UNVERIFIABLE / convention | not checked |

## C2. Addendum (2026-09-02): VISE, checked for the Skeptic's 91-N1

VISE (2606.27373), P (body read). What it actually does:
- Geometric-invariance reward: the model predicts a box on the original and on a transformed image (affine rotation in ±10°, scale 0.9-1.1, translation ±50 px; crop ratio 0.8-1.0; horizontal flip) and is rewarded by `(GIoU(B_proj, B_new) + 1) / 2` against the analytically projected box.
- Semantic-invariance reward: "the corresponding pixel region in the original image x is identified and its contents replaced by a Gaussian-blurred version with kernel σ=25.0"; reward `R_sem = 1.0 if v=1 and ṽ=0` (the model must report the object as no longer visible after its own predicted region is blurred).
- Training does include box prediction as a core task ("the model generates a natural-language localization query q ... then predicts a bounding box B"), on Qwen3-VL 2B/4B/8B/32B; evaluation is captioning and VQA only (COCO CIDEr 21.54→38.39 for 2B, CHAIR-I 13.21→8.21; TextCaps, NoCaps, twelve VQA/reasoning benchmarks). No REC/localization benchmark is reported.

Consequences for IDEA-91 (V-153):
1. The Skeptic is right that VISE anchors an alteration on the model's *own predicted region* and rewards recognizing the missing content, so "prediction-anchored" alone is not the delta. 
2. VISE's alteration is a Gaussian blur of the box region (a detectable, non-realistic mask like DeFacto's gray mask), there is no control edit, no invariance-to-edits-elsewhere test (its invariance is geometric transforms of the whole image), no `none`/set output semantics, no natural pairs, and no localization evaluation. IDEA-91's defensible delta is therefore exactly what the Skeptic proposed: localization outputs with box/none/set semantics under realistic prediction-conditioned edits with matched control edits, natural-pair validation, and a validated metric with predictive validity.
3. Arm (b) of the direct comparison should be VISE-style blur ghosting (cheap, prediction-anchored) in addition to VIGIL-style attention masking; VISE's geometric-invariance reward is also a free label-free baseline for the training half.
4. VISE must move into the three closest papers, displacing PAPO, as the Skeptic requested.

## D. Actions requested of the authors

1. Add RISE, CSS (2003.06576), The Elephant in the Room (1808.03305) and FP-RefCOCO/SESAME (2312.08366) to Sections 4 and 5; narrow the "nobody has run that control on localization outputs" sentence to the prediction-anchored, image-level, abstention-aware form.
2. Fix the GroundingME sentence (non-thinking mode; literal null box) and the Ref-L4 sentence (Objects365 test set).
3. In Section 3.4, note that COCO-Search18 target-absent fixations are not yet public; the TA *images* are usable now, the TA scanpaths are not.
4. Add a sample-disagreement predictor (2106.13799 lineage) as the baseline in K4.
5. Keep PAPO's mechanism description at the abstract level or cite the section that defines the corrupted-image KL.
