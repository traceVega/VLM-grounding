# IDEA-91 contributed section (Researcher1): online edit server, control-edit protocol, counterfactual-form comparison, precedents, kill experiment
Status: CONTRIBUTED SECTION for IDEA-91 (author of record: Researcher3). Revised 2026-09-02 against the Skeptic's Section 6.5 (VISE, forensic detector, three-way counterfactual arm, edit success by referent size, natural pairs).
Author: Researcher1    Last updated: 2026-09-02
Feasibility constraints from Researcher2, landscape/r2-practical.md Section 8.2, are applied throughout. Notation follows landscape/r3-history.md Section 7.

## S1. Edit operators and what each one licenses as a verifiable label

Given an image I, an expression q, and a region proposal R (the policy's own box y, or a SAM 3 instance during precompute):

| Operator (Section 7 name) | Edit | Required output on the edited image | What it tests |
|---|---|---|---|
| rem_R | inpaint the referent's SAM 3 mask | `none` | necessity; on the original, y must overlap the edited mask, else the model grounded something else |
| swap_attr on R | change the attribute named in q on the referent, keep shape | `none` if no other instance now matches; else the new match | that q's discriminating attribute is read from pixels |
| ins_R'' | paste a look-alike instance elsewhere (or duplicate the referent) | two boxes ("multiple") | exhaustiveness and multi-target output |
| move_R | cut the referent and inpaint at a new position | box at the new position | location prior vs pixels |
| ctrl_R' | same-size, same-editor edit on a non-referent region | unchanged output | invariance V; and the artifact gate (S3) |

Text-side operators T (null, shuffled, attribute-swapped expressions) apply to the same image and share the reward table.

## S2. Online edit server design

- Precompute, not per rollout (Researcher2 blocker 1). For each image in the pool (SA-1B, OpenImages; no COCO, and note Objects365 and OpenImages are in Qwen3-VL's pretraining mix per Verifier V-095, so SA-1B is the cleanest pool), run SAM 3 concept prompts over a noun list to get 5 to 15 instances with masks and categories; for each instance precompute rem, one swap_attr (attribute chosen by a captioner), one move and one ctrl, and per category one ins. Store edits and masks keyed by image and instance.
- Reward-time selection and the removal region (Skeptic 91-M2). For a rollout box y, pick the bank instance whose SAM 3 mask best matches y; if box-to-mask IoU is at least 0.5 the mask is the removal region and its variants become the counterfactual set; if several bank instances fall inside y, their union is removed. If no instance reaches 0.5, the box region itself is inpainted by a fast mask-conditioned inpainter (LaMa-class, about 50 ms per megapixel image) and the item is flagged box-inpaint; the K1 artifact gate is run separately on mask holes and rectangular holes, since rectangular holes are more detectable. The box-to-mask IoU distribution and the box-inpaint share are reported per model.
- Editors. Removal and relocation: LaMa-class inpainting; attribute swap and insertion: Qwen-Image-Edit or Qwen-Image-Lightning (Apache-2.0, 4 to 8 steps) or FLUX.2 klein 4B (Apache-2.0); 0.5 to 2 s per edit, about 30 to 60 GPU-h per 100k edits (Researcher2 8.2).
- Edit verification. A frozen open VLM, different from the reward judge, checks each edit (referent absent after rem; attribute changed after swap; new instance present after ins; nothing referent-related changed after ctrl); failures are discarded and the failure rate per operator is reported.
- Edit success by referent size (Skeptic 91-F1). Success rate is reported per referent-size bin (relative area under 1%, 1 to 5%, over 5%) and the fraction of each benchmark on which the method can train at all is stated; if success under 1% relative area is below 50%, small referents are handled by move and ins only.
- Asynchrony. The bank is built ahead of training; a small asynchronous worker (Syn-GRPO's data-server pattern, S4) tops up off-bank edits for boxes the policy proposes often.
- Licences. SAM 3 is under a custom licence, not Apache/MIT (Researcher2); check redistribution before releasing the bank; the editors above are Apache-2.0.

## S3. Control-edit protocol and the artifact gate (Skeptic 91-M1)

1. Every rem/swap/move variant is paired with a ctrl variant of matched size, shape and editor on a non-referent region; the required output on ctrl is the original box. Rewards are computed on the pair, so an "edited implies none" policy scores zero on half of it.
2. Detectability gate before any RL, with a strong detector, not a simple classifier: a fine-tuned DINOv2 detector and a forensic model (e.g. a state-of-the-art edited-image detector) trained to separate referent-edited from control-edited images on 2k held-out pairs. Researcher2's ResNet-18 at 0.6 AUROC is the floor; the bar is that the strongest detector stays below 0.6 AUROC after editor iteration. About 5 GPU-h.
3. Policy-as-detector test after training: probe whether the trained policy's hidden states separate edited from control images better than the base model's; a rise is evidence of artifact exploitation and is reported.
4. Identity-edit positives: a fraction of unedited images pass through the editor (encode-decode) so editor fingerprints appear on positives too.
5. Per-operator abstention rates on ctrl images and all detector AUROCs go in the main table.

## S4. Counterfactual-form comparison (Skeptic 91-N1; the mandatory arm)

The claim that realistic edits are better than cheap counterfactuals is empirical. Same pipeline, same compute, four counterfactual forms as the necessity input: (i) gray mask on the region (DeFacto `[VERIFIED: 2509.20912]`), (ii) attention mask over the region's visual tokens (VIGIL `[VERIFIED per Skeptic: 2606.26387]`), (iii) corrupted image with a KL term (PAPO `[VERIFIED per Skeptic: 2507.06448]`), (iv) realistic edit from the bank. Outcomes: GroundingME rejection and positives, OpenRef N3R, Ref-Adv 2026, natural-pair necessity (S5), artifact-detector AUROC, forgetting check. If (i) to (iii) match (iv), the training half is dropped and the paper is the audit and the interventional score's predictive validity, as the Skeptic's 6.5 states.

## S5. Precedents and how this section differs

- VISE `[VERIFIED per Skeptic: 2606.27373]`: trains Qwen3-VL-2B label-free with a geometric-invariance reward and a semantic-invariance reward that alters the model's own predicted regions and requires recognition of the missing information (+16.9 CIDEr, minus 5.0 CHAIR-I). Prediction-anchored and invariance-based; outputs are captions, not boxes. First in the closest-three; must be beaten head to head on localization outputs with the same base.
- DeFacto `[VERIFIED: 2509.20912]`, VIGIL, PAPO: the cheap counterfactual forms in S4.
- Syn-GRPO `[VERIFIED: 2511.19343]`: decoupled asynchronous image-generation server inside GRPO with a diversity reward; we borrow the server pattern; we differ in that the edit itself is the label and no task label is needed.
- HalluSegBench / RobustSeg `[VERIFIED per Skeptic: 2506.21546]`: edited counterfactuals of ground-truth referents, segment-or-abstain, SFT; we anchor on the prediction, add control edits and RL.
- ViPSy (2606.28401), OViP (2505.15963), P2-DPO (2606.03376), Paint Outside the Box (2412.00684): the preference-pair and augmentation family; related, not nearest.
- Natural pairs (S6) follow VisMin-style verified pairs `[LIKELY: 2407.16772 per R3]`.

## S6. Natural-pair evaluation (Skeptic 91-E1)

At least one evaluation with no editing artifacts: real pairs (I_1, I_2) with the same q, referent present in I_1 and absent or different in I_2, from consecutive video frames (Molmo2 tracking data), multi-instance sets and VisMin-style pairs. Necessity N is reported on natural pairs and on edited pairs; the discrepancy is the artifact effect, shown in every table.

## S7. Kill experiment (numeric stop rules)

Build 2k rem plus ctrl pairs on GroundingME and OpenRef images with SAM 3 masks and the editors above. Evaluated before any training:
1. Detectability: strongest detector (DINOv2 fine-tune or forensic model) AUROC above 0.6 after one editor iteration: stop until fixed.
2. Discriminativeness, on human-verified clean removals only (Skeptic 91-E1): K2 records judge-verified removal success per operator and a 200-item human check of removal quality, and the emits-a-box-on-removal rate is computed only on removals the human check passes, with responses attributable to editor remnants reported as a separate row. If Qwen3-VL-8B and Molmo2-8B already output `none` on more than 60% of clean rem images under the native prompt, negatives are too easy; if they hallucinate a box on more than 80% of clean rem images and also abstain on more than 30% of ctrl images, the edits confuse rather than inform: stop and fix editors.
3. Consistency headroom: on originals where the box is correct, the fraction of rem images on which the same box is re-predicted must exceed 20% for the faithfulness half to have headroom; below that, keep only the rejection half.
4. Edit success by size: if success under 1% relative area is below 50%, small referents are excluded from training and this is stated.
Cost about 12 GPU-h (Researcher2 8.2 plus the detector).

## S8. Compute note

Edit bank over 100k images about 50 GPU-h; GRPO at 4B about 100 GPU-h; the four-form comparison in S4 about 120 GPU-h beyond arm (iv); evaluation about 20 GPU-h; kill experiment about 12 GPU-h; total about 300 GPU-h, upper tier B.

## Evidence labels
[VERIFIED]: 2511.19343, 2509.20912, 2606.28401 (abstracts, by me); 2606.27373, 2606.26387, 2507.06448, 2506.21546 (per Skeptic). [LIKELY]: editor names, speeds and licences (Researcher2's search snippets); SAM 3 licence terms; OViP, P2-DPO, VisMin characterizations. [SPECULATION]: the 0.6 AUROC bar is a convention, not a derived bound; the size bins.
