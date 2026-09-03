# Verifier report: IDEA-21 (what bounds localization precision: token pitch, coordinate format, or decoding)

Verifier, 2026-09-02 (Phase 2, after round 1). File checked: `ideas/IDEA-21-precision-bounds-factorial.md` (status REVISED, author Researcher2). **P** = primary source read (arXiv abstract or HTML body, HF config or dataset card, official terms page, saved report text); **S** = secondary. Claim ids refer to `verifications/claims-log.md` (V-157 to V-159, V-166 to V-179, V-192).

## 0. Summary verdict

- **Section 1 premises: VERIFIED.** The five "bottleneck" papers exist and say what the file says (Hi-Token, LocateAnything, PaDT, VPSG, RULER/I-MRoPE, PGT, UHR-Micro: V-157). MolmoPoint's "too coarse-grained" is a verbatim quote about pooled 28x28-px patches, and the paper claims about 4.7-px precision from its three-token scheme (V-175). Qwen's coordinate-convention change is justified in the Qwen3-VL report by robustness and post-processing only, and the report's ablation section covers the ViT and DeepStack, not coordinates: "rationale but no ablation" is VERIFIED (V-169). ScreenSpot-Pro's "0.07% of the screenshot area" and 1,581 items are in the paper body (V-173); Ref-L4's 45,341 annotations and 30 to 3,767 px scale range are in its abstract (V-174).
- **ExpVG (the design-space premise): VERIFIED for direction, PARTIALLY for magnitude.** The body states integer coordinates "significantly" beat location tokens and "slightly" beat decimals; per-factor point differences live in figures the fetch could not read (V-158). Two consequences: "several points" stays PARTIALLY, and the reason given in §3.2 for dropping the 0-1 float level ("ExpVG already showed they lose to integers") rests on a difference ExpVG itself calls slight (V-192).
- **Researcher2's three requests: (1) VERIFIED, (2) VERIFIED, (3) PARTIALLY** (Section C).
- **Novelty: three closest papers VERIFIED as described; deltas hold; no matched-data factorial found (NOT CONTRADICTED, V-179).** Two precedents are missing and a reviewer will name the first: Shikra's §6.2 "Location tokens or just numbers?" (2023) is the earliest controlled text-vs-token comparison and already reports numerals ahead of bin tokens by 0.4 to 3.1 points on RefCOCO-family splits (V-177); "Exploring Perceptual Limitation of MLLMs" (2402.07384) is the earliest controlled small-object study and names object location among the causes (V-178).
- **Section 9 [LIKELY] items: all resolved** (Section D). No REFUTED item. PARTIALLY: the Objects365 licence wording (terms say "academic purpose only" with CC BY 4.0 annotations, not a formal non-commercial licence, V-171) and the two ExpVG points above.

## A. Novelty test (three closest papers)

| Paper named by author | Says what the author says? | Does the stated delta hold? |
|---|---|---|
| ExpVG (2508.08066) | Yes, P (body): CLIP-ViT-L-336px + Vicuna-7B-v1.5 (LLaVA-1.5); SFT design space; integer format "significantly surpassing the location token format and slightly outperforming the decimal format"; combined design +5.6 / +6.9 / +7.0 over LLaVA-1.5 on RefCOCO val/testA/testB (V-158). | Yes. Fixed 336-px input, Acc@0.5 on RefCOCO, no decoder, pitch or position-encoding factors, no probe. The delta is accurate. |
| LocateAnything (2605.27365) | Yes, P (abstract): parallel box decoding as atomic units; 138M samples; throughput and high-IoU quality claims (V-157). | Yes. Its own data and one decoder family; no pitch or RoPE factor, no probe. |
| Hi-Token (2608.03471) | Yes, P (abstract): axis-specific hundreds/tens/ones tokens; gains across the IoU range on three backbones; Hi-GAR reward (V-157). Search snippet adds: Qwen2.5-VL-3B-Instruct base, SFT on 80k RefCOCO samples (S). | Yes. One format on existing backbones with RefCOCO training data; no decoder or pitch factor. |

## B. Closer or missing prior work

1. **Shikra (2306.15195), §6.2, P:** the text-vs-token question of factor F1 was first tested in 2023 on Shikra-7B: numerals 81.47 vs bin tokens 81.03 (RefCOCO val), 63.08 vs 59.95 (RefCOCO+ testB), 75.69 vs 72.81 (RefCOCOg val-u); the paper notes numerals cost more tokens per box (V-177). Add to §5 as the earliest controlled comparison; the delta is native resolution, high-IoU size-stratified evaluation, matched vocabulary size, and the decoder and pitch crosses.
2. **Exploring Perceptual Limitation of MLLMs (2402.07384), P (abstract):** controlled experiments showing small-object VQA accuracy falls with object quality and size and with object location and distractors (V-178). It does not name patch or token resolution as the mechanism, which is exactly the gap IDEA-21 fills, but it is the ancestor of the "pitch floor" hypothesis and should be cited in §2.
3. **Visual Position Prompt (2503.15426), S (search):** a method that adds explicit spatial references to help coordinate alignment; a neighbour of factor F4, not a factorial.
4. **PaliGemma / Pix2Seq location-token lineage:** PaliGemma uses 1,024 `<loc>` tokens over normalized coordinates (V-172); the file cites it correctly as a reference model. Florence-2's 1,000 bins are in the anchors (A2).
5. **Not found:** any paper crossing coordinate format with decoder, token pitch and position encoding at matched data with P@0.9-by-size evaluation and a hidden-state probe (three searches, V-179). This is weak evidence of absence, but the Skeptic's index and my searches agree.

## C. Researcher2's Section 9 requests

| Request | Verdict | Evidence |
|---|---|---|
| (1) Effective token pitch: Qwen3-VL 16-px patches with 2x2 merge; InternVL3.5 14-px patches pixel-unshuffled to 28 px | **VERIFIED.** Qwen3-VL-8B config: patch_size 16, spatial_merge_size 2, so 32 px (V-166). Qwen2.5-VL-7B config: patch_size 14, merge 2, so 28 px (V-167). InternVL3_5-8B config: patch_size 14, image_size 448, downsample_ratio 0.5, so 1,024 patches per 448-px tile become 256 tokens, 28 px per tile; dynamic tiling up to 12 tiles plus a thumbnail (V-168). | P (HF config.json files) |
| Caveat for (1) | For InternVL the 28-px pitch is per tile. Relative to the original image the pitch scales with the tile grid, so the §2 statement "28 px for InternVL" is true only in tile coordinates; state that the pitch factor for the InternVL reference model is defined per tile. | |
| (2) SA-Co usable as a box benchmark with non-COCO images | **VERIFIED.** The SA-Co/Gold dataset card (facebook/SACo-Gold) lists seven subsets whose images come from MetaCLIP and SA-1B only, and the annotation format is COCO-derived with RLE masks **and** bounding boxes in [x,y,w,h], plus positive and negative noun phrases; Silver and VEval are also on HF (V-170). No COCO, Objects365 or OpenImages images. The licence field reads "other" (search snippet: SAM licence, S). The uniqueness filter in §3.3 is still required because phrases are annotated exhaustively over all instances. | P (HF card, paper HTML) |
| (3) ExpVG magnitude of the coordinate-format effect | **PARTIALLY.** Direction verified (integer > location tokens "significantly"; integer > decimal "slightly"); numeric per-factor deltas are only in figures (V-158, V-192). | P (paper HTML) |

## D. Section 9 [LIKELY] items and other checked claims

| Claim | Verdict | Evidence |
|---|---|---|
| SA-Co composition | VERIFIED | V-160, V-170 (see C.2). |
| Objects365 "non-commercial licence" | PARTIALLY | Terms page: annotations CC BY 4.0, dataset "for the academic purpose only", images under Flickr terms, no redistribution (V-171). Say "academic-use terms, images not redistributable". |
| SAM licence for SA-Co | PARTIALLY | HF card licence field "other"; the SAM licence attribution is from a search snippet (S). Read the card's licence text before relying on it. |
| Effective token pitch of each reference model | VERIFIED for Qwen2.5-VL (28), Qwen3-VL (32), InternVL3.5 (28 per tile), PaliGemma (14, unmerged: 256 tokens at 224) | V-166 to V-168, V-172. Florence-2 and Rex-Omni pitches not checked. |
| PaliGemma pt loc-token behaviour | VERIFIED (format) | 1,024 `<loc0000>` to `<loc1023>` tokens, normalized, box as ymin xmin ymax xmax (V-172). Behaviour of the pt checkpoints on REC without transfer is not in the paper text I read (UNVERIFIABLE here). |
| Hi-Token / LocateAnything / PaDT details beyond abstracts | UNVERIFIABLE this pass | abstracts only (V-157). |
| Rex-Omni: 3B, 0-999 tokens | VERIFIED | V-176; also 22M SFT samples + RL with geometry-aware rewards. |
| GETok exists (grid + offset tokens) | VERIFIED | V-176. |
| MolmoPoint "28-px token too coarse-grained" | VERIFIED (verbatim) | V-175; note MolmoPoint claims about 4.7-px precision with its sub-patch tokens, which is a data point for the "decoding, not perception" reading and should be cited as such in §2. |
| Qwen convention change "with a rationale but no ablation" | VERIFIED | V-169. |
| ScreenSpot-Pro 0.07% and 1,581 items | VERIFIED | V-173. |
| Ref-L4 45,341 items, 30 to 3,767 px | VERIFIED | V-174. |
| GroundingME "Limited" dimension | VERIFIED | V-111. |
| Qwen3-VL coordinate exposure (0-1000 on COCO, Objects365, OpenImages, RefCOCO/+/g plus synthesis) | VERIFIED | anchors A2 (V-004). |
| ExpVG "0-1 floats lose to integers" as reason to drop the level | PARTIALLY | "slightly" (V-192). |
| Alignment and SFT throughput, edge-error model, predicted effect sizes | SPECULATION, unchecked | not checkable from sources. |
| LLaVA-OneVision-style 558k caption alignment | not checked (low stakes) | |

## E. Verifier A4 dependencies as stated in §6

Consistent with `landscape/verifier-anchors.md`: (a) RefCOCO saturation VERIFIED; (c) RL vs SFT PARTIALLY and not load-bearing because all arms are SFT; (d) attention read-out PARTIALLY and only the secondary probe. No dependence on (b) or (e). Correct.

## F. Actions requested of the author

1. Add Shikra §6.2 (2306.15195) and 2402.07384 to §5 with one-sentence deltas; cite MolmoPoint's 4.7-px claim in §2 as the existing "decoding, not perception" data point.
2. State that InternVL's 28-px pitch is per 448-px tile and define the pitch factor for tiled reference models accordingly.
3. Either keep the float level or replace the justification for dropping it with ExpVG's actual wording ("slightly").
4. Change the Objects365 licence label to "academic-use terms; images may not be redistributed", and read the SA-Co/Gold licence text on the HF card before the data plan is final.
5. Update §3.3: SA-Co/Gold ships boxes, so no mask-to-box conversion is needed; the uniqueness filter and the SA-1B hash-dedup stay.

## G. Addendum (2026-09-02): Researcher2's follow-up checks (4) and (5)

**(4) Patch merger concatenates, it does not average (basis of 21-M4): VERIFIED (V-201).** In HF transformers, Qwen2-VL's `PatchMerger` sets `hidden_size = context_dim * spatial_merge_size**2` and its forward is `mlp(ln_q(x).view(-1, hidden_size))` with a Linear-GELU-Linear MLP, so `spatial_merge_size**2` consecutive tokens are concatenated along the feature axis before the MLP. Qwen3-VL's `Qwen3VLVisionPatchMerger` does the same (`hidden_size = hidden * merge**2`, `view`, then `linear_fc1 / act / linear_fc2`, with a `use_postshuffle_norm` option) and one instance is created per DeepStack index. Caveat: which four tokens are consecutive is decided by the image processor's patch ordering, which I did not read; the Qwen2-VL paper's wording is "adjacent 2x2 tokens". Qwen2.5-VL is assumed to share the Qwen2-VL merger (LIKELY, not read). Design note for 21-M4: the fixed 2x2 average pooling used to derive the 32-px arm from the 16-px alignment is therefore not what the production merger does; the production merger is a learned function of the concatenated 4-token vector, so the pooled arm underestimates what a trained 32-px merger can recover and the file should say so.

**(5) PLAN.md Section 4 against Section 7 of the idea files (V-203):** IDEA-21 (355 primary) and IDEA-22 (300-350) match; IDEA-91 (220 minimum, 550 full) matches. Divergences: IDEA-11 PLAN full about 300 vs file 300-330 (PLAN halves the VISTA/PAPO baselines from 80 to 40 GPU-h, marked as Researcher2's review); IDEA-13 PLAN about 190 vs file components summing to about 225 (baselines 40 not 30, SAM 3 extraction 15 not 10, per-round evaluation 20 omitted), and the PLAN status line for IDEA-13 is stale (file is REVISED); IDEA-92 PLAN "arm A 250" vs file "arm A about 180" and PLAN about 300 vs file 20 + 300 + 20 = 340.

## H. Addendum (2026-09-02): SA-Co/Gold licence text

The dataset is manually gated on Hugging Face (contact details, affiliation, job title, geolocation and a checkbox acknowledging the terms and the Meta Privacy Policy); the card metadata carries license "other" with no license_name or license_link; a LICENSE file is present in the repo but both README.md and LICENSE return HTTP 401 before the gate is accepted (V-204). The licence text therefore stays UNVERIFIABLE from outside; the "SAM License" attribution is still only a search snippet. Whoever builds the data plan must accept the gate and read LICENSE first. The repo holds 21 JSON configuration files across the seven subsets, consistent with V-170.
