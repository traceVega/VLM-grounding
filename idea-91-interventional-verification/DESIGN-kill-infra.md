# Design: infrastructure for the IDEA-91 kill experiments

Status: v1.0, 2026-09-02. Review converged after four rounds with three reviewers (Section 13); the conditions attached to the final verdicts are applied as written. Scope: E91-0 to E91-2 in `EXPERIMENTS.md` (setup, the K1 artifact gate, the K2 removal test with its frontier subset and edit-free row). Section 2 is the pre-registration text; the B0a and B0b tags make it binding under the two-stage freeze rule P20.

Sources: `../team/ideas/IDEA-91-interventional-grounding-verification.md` Sections 3.1, 3.2, 3.4, 3.6a (S1 to S5) and 6; `../team/landscape/r3-history.md` Section 7 (shared notation); `../team/landscape/r2-practical.md` Sections 5.1, 5.6, 8.2 and 8.10; `../team/PLAN.md` S1, S2, S4, S5; `EXPERIMENTS.md` in this folder for machine specs; IDEA-11's `DESIGN-kill-infra.md` for the shared runtime; `../shared/harness/SPEC.md` v1.2 for the harness contract.

Deviations from the team's protocol, made after review and recorded for the team: (1) the K1 gate no longer uses hidden random crops, because in-mask compositing leaves no pixel outside the hole changed, so a hidden-crop AUROC is 0.5 by construction; the gate is a location-agnostic classifier on the full image at the resolution the models see. (2) The primary control edit is the removal of a non-referent object of matched size and centrality (CONTROL_OBJ), not a background hole, because a background fill is near-invisible while an object removal smears. (3) Under the benchmark's rejection protocol, stop rule (b) cannot fire on abstention (the models' rejection rate is near zero), so a box-shift form of (b) is pre-registered for the primary protocol, with a background-control comparison so instability does not count as confusion. (4) The verifier's clean rule asks about the head noun on the edit window, not the full expression, because relational context usually lies outside the window. Section 12 proposes the corresponding amendment to `r3-history.md` Section 7; this document does not edit team files.

## 1. Purpose, scope and success criteria

Purpose. Build the smallest set of code, data and environment that (K1) decides whether a classifier can tell a referent removal from a matched non-referent removal without being told where the hole is, at the resolution a policy sees, and (K2) measures, on verified clean removals, whether current grounders re-predict the same box when the referent is gone, whether their boxes move under matched control edits, and how the frontier models behave on a subset, such that the stop rules can be evaluated with no choice made after seeing results.

In scope: environment additions, data pipeline (pools and benchmarks), the instance and edit stack with the window operator, the artifact gate, the judge service with the edit verifier, the harness additions (conditions, rejection protocols, Molmo2 path, backend parity), the relation code for necessity, box shift and text-side controls, the frontier API client, the human-check protocol, analysis.

Out of scope: attribute swaps and outside edits, the crop judge and sufficiency, necessity-plus, the twelve-class audit, K3 and K4, GroundLive, the training bank, any training.

Acceptance checks:
1. Gate validity. (1a) Two sensitivity ladders, both run on the 1b pairs so no real class signal contaminates them. Global ladder (classifier capacity): one of the two CONTROL_OBJ edits per image re-encoded whole at JPEG quality 75, 90 or 92 against the other at quality 95; AUROC above 0.9 at q75 is required, and the q92 step must reach AUROC at least 0.7 in at least one gate row of P5 at the frozen gate resolution, else the gate resolution is raised before the K1 freeze. Local ladder (what the editor can leave): the same pairs with JPEG re-encoding at quality 75, 50 or 30 composited inside the dilated hole only, original pixels elsewhere; the smallest detected local step (AUROC at least 0.7) per hole-area bin is the gate's local floor and is carried into the first table of every downstream use; a local floor coarser than q75 in any hole-area bin reads as the gate being blind to editor-scale artifacts in that bin, not as the editor being clean. (1b) Identical-distribution null: two independent CONTROL_OBJ edits per image, one labelled each way, AUROC within 0.5 plus or minus 0.05 in every gate row. (1c) Paired-crop null: crops at identical coordinates outside both holes from the REMOVE and CONTROL_OBJ versions are pixel-identical (compositing assert) and give AUROC 0.5.
2. Sampler validity. No control hole overlaps the exclusion set of P3 with the candidate's own mask removed from it, asserted over every pool and item.
3. Verifier validity. Human-versus-verifier Cohen's kappa on clean versus not clean, under the P10 head-noun rule and on determined items only (P10's undetermined outcome is excluded from kappa), at least 0.6 on the random 200 of P11; this alone decides whether verifier-clean is the decisive set. The paired difference in same-box between human-clean and verifier-clean items is reported with its bootstrap CI as a calibration line and does not gate. If kappa fails, the human check escalates per P11 and human-clean carries the headline.
4. Harness validity. IDEA-11's acceptance check 1 (its run ids are cited, not re-run); a Qwen3-VL vLLM parity run under this idea's P21 override set on dev slice (a) per SPEC Section 5, cited in every K2 manifest; and a Molmo2 smoke on the `p12` dev slice under `--non-kill`: 50 items through the vLLM path with points and abstentions parsed, and at least 95% point-in-box agreement with a `trust_remote_code` Hugging Face spot check at batch one.
5. Regeneration. Every number in the K1 and K2 tables regenerates from the configs, the edit index and the result store with one command; manifests as in IDEA-11's check 4.

## 2. Pre-registered definitions

| ID | Decision | Value | Why |
|---|---|---|---|
| P1 | K1 pool | 10,000 OpenImages images (CC-BY; author, URL and licence recorded per image from the images CSV for attribution) with at least one annotated object whose SAM 3 mask, prompted with the image's class labels, covers 0.5% to 15% of the image; one referent instance per image with seed 0. All class labels of the image are prompted and class-agnostic masks above 0.5% area are added, so the exclusion set covers unlabelled objects. SA-1B is the fallback pool and makes that bank non-releasable | CC-BY images make the K1 bank the only releasable edited artefact |
| P2 | Hole types and window operator | Mask hole: the SAM 3 mask dilated by 5 px. Rectangular hole: the mask's bounding box. Editing runs on a window of 4 times the hole's longer side, at least 768 px and at most 2,048 px on the longer side, clipped to the image, resized to at most 2,048 px for LaMa and pasted back at native scale; compositing is in-mask (LaMa output inside the dilated mask, original pixels elsewhere; asserted by check 1c). The window PNG and its offset are stored; the full edited image is composed at load time | GroundingME images run to 7,680 px |
| P3 | Control edits | CONTROL_OBJ (primary): removal, with the same operator and hole type, of a non-referent instance, preferring a labelled instance of another class (K1) or a SAM 3 instance of a non-referent noun phrase (K2), and falling back to a class-agnostic mask, recorded as `control_source`; area between 0.5 and 2 times the referent's; centre distance from the image centre within plus or minus 20% of the diagonal relative to the referent's; no overlap with the exclusion set after the candidate's own mask is removed from it. CONTROL_BG (secondary): the referent's hole shape at a position with no overlap with the exclusion set, inpainted the same way. Exclusion set: the referent mask dilated 16 px, every SAM 3 instance of every parsed noun phrase of the expression (K2) or of every class label (K1), and every class-agnostic mask above 0.5% area. Rejection sampling up to 100 tries; items with no valid CONTROL_OBJ are dropped and counted per set and bin | Matched visibility, matched position and matched source, so neither "edited", "central smear" nor "stuff versus thing" separates the classes |
| P4 | Always-edited design | K1: every image has exactly one hole; the label is whether the hole covered the referent (REMOVE) or a matched other object (CONTROL_OBJ). K2: every item has ORIGINAL, REMOVE, CONTROL_OBJ and CONTROL_BG versions | "Was this image edited" is uninformative by construction |
| P5 | Gate inputs | Location-agnostic rows, no hole coordinates given to any classifier: (i) ResNet-18 on the full edited image at the smaller of native size and the 2.4 Mpx cap; (ii) ViT-S/16 on native-resolution 512 px tiles covering the image, each tile inheriting the image label, aggregated per image by the maximum tile score; (iii) both classifiers on the full image at 1,024 px longer side. Reported rows: both classifiers at 512 px longer side; a shown crop centred on the hole (1.5 times the hole's longer side, resized to 256). Paired crops outside both holes: the null of check 1c only | The gate must see what the policy sees; a 512 px downscale removes the high-frequency inpainting texture |
| P6 | Classifiers | ResNet-18 and ViT-S/16 (dynamic image size) from timm, ImageNet-pretrained, 5 epochs, AdamW lr 1e-4, batch 8 to 16 with mixed precision at the native rows and 64 at 512 px, 80/20 split by source image, AUROC per image on the 20%, 3 seeds, mean and bootstrap CI over images; AUROC also split by `control_source`. Strong adversaries, reported without a hard gate, one seed each with activation checkpointing at batch 8: a fine-tuned DINOv2-B on the 1,024 px input and on the tiles, and a forensic detector on full images: TruFor if its weights are obtainable under research terms, else an RGB-domain detector of the PSCC-Net or MVSS-Net class (CAT-Net is excluded because it reads JPEG DCT streams and the windows are PNG); the choice, weights and sha256 in `PINS.md` | The team's probes and adversaries on the policy's input |
| P7 | K1 gate | Pass if the maximum over the gate rows (i), (ii) and (iii) of the AUROC of ResNet-18 and of ViT-S/16 on REMOVE versus CONTROL_OBJ is at most 0.6, on both hole types. REMOVE versus CONTROL_BG is reported on the same rows. A DINOv2-B AUROC above 0.7 on any gate row is reported in the first table of every downstream use. The sensitivity requirement of check 1a must hold at the frozen gate resolution, and a PASS is always reported together with its local floor per hole-area bin from the local ladder. Frozen at the K1 freeze point (P20) | The team's 0.6 convention on the input a policy actually sees, with a proven sensitivity floor |
| P8 | K2 items | Anchored on ground truth: all GroundingME positive single-box items whose ground-truth box has a SAM 3 mask (box prompt on the window) at box-to-mask IoU at least 0.5 and referent area at most 30% of the image, then OpenRef single-target positives under the same rules until 2,000 pairs, stratified by size bin (IDEA-11's D3). Items with box-to-mask IoU below 0.5 get a rectangular hole and the `box_inpaint` flag, capped at 20% of the set. A count table per set, bin and control source is committed before the K2 freeze; if OpenRef is unavailable (O1), K2 is every eligible GroundingME item and the CI half-widths in P15 are restated | The team's K2 sets; the later audit anchors on predictions |
| P9 | Removal region | The SAM 3 mask when box-to-mask IoU is at least 0.5, else the box; if the ground-truth box contains several SAM 3 instances of the head noun, their union; the box-to-mask IoU distribution is reported per set. R denotes this region | Shared notation, applied to the ground-truth box |
| P10 | Edit verifier | Gemma4-12B through the judge service at a pinned commit SHA; temperature 0, `max_tokens 4`, first-word match on yes or no, unparseable counted as not clean. Three-way outcome for REMOVE, decided before the question is asked: if any other SAM 3 instance of the head noun (from the P3 noun-phrase instances) overlaps the edit window, the item is undetermined by a same-noun neighbour and gets the secondary rule below; otherwise, on the edit window (shown at up to 1,536 px), "Is there a {head noun} in this image? Answer with one word, yes or no." gives clean on "no" and not clean on "yes". Secondary rule for undetermined items: the same question on a tight crop, the removal region R dilated by 50% of its size and at least 256 px on the longer side; if a neighbour instance overlaps the tight crop too, the item stays undetermined, is excluded from every verifier-based clean statistic and from kappa, and its share is reported per set and GroundingME dimension in the K2 count table; an undetermined item that carries human labels (the pilot, or any escalation draw) stays in the human-clean column, and human-versus-V1 disagreement on such items is reported descriptively. K1 removal success: the same class-label question on the window for both K1 classes (REMOVE and CONTROL_OBJ) where a class label exists; removal success and the gate AUROC on verified-removed pairs are reported per `control_source`, and controls with `control_source` equal to class_agnostic, which have no label to ask about, are marked unverified (V2 only) and never mixed into the verified-pairs row. Validity rule for CONTROL_OBJ and CONTROL_BG: the same head-noun question on the referent's window in the control image (the referent is untouched there) must be "yes". Secondary, reported: the full expression on the full image at the 2.4 Mpx cap for REMOVE and controls. V2 on REMOVE, reported only: "Does this image show signs of editing such as blurred, smeared or repeated texture? Answer yes or no." Prompts are versioned files with hashes in `verifier.parquet` | Relational context usually lies outside the window, so the full expression on the window would answer "no" whether or not remnants remain; the head noun is what the removal changes |
| P11 | Human check | Frame: after the ORIGINAL condition has been scored (a pre-specified, data-dependent rule), 200 REMOVE items drawn with seed 0, stratified by size bin, from determined items (P10) whose ORIGINAL box is correct for at least one of the two models, plus 100 items drawn from determined items that V1 calls not clean; the undetermined share of the ORIGINAL-correct frame is reported next to the per-set shares, and the human-clean and verifier-clean columns are therefore computed on the same determined frame. Three paid non-author annotators recruited on a crowd platform, each labelling every item, blind to condition and to each other; shown the edit window at 2 times the hole and the full image, with the head noun and the expression, never the original; labels clean, remnant (traces of the object remain), failed (object still present); majority label; written instructions with two examples per label; private Label Studio instance on a small VPS with HTTPS and individual accounts (the images are research-licensed and never public). Statistics: human-human Fleiss kappa (3-way and binarised) on the random 200 and on all 300; human-versus-V1 Cohen's kappa on the random 200 (check 3); expected human-clean denominator for same-box about 170 (half-width about 6 points at 20%). Cost about 300 dollars at about 0.30 dollars per label plus platform fee; turnaround one week; no personal data beyond platform ids. Escalation if check 3 fails: 500 items drawn with seed 0 from the same determined, ORIGINAL-correct frame (denominator about 425, half-width about 4 points), about 500 dollars | The 200-item check the team wrote, made defensible and drawn where the same-box statistic lives |
| P12 | Rejection protocols | Per model class, literal strings in `configs/prompts/`. Qwen3-VL primary: GroundingME's own instruction and null-box convention; secondary: the same instruction followed by "If the object is not present, output none." Molmo2 primary: its native pointing instruction "Point to the {expr}." with its native abstention (a no-such-object sentence or an empty point list, parsed per its card); secondary: the same followed by "If there is no such object, say none." The secondary protocols are validated on the `p12` dev slice of 100 gRefCOCO no-target items and 100 positives under `--non-kill`; all K2 conditions are run under both; the stop rules are evaluated under the primary; the secondary block is descriptive | The team's numbers are under the benchmark's protocol |
| P13 | Conditions | ORIGINAL, REMOVE, CONTROL_OBJ, CONTROL_BG, T_NULL (the grounding instruction with the expression replaced by "the object"), T_HEAD (the head noun replaced by a category from a fixed list of 50 common categories whose SAM 3 concept prompt returns no instance in the image, chosen with seed 0; up to 50 prompts per image, budgeted), T_ATTR (the discriminating attribute swapped by Qwen3.5-9B text-only; reported, not scored). A centre-and-median-size prior box is computed per item as the baseline for T_NULL | The team's text-side controls |
| P14 | Same-box | On items where the ORIGINAL box is correct (IoU with ground truth at least 0.5) and the REMOVE is clean, same-box means IoU(box on REMOVE, box on ORIGINAL) at least 0.5. Also reported: IoU(box on REMOVE, R) at least 0.3, the complement of the shared notation's necessity pass. Molmo2: the REMOVE point lies inside R | The team's K2 rule speaks of re-predicting the same box |
| P15 | Rates and power | Per model, protocol, set and size bin, with item-level bootstrap CIs (2,000 resamples, clustered by image): `none` on clean REMOVE; box on clean REMOVE (verifier-clean and human-clean columns; editor-remnant responses their own row); same-box (P14); box shift on CONTROL_OBJ and on CONTROL_BG, computed on ORIGINAL-correct items (IoU with the ORIGINAL box below 0.5); abstention on CONTROL_OBJ; T_NULL pass (IoU with the ORIGINAL box below 0.5 or `none`) next to the prior baseline; T_HEAD pass (moved or `none`). UNRESOLVED: `output_type` changes between ORIGINAL and CONTROL_OBJ or IoU(CONTROL_OBJ box, ORIGINAL box) below 0.5; excluded from same-box, share reported. REDUNDANT: more than one head-noun instance in the image; reported separately. Expected denominators, before the undetermined exclusion of P10 whose share is reported: about 2,000 pairs, about 85% verifier-clean, about 60% ORIGINAL-correct, so about 1,000 items per model for same-box under verifier-clean (half-width about 3 points at 20%) and about 170 under human-clean (about 6 points), both after the undetermined exclusion, whose share reduces them proportionally and is reported | Every rule's denominator and CI is known before the run |
| P16 | Stop rules (primary protocol; verifier-clean decisive when check 3 passes, else human-clean) | (a) Both Qwen3-VL-8B and Molmo2-8B output `none` on more than 60% of clean REMOVE items: no rejection headroom, stop the training half; expected to be unfireable under the primary protocol and reported as such. (b) Secondary-protocol block only: both models emit a box on more than 80% of clean REMOVE items and abstain on more than 30% of CONTROL_OBJ items. (b′) Primary-protocol form of (b): both models emit a box on more than 80% of clean REMOVE items, their box shifts on more than 30% of ORIGINAL-correct CONTROL_OBJ items, and that shift exceeds the CONTROL_BG shift on the same items by more than 10 points; a CONTROL_BG shift within 10 points of the CONTROL_OBJ shift reads as instability of the model, not confusion by the edit, and does not fire. Consequence of (b′): fix the editor before any other use. (c) Same-box below 20% for both models: keep only the rejection half. Point estimates decide; CIs reported. Added reading, not a stop: same-box on the frontier subset below 5% means the audit headline will be "boxes are load-bearing" and the metric path is planned | The team's S5 rules made computable, with the abstention leg translated and the instability confound removed |
| P17 | Frontier subset | 500 K2 pairs stratified by size bin and set, conditions ORIGINAL, REMOVE and CONTROL_OBJ. Providers: Qwen3-VL-235B-Instruct on Alibaba Model Studio (first-party; the Thinking variant is not used), Gemini 3 Pro on Google's API. Per-provider YAML pins the dated model id, temperature 0, thinking disabled where configurable, media resolution setting, safety settings, coordinate convention (Gemini `[ymin, xmin, ymax, xmax]` on 0 to 1000); images are the composed edited image resized to the P21 cap and sent inline as PNG bytes, never by URL; if the PNG exceeds 5 MB it is re-encoded at PNG compression level 9 and, if still over the provider's hard limit, downscaled to fit with the size logged; only provider tiers with no retention of inputs and no training on inputs are used, and the tier and retention terms are recorded per provider in `LICENSES.md` because the images are research-licensed; `image_px_sent` logged per row; per call log of request model string, response model string, timestamps, token counts and cost; a hard cap of 200 dollars; a 500-item drift re-query scheduled at submission. Provider terms recorded in `LICENSES.md` | The team's conditions plus what survives drift |
| P18 | Edit-free row | COCO-Search18's own distributed images (1,680 by 1,050, padded; MIT data terms with no commercial use and no redistribution, recorded) for all 18 categories, target-present and target-absent; prompt "the {category}" under the primary protocol; primary statistic: the box rate on target-absent images (no annotations needed); secondary: on present images, IoU at least 0.5 with any instance of the category from COCO 2017 instance annotations, mapped by filename first and by pHash against both train2014 and val2014 hashes otherwise (val2014 is a 6 GB contingency download), with mapping coverage reported against the full COCO id space and a 20-image visual check of the letterbox transform; Qwen3-VL-8B and Molmo2-8B | The natural-pair check that needs no editing |
| P19 | Molmo2 path and resolution | Served by vLLM (native support), pinned revision; the Hugging Face `trust_remote_code` path is the batch-one parity spot check of check 4; point-in-box scoring; abstention per P12. Resolution policy: the processor's default crop settings (maximum crop count and crop size), recorded in the model config as `resolution_policy`; Molmo2's own visual-token-count histogram over K2 and COCO-Search18 images is written before the run; `image_px_sent` filled for every row | Molmo2's token count comes from its crop tiling, not from `max_pixels` |
| P20 | Freeze rule | Two freeze points. K1 freeze: P1 to P7 are frozen after the pool, instances, edits and count tables exist and before the first classifier trains on real removals (the JPEG ladder and the nulls of check 1 may run before, on their own pairs). K2 freeze: P8 to P21 are frozen after the K2 instances, edits and count tables exist and before the first ORIGINAL item is scored. Each manifest carries the version in force. After a freeze no value changes except through the declared contingencies (O1 for OpenRef; the check 1a resolution raise before the K1 freeze); any other change forks the document and both versions' results are reported | The gate result cannot influence the gate constants |
| P21 | Resolution and backend for models under test | Qwen3-VL: run-level override `max_pixels = 2,457,600`, processor default `min_pixels`, vLLM `max_model_len 4096`; the share of images hitting the cap is reported per set; a token-count histogram is written before the run and no request may be rejected. This differs from IDEA-11's D4, which fixes `min = max` at the same value and therefore upscales smaller images; the two are identical only for images at or above 2.4 Mpx, and the setting for the later joint audit table is decided in the audit design. Backend: Qwen3-VL runs on vLLM only after the parity run of check 4 under this override set; Molmo2 per P19 | GroundingME images would otherwise reach 16k tokens; vLLM throughput is what the K2 budget assumes |

## 3. System overview

```
data/raw/<pool|set>                          downloads, revision SHAs recorded
data/prepared/<set>/items.parquet            shared schema (pair_id set for K2)
        |
        v
idea91/instances  --sam3-->                  edits/index.parquet rows with masks      (P1, P8, P9)
idea91/edits      --window, lama, sampler--> edits/<pool>/<image_id>/<instance_id>/<operator>.png plus offsets  (P2, P3, P4)
        |
        v
idea91/gate       --k1-->                    tables/k1.md                             (P5, P6, P7)
        |
        v
shared/judges     --serve verifier-->        edits/verifier.parquet                   (P10)
        |
        v
shared/harness    --run model conditions-->  results/<run_id>/outputs.parquet         (P12, P13, P19, P21)
idea91/frontier   --api-->                   results/<run_id>/outputs.parquet         (P17)
        |
        v
idea91/relations  --n v t-->                 results/<run_id>/relations.parquet       (P14, P15)
idea91/analysis   --k2-->                    tables/k2.md, tables/k2_frontier.md, tables/cocosearch.md  (P16, P18)
```

Code homes: `shared/` (environment, data, harness, result store, judge service) is shared with IDEA-11; `idea91/` holds instances, edits, gate, frontier client, relations and analysis. The harness contract is `../shared/harness/SPEC.md` v1.2 with `schema.py`; this document carries no copy and lists its additions in Section 5.

## 4. Components

### 4.1 Environment and runtime

IDEA-11's design Section 4.1 applies (Windows host; native Linux or WSL2 with `memory=28GB`, data on ext4, containers with `--gpus all`; one Dockerfile and lockfile). Additions, each pinned by name, revision and file sha256 in `shared/env/PINS.md`: the SAM 3 repository and checkpoint (SAM 3.1 if that is the current release at pin time; recorded), the LaMa package and big-LaMa weights, timm, DINOv2-B weights as a file, the forensic detector of P6, SAM 2 with its automatic mask generator if SAM 3 has no generic-object mode (O6), the vLLM build. Memory on the 5090: SAM 3 under 8 GB; LaMa under 2 GB on 2,048 px windows; classifier training under 8 GB at 512 px and under 16 GB at the native rows with mixed precision and batch 8 to 16; DINOv2-B fine-tune under 16 GB; Gemma4-12B on vLLM about 25 GB with `gpu_memory_utilization` derived from measured idle VRAM (about 0.88 with a display attached), `max_model_len 4096`, alone on the card; Qwen3-VL-8B or Molmo2-8B on vLLM about 17 GB, alone on the card. B1's test serves Gemma4-12B, answers 50 verifier questions and one Qwen3.5-9B text call, with peak VRAM logged.

### 4.2 Data pipeline

| Set | Source and pin | Size | Use |
|---|---|---|---|
| OpenImages subset | official downloader by image id; images CSV for attribution | 10 to 20 GB | K1 pool (P1) |
| SA-1B shards | Meta release, two shards; research licence, non-releasable derivatives | 20 GB | K1 fallback pool |
| GroundingME | Hugging Face at a revision SHA | about 3 GB; images 1,500 to 7,680 px | K2 (P8) |
| OpenRef | release of arXiv 2605.25706 (O1) | 2 to 5 GB | K2 (P8) |
| Ref-L4 dev slice (a) | IDEA-11's `shared/data` materialization | shared | the parity run of check 4 |
| gRefCOCO no-target | annotations; images from COCO train2014 (13 GB; the only COCO download in either idea, val2014 as a P18 contingency) | 13 GB | the `p12` dev slice |
| COCO-Search18 | the dataset's own images and present/absent lists; COCO 2017 instance annotations for the secondary statistic | about 5 GB plus annotations | P18 |

`items.parquet` uses the shared schema with `pair_id` set for K2 and `dev_slice` equal to `p12` for the gRefCOCO slice. Count tables per set, hole type, control type and control source are committed at each freeze. The single `data/LICENSES.md` carries rows for SAM 3 (custom licence; the GitHub text permits derivative distribution with a copy of the agreement and acknowledgement, to be confirmed against the full text behind the gate, O7), OpenImages (CC-BY, attribution columns), SA-1B (research; no redistribution of derivatives), GroundingME (research use under SA-1B and HR-Bench terms; K2 edits are not releasable), OpenRef (to be read), COCO-Search18 (no commercial use, no redistribution), and the API providers' terms. Releasable artefacts, if any: the OpenImages K1 bank only.

### 4.3 Instance and edit stack

`idea91/instances`: SAM 3 concept prompts per image (class labels for K1; for K2 the head noun and every noun phrase of the expression, parsed by the text-only LLM Qwen3.5-9B with the parser output stored); class-agnostic masks above 0.5% area from SAM 3's generic mode if available, else SAM 2's automatic mask generator (O6; 1 to 3 s per image, 5 to 8 hours over the K1 pool and K2 images as a contingency); for K2 the referent mask by SAM 3 box prompt on the edit window; masks as RLE with area fraction, bounding box, centre, `control_source` and box-to-mask IoU (P9).

`idea91/edits`: the window operator (P2), the removal and rectangular operators, the CONTROL_OBJ and CONTROL_BG samplers (P3, with the assertion of check 2), the always-edited builder (P4), and the composer that rebuilds a full edited image from the original plus window at load time. Every edit is indexed in `edits/index.parquet` (Section 5) with editor name, weights sha256 and settings hash.

Edit counts and throughput on the 5090: K1 needs 10,000 REMOVE, 10,000 CONTROL_OBJ, 10,000 second CONTROL_OBJ edits for check 1b and 10,000 CONTROL_BG rows, about 40,000 edits, about 3 to 4 hours at about 0.1 s per LaMa window plus SAM 3 time; K2 needs about 8,000 edits, under 1 hour; T_HEAD's up to 50 SAM 3 prompts per K2 image about 3 to 4 hours.

### 4.4 Artifact gate

`idea91/gate`: the gate inputs of P5 (full image at the cap, native tiles with max aggregation, 1,024 px, 512 px, shown crop, paired crops), classifier training and evaluation (P6), the DINOv2-B and forensic adversaries, the global and local JPEG ladders on the 1b pairs and the two nulls of check 1, AUROC per image with bootstrap CIs over images, `tables/k1.md` with rows per control type (CONTROL_OBJ gate rows first, CONTROL_BG comparison rows), per hole type, per classifier, per input row, per `control_source`, the removal-success rate per class and per `control_source` with class-agnostic controls marked unverified, the gate AUROC on verified-removed pairs per `control_source`, and the local floor per hole-area bin. Frequency-domain check: high-pass residual energy inside versus outside the hole, once per editor, reported.

### 4.5 Judge service and edit verifier

`shared/judges`: `judges.yaml` (name, hf path, commit SHA, role, lineage family) and `serve.sh` for one judge at a time; a client with a cache keyed by sha256 of (model, prompt version, image bytes, question); prompts as versioned files; the lineage rule enforced in config. Outputs to `edits/verifier.parquet` with an unparsed count. Throughput: about 3 to 5 judgments per second on the 5090 at window size; K2's about 10,000 verifier calls (REMOVE window and full, tight crops for undetermined items, two control windows, V2) in about 45 minutes; K1's 20,000 class-label calls in about 1.5 hours.

### 4.6 Harness additions

The contract is SPEC v1.2. This idea adds: `--conditions` resolving REMOVE, CONTROL_OBJ, CONTROL_BG, T_NULL, T_HEAD and T_ATTR to composed images or rewritten expressions; two `abstain_protocol` entries per model (P12); the Molmo2 vLLM path with the point parser, point-in-box scorer and `resolution_policy` (P19); the P21 override set and its parity run on dev slice (a); token-count histograms for both models over K2 and COCO-Search18 images written before the run; a check that no request is rejected by the server; `image_px_sent` on every row.

### 4.7 Relation code

`idea91/relations`: functions over `outputs.parquet` joined on `pair_id`, `condition` and `abstain_protocol` that compute P14 and P15 per item, the clean and validity flags from `verifier.parquet` under the P10 rules and from `human_labels.csv`, and the REDUNDANT and UNRESOLVED flags. Sufficiency, necessity-plus and the score are not implemented here.

### 4.8 Frontier API client

`idea91/frontier`: one adapter per provider with the P17 YAML, the P21-sized composed image inline as PNG, logging, caching, rate limiting with retries, the cost cap, refusals recorded as their own `output_type`. Outputs land in `outputs.parquet` with `model_id` set to the response model string. The B10 test uses dev-slice items, never K2 items before the K2 freeze. The drift re-query is the same command with a new run id, and its diff of model strings is a table.

### 4.9 Human-check protocol

P11 in full: frame and seed, recruitment, blinding, what is shown, label definitions with examples, majority rule, statistics with expected denominators, hosting on a small VPS with HTTPS, cost, turnaround and escalation. The instruction text and the two examples per label are committed with the K2 freeze. B9's pilot (10 items from the `p12` dev slice edited the same way, three annotators) reports human-human kappa before any K2 scoring.

### 4.10 Analysis

`idea91/analysis`: `k1.py` (Section 4.4 tables, the check 1a sensitivity floor and the P7 verdict), `k2.py` (P15 rates with CIs per model, protocol, set and bin; P16 verdicts as single lines with the CONTROL_BG comparison; verifier-clean and human-clean columns side by side with the calibration line of check 3; REDUNDANT and UNRESOLVED shares; the prior baseline for T_NULL), `frontier.py` (the same rates on the 500 pairs per API model and the drift diff), `cocosearch.py` (P18 with mapping coverage). Figure 1 draft: one item in four panels, original with the box, removal with the box unchanged, matched object removal with the box unchanged, a COCO-Search18 absent image with a box still emitted.

## 5. Interfaces

`items.parquet` and `outputs.parquet`: SPEC v1.2 (conditions, `pair_id`, `abstain_protocol`, `point_xy_px`, `point_in_gt`, `image_px_sent` are columns of the normative schema; `iou_gt` and `is_failure` are defined for ORIGINAL rows only).

`edits/index.parquet`: `image_id`, `set_or_pool`, `instance_id`, `pair_id`, `operator` (REMOVE, CONTROL_OBJ, CONTROL_BG, RECT_REMOVE, RECT_CONTROL_OBJ, RECT_CONTROL_BG), `editor`, `editor_weights_sha256`, `editor_settings_hash`, `mask_rle`, `hole_type`, `mask_area_frac`, `box_to_mask_iou`, `box_inpaint_flag`, `control_instance_id`, `control_source` (labelled_other_class, noun_phrase_instance, class_agnostic), `control_area_ratio`, `control_centrality_delta`, `window_xyxy_px`, `window_path`, `window_sha256`, `created_at`, `freeze_version`.

`edits/verifier.parquet`: `window_sha256`, `verifier_model`, `verifier_commit`, `prompt_version`, `question` (V1_headnoun_window, V1_headnoun_control_window, V1_expr_full, V2), `answer_raw`, `answer_bool`, `parsed_ok`, `logprob_yes` if available.

`edits/human_labels.csv`: `window_sha256`, `annotator_id`, `label`, `timestamp`, `sample_stratum` (random, v1_negative, escalation).

`results/<run_id>/relations.parquet`: `pair_id`, `model_id`, `abstain_protocol`, `orig_correct`, `remove_clean_verifier`, `remove_clean_human`, `control_obj_valid`, `control_bg_valid`, `none_on_remove`, `box_on_remove`, `same_box_50`, `same_box_r30`, `shift_on_control_obj`, `shift_on_control_bg`, `abstain_on_control_obj`, `t_null_pass`, `t_null_prior_pass`, `t_head_pass`, `t_attr_moved`, `redundant`, `unresolved`.

`tables/k1.md`, `tables/k2.md`, `tables/k2_frontier.md`, `tables/cocosearch.md`: the fields in 4.10 plus the freeze version and run ids.

## 6. Storage and compute budget

| Item | Local 5090 | Cloud |
|---|---|---|
| Models | Qwen3-VL-8B 17.6 GB, Molmo2-8B 17 GB, Gemma4-12B 25 GB, Qwen3.5-9B 19 GB, SAM 3 about 2 GB, SAM 2 about 1 GB (contingency), LaMa 0.2 GB, DINOv2-B and the forensic detector under 1 GB | none for K1 and K2 |
| Data | OpenImages subset 10 to 20 GB, COCO train2014 13 GB (val2014 6 GB contingency), GroundingME 3 GB, OpenRef 2 to 5 GB, COCO-Search18 about 5 GB, SA-1B fallback 20 GB, Ref-L4 shared | none |
| Edits and outputs | K1 about 40,000 windows about 16 GB plus 1,024 px and tile inputs about 6 GB; K2 about 8,000 windows about 6 GB; verifier and outputs under 2 GB | none |
| GPU time | K1 14 to 20 h (edits 3 to 4, removal verification 1.5, gate classifiers 4 to 6, adversaries at one seed 5 to 8, ladders 1); K2 25 to 35 h (SAM 3 and T_HEAD 4, edits 1, verifier 1, parity run 1, two models times two protocols times six conditions about 25,000 prompts at 3 to 5 per second on vLLM about 2 h each block, COCO-Search18 1 h). Contingency: SAM 2 automatic masks 5 to 8 h | none |
| Cash | about 300 dollars for the human check (500 more on escalation); 50 to 150 dollars API; a VPS for Label Studio about 10 to 20 dollars a month | none |

## 7. Build plan

| Step | Deliverable | Effort | Test |
|---|---|---|---|
| B1 | `shared/env` additions with `PINS.md`; runtime per 4.1 | 1 to 2 days (SAM 3 access may gate this) | SAM 3 masks on a sample; LaMa fills a window; Gemma4-12B serves 50 questions; peak VRAM logged |
| B2 | `shared/data`: pools, benchmarks, COCO-Search18, gRefCOCO, `LICENSES.md` rows, attribution columns | 2 days, mostly waiting | counts committed; area-band histogram |
| B3 | `idea91/instances` and `idea91/edits` for the K1 pool | 4 to 5 days | check 2; compositing assert (check 1c); PNG plus offset round-trip; index rows complete with `control_source` |
| B4a | `idea91/gate` ladders and nulls on the 1b pairs | 1 day | check 1a global and local ladders (sensitivity requirement met or the resolution raised), check 1b null, check 1c; per-image AUROC CI on a toy set |
| B0a | K1 freeze (P1 to P7) with the K1 count tables and the frozen gate resolution | half a day | tag created; no classifier has trained on real removals |
| B4b | `idea91/gate` classifiers, adversaries and the removal-success rows | 1 day | K1 table code recomputes a toy table exactly |
| B5 | K1 full run | 1 day | `tables/k1.md`; P7 verdict |
| B6 | K2 instances and edits, count tables | 2 days | index rows complete; drop rates per set and bin |
| B7 | `shared/judges` and the verifier under the P10 rules | 1 to 2 days | 50 handmade cases including relational expressions with context outside the window; cache hits; unparsed count |
| B8 | harness additions (4.6), Molmo2 path, the P21 parity run on dev slice (a) and the `p12` dev-slice runs, all `--non-kill` | 2 days | check 4; token-count histograms for both models |
| B9 | `idea91/relations`, `idea91/analysis`, the human-check project and the 10-item pilot on `p12` items | 2 to 3 days | toy table recomputed exactly; UNRESOLVED and REDUNDANT unit tests; pilot kappa reported |
| B0b | K2 freeze (P8 to P21) with the K2 count tables and the P11 instruction text | half a day | tag created; no ORIGINAL item scored |
| B10 | `idea91/frontier` tested on dev-slice items | 1 day | 10 real calls per provider with every logged field inspected; cap test; `image_px_sent` at most 2.4 Mpx |
| B11 | K2 full run: ORIGINAL first, then the P11 draw and annotation in parallel with the remaining conditions, the frontier subset and the COCO-Search18 row | 2 to 3 days of GPU plus one week of annotation turnaround | `tables/k2.md`, `k2_frontier.md`, `cocosearch.md`; P16 verdicts |

About 4.5 person-weeks to the K2 verdict when IDEA-11's harness exists, 5.5 otherwise.

## 8. Risks and mitigations

- SAM 3 access is delayed: K1 can start with SAM 2 masks prompted by the OpenImages ground-truth boxes, labelled as such; K2 waits.
- The gate's sensitivity requirement fails at 1,024 px and the cap: the resolution rule in check 1a raises it before the K1 freeze; the classifier budget grows accordingly.
- Referents in benchmarks are more central or larger than other objects: P3 matches centrality, area and source; drop rates and per-source AUROC show what remains.
- The verifier is unreliable (check 3 fails): the human-clean set carries the headline and the check escalates to 500 items (about 500 dollars).
- Near-zero rejection under the primary protocol: (a) is expected to be unfireable and is reported as such; (b′) carries the confusion test with the instability guard; the secondary block shows the prompt effect.
- GroundingME's largest images: the window operator, the P21 cap and the token-count histograms; the share at the cap is reported.
- Molmo2 support in vLLM changes: the pinned revision and the smoke of check 4.
- API drift or refusals: P17 logging; refusals as their own output type; the drift re-query.
- OpenRef unavailable (O1): K2 on GroundingME alone with restated CIs.
- The 20% cap on rectangular holes truncates the set: the dropped count is reported.

## 9. Open items

- O1 (open): OpenRef's release format and licence.
- O2 to O4 (closed in v0.2).
- O5 (open, low priority): 1,000 frontier pairs would halve the CI at about twice the API cost; 500 is pre-registered.
- O6 (open until pin time): SAM 3's generic-object mode for class-agnostic masks, or SAM 2's automatic generator as the fallback.
- O7 (open until read): the exact SAM 3 licence text behind the gate.
- O8 (new, deferred to the audit design): the resolution setting for the joint audit table shared with IDEA-11.

## 10. Not designed here on purpose

Attribute swaps and outside edits, the crop judge and text-only prior, necessity-plus, the audit, K3, K4, GroundLive, the training bank, rewards and training.

## 11. Glossary

REMOVE, CONTROL_OBJ, CONTROL_BG, T_NULL, T_HEAD, T_ATTR: conditions (P13). Clean: a removal the verifier (head-noun question on the window) or a person labels as object absent. Same-box: P14. Gate rows, shown, paired crops: P5. Gate: P7. R: the removal region (P9).

## 12. Proposed amendment to the shared notation (for the team, not applied here)

`r3-history.md` Section 7 defines `ctrl_R'(I)` as an artifact-matched control edit on a non-overlapping region of the same size and shape. This design splits it into `ctrl_obj` (removal of a non-referent instance of matched area, centrality and source, the primary control for K1, K2 and, later, the reward pairs) and `ctrl_bg` (the background hole, kept for invariance and as the instability comparison). It also adds the window operator for high-resolution images and the head-noun clean rule for the verifier. IDEA-11's Gate 2, which inherits "IDEA-91's control-edit protocol", should use `ctrl_obj` for its control-mask drop.

## 13. Review log

Round 1, 2026-09-02. Three reviewers: systems (RS), methods (RM), data and operations (RD). Verdict before revision: NOT CONVERGED. Responses (v0.2):

- RS-91-1 and RM-91-1 (MAJOR and BLOCKING, hidden crops leak or are vacuous): ACCEPT. In-mask compositing stated and asserted; the gate became a location-agnostic full-image classifier; paired crops survive only as the null.
- RM-91-2 (MAJOR, matched visibility): ACCEPT. CONTROL_OBJ primary, CONTROL_BG comparison.
- RM-91-3 (MAJOR, exclusion set): ACCEPT.
- RM-91-4 (MAJOR, freeze timing): ACCEPT. Two freeze points.
- RM-91-5 and RS-91-7 (MAJOR and MINOR, power and the decisive clean set): ACCEPT. Verifier-clean decisive under check 3; UNRESOLVED defined; denominators in P15.
- RM-91-6 (MAJOR, unfireable rules): ACCEPT. (b′) pre-registered.
- RM-91-7, RM-91-8 (MINOR): ACCEPT.
- RS-91-2, RS-91-3 (MAJOR, Molmo2 path and protocol): ACCEPT.
- RS-91-4 (MAJOR, resolution): ACCEPT. P21 cap.
- RS-91-5 and RD-11-5 (MAJOR, Windows runtime): ACCEPT by reference.
- RS-91-6 (MINOR): ACCEPT.
- RD-91-1 (BLOCKING, human protocol): ACCEPT. P11 and 4.9.
- RD-91-2 (MAJOR, image sizes): ACCEPT. Window operator, P21.
- RD-91-3 (MAJOR, licences): ACCEPT.
- RD-91-4 (MAJOR, API logging): ACCEPT.
- RD-91-5 (MAJOR, COCO-Search18): ACCEPT.
- RD-91-6 (MAJOR, pins): ACCEPT.
- RD-91-7, RD-91-8 (MINOR): ACCEPT.
- Cross-document items: ACCEPT, resolved as in IDEA-11's log. RM-X-2 (shared notation): ACCEPT in substance; Section 12 proposes the amendment.

Round 2, 2026-09-02. Verdicts on v0.2: RD CONVERGED (minor items), RM NOT CONVERGED (four major), RS NOT CONVERGED (one major). Responses (v0.3):

- RM-91-9 and RS-91-8 (MAJOR, gate input resolution): ACCEPT both. P5 gate rows are the full image at the cap for ResNet-18, native-resolution 512 px tiles with max aggregation for ViT-S/16, and both at 1,024 px; 512 px is a reported row; P7 gates on the maximum over the gate rows; check 1a requires the q92 step to be detected at the frozen gate resolution, else the resolution is raised before the K1 freeze; Section 6 restated.
- RM-91-10 (MAJOR, verifier clean rule): ACCEPT. P10 asks the head-noun question on the REMOVE window; controls are validated by the head-noun question on the referent's window in the control image; the full expression on the full image is secondary; conservative failure on other head-noun instances in the window with the share reported; deviation (4) recorded.
- RM-91-11 and RD-91-N1 (MAJOR and MINOR, calibration clause power): ACCEPT. Check 3 is decided by kappa alone; the same-box difference is a calibration line with its paired-difference CI; the human frame is items ORIGINAL-correct for either model, drawn after the ORIGINAL condition (pre-specified, data-dependent), so the human-clean denominator is about 170 and about 425 on escalation (P11, P15).
- RM-91-12 (MAJOR, (b′) denominator): ACCEPT. Shift computed on ORIGINAL-correct items; (b′) requires the CONTROL_OBJ shift to exceed the CONTROL_BG shift by more than 10 points; the instability reading is pre-registered (P15, P16).
- RM-91-13 (MINOR, control source): ACCEPT. `control_source` with labelled other-class instances preferred; AUROC split by source (P3, P6, Section 5).
- RM-91-14 (MINOR, ladder pairing): ACCEPT. The ladder runs on the 1b pairs (check 1a).
- RM-91-15, RS-91-9, RD-91-N3 (MINOR, non-kill smoke and dev-slice runs): ACCEPT. Check 4 and P12 name the `p12` dev slice and `--non-kill`; the frontier client sends the P21-sized composed image at most 5 MB with `image_px_sent` logged (P17, 4.8, B8, B10).
- RS-91-4 residual and RD-91-N5 (MINOR, Molmo2 resolution): ACCEPT. P19 records Molmo2's crop policy as `resolution_policy`, writes its own token histogram and fills `image_px_sent`.
- RS-91-10, RD-91-N4, RD-X-N1 (MINOR, budgets and backend parity): ACCEPT. Edit counts and hours restated (4.3, Section 6); SAM 2 masks, the escalation and the VPS as contingencies; Qwen3-VL runs K2 on vLLM only after the P21 parity run on dev slice (a), listed in B8 and check 4; Ref-L4's dev slice is in `shared/data`.
- RD-91-5 residual (MINOR, COCO-Search18 mapping): ACCEPT. Filename matching first, then pHash against train2014 and val2014 (val2014 as a contingency download); coverage reported against the full COCO id space (P18).
- RD-91-N2 (MINOR, forensic detector): ACCEPT. CAT-Net excluded; TruFor or an RGB-domain detector, pinned (P6).
- RS-X-4 and RM-X-5 (MINOR, "the same setting"): ACCEPT. P21 states the difference from IDEA-11's D4; the joint audit setting is deferred (O8).
- RD-11-N4 as it applies here (MINOR, freeze order): ACCEPT. B0a follows B3 and B0b follows B9, so each freeze has its count tables.

Round 3, 2026-09-02. Verdicts on v0.3: RS CONVERGED, RM CONVERGED conditional on RM-91-16 applied as written, RD NOT CONVERGED (two major). Responses (v0.4):

- RD-91-N7 (MAJOR, exclusion set contains the candidate): ACCEPT. P3 and check 2 exclude the candidate's own mask from the exclusion set before the overlap test; this was a wording bug that would have failed every CONTROL_OBJ hole.
- RD-91-N6 (MAJOR, construct mismatch in the clean rule): ACCEPT with one change to the suggested secondary rule. P10 is a three-way outcome (clean, not clean, undetermined by a same-noun neighbour detected from the P3 instances); kappa and the oversample use determined items only; the undetermined share is reported per set and dimension in the K2 count table; the secondary rule for undetermined items is the head-noun question on a tight crop around the removal region rather than a 2x or neighbour-masked window, because a larger window admits more neighbours and masking a neighbour would add an edit; items whose tight crop still contains a neighbour stay undetermined.
- RM-91-16 (MAJOR, local sensitivity): ACCEPT as written. Check 1a adds the in-mask JPEG ladder (q75, q50, q30) on the 1b pairs with the local floor per hole-area bin; P7 carries the floor with every PASS and the pre-registered reading; the global ladder is kept as the classifier-capacity check.
- RM-91-17 (MINOR): ACCEPT. Removal success is verified for both K1 classes with the class-label question; the gate AUROC on verified-removed pairs is a row (P10, 4.4).
- RS-91-11 (MINOR, adversary cost): ACCEPT. One seed with activation checkpointing at batch 8 for the adversary rows; Section 6 restated at 14 to 20 hours for K1.
- RD-91-N8 (MINOR, freeze order): ACCEPT. B4 is split; the ladders and nulls run before B0a and the classifiers after.
- RD-91-N9 (MINOR, API tiers and the size limit): ACCEPT. P17 restricts providers to no-retention, no-training tiers recorded in `LICENSES.md` and states the over-limit rule.
- RS-X-5 (MINOR, SPEC wording and `effective_resolution`): ACCEPT in SPEC v1.2.

Round 4, 2026-09-02. Verdicts on v0.4: RD CONVERGED (two minor), RM CONVERGED (one minor, and the P10 change reopens neither RM-91-10 nor RM-91-11), RS converged in round 3. Responses (v1.0):

- RD-91-N10 (MINOR, scope of the undetermined exclusion): ACCEPT. P10 scopes the exclusion to verifier-based statistics and kappa; human-labelled undetermined items stay in the human-clean column with the disagreement reported descriptively.
- RD-91-N11 (MINOR, unverifiable class-agnostic controls): ACCEPT. Removal success and the verified-pairs AUROC are reported per `control_source`; class-agnostic controls are marked unverified and kept out of the verified-pairs row (P10, 4.4).
- RM-91-18 (MINOR, human frame): ACCEPT as written. The random 200 and the escalation are drawn from determined items in the ORIGINAL-correct frame; the undetermined share of that frame is reported; P15's denominators are stated after the exclusion (P11, P15).
- Verdict after application: all three reviewers converged; v1.0.
