# IDEA-91 experiments: local RTX 5090 versus cloud

Date: 2026-09-02. Local machine: one RTX 5090 (32 GB VRAM) with 32 GB CPU RAM. Cloud: RunPod-class rental; prices below were queried on 2026-09-02.

Source: `../team/ideas/IDEA-91-interventional-grounding-verification.md` Sections 3, 6, 7 and 11, `../team/landscape/r2-practical.md` Sections 8.2 and 8.10, and `../team/PLAN.md`. H100-hour budgets are the team's. Memory peaks, 5090 wall-clock, cloud machine specs and dollar figures are my estimates, assuming bf16 weights and vLLM for serving; treat them as within about 30%.

## 0. Rules of thumb for this machine

What fits on the 5090 (32 GB VRAM):
- One model up to about 12B in bf16 through vLLM at a time: Qwen3-VL-4B (9 GB), Qwen3-VL-8B (17.6 GB), Molmo2-4B and 8B (9 and 17 GB), InternVL3.5-8B (17 GB), the 7B box-in-trace reasoners and crop models (about 15 GB each), judge A Qwen3.5-9B (19 GB), judge B and edit verifier Gemma4-12B (25 GB, with `gpu_memory_utilization` 0.92 and `max_model_len` 4096).
- The whole editing stack: SAM 3 (under 8 GB), big-LaMa (under 2 GB), and a 4B instruction editor for attribute swaps (FLUX.2 klein 4B or Mage-Flow-Edit-Turbo 4B, 16 to 20 GB peak at 1024 by 1024), SAM 3 and LaMa together, the editor on its own.
- K1's probes: ResNet-18, ViT-S/16, a TruFor-class forensic detector, a DINOv2-B fine-tune (under 16 GB).

What must run sequentially rather than side by side: a policy under test and a judge (vLLM cannot hold Qwen3-VL-8B and Gemma4-12B at once on 32 GB). Dump the crops and outputs to disk, swap models, judge in a second pass. This doubles wall-clock relative to a two-card box and is the main reason the audit batch is cheaper on cloud.

What never fits locally: Qwen3-VL-32B or UI-Venus-2-27B in bf16; any GRPO run (EasyR1 needs 4 cards for a 4B policy plus one for the judge); the training edit bank at a useful pace (three to four days of continuous editing on one card is possible but ties the machine up).

CPU RAM at 32 GB: load with `low_cpu_mem_usage=True`; keep dataloader workers at 4 or fewer; stream edited images to disk rather than holding batches in memory; vLLM `swap_space` 2. The perceptual-hash index over COCO (about 120k images) fits in RAM; indexes over Objects365, OpenImages or SA-1B do not fit on this machine's disk, see E91-6.

Software for Blackwell (sm_120): PyTorch wheels built for CUDA 12.8 or newer, a vLLM build with Blackwell support, diffusers for the 4B editors, the SAM 3 repository with its gated checkpoint, a LaMa inpainting package.

Conversion: 5090 wall-clock is about 3 times the H100-hour figure for prefill-heavy inference, about 1.5 to 2 times for the small editors and classifiers, and 4 to 5 times for training, which is not done locally. An RTX PRO 6000 Blackwell 96 GB runs at about 5090 speed with three times the memory: it holds a policy and a judge together, or the 32B model, at about 2 dollars an hour.

## 1. Cloud options (RunPod, queried 2026-09-02, USD per GPU-hour)

| GPU | VRAM | Secure | Community | Stock | Use it for |
|---|---|---|---|---|---|
| RTX PRO 6000 Blackwell, server edition | 96 GB | 2.09 | 1.69 | high | policy plus judge on one card, 32B audit rows, edit-bank building; same software stack as the 5090 |
| H200 SXM | 141 GB | 4.59 | 3.59 | medium | fastest single card; 32B rows with headroom |
| H100 SXM | 80 GB | 3.29 | 2.69 | low | EasyR1 training nodes (E91-10, E91-11), throughput batches |
| H100 PCIe or NVL | 80 or 94 GB | 2.89 or 3.19 | 1.99 or 2.59 | low | single-card throughput when SXM is out |
| A100 80 GB, PCIe or SXM | 80 GB | 1.39 or 1.59 | 1.19 or 1.39 | low | cheapest card for the edit banks and inference batches |

Dollar lines below use secure prices; community is about 20% lower. Network volumes cost on the order of 0.07 dollars per GB per month and are the right home for the training edit bank, which the training node must read.

## 2. Summary

| ID | Experiment | Where | Local wall-clock (5090) | Cloud machine | Cloud hours | About USD |
|---|---|---|---|---|---|---|
| E91-0 | Setup: SAM 3 access, editors, judges, data | LOCAL | engineering, 1 to 1.5 person-weeks | none | 0 | 0 |
| E91-1 | K1 artifact gate | LOCAL | 12 to 16 h (gate rows at the policy's resolution, design P5) | none | 0 | 0 |
| E91-2 | K2 removal test, frontier subset, edit-free row | LOCAL plus API | 25 to 35 h | none (API rows) | 0 | 50 to 150 API, about 300 human check (500 more on escalation) |
| E91-3 | Evaluation edit bank, 8k items times 4 edits | EITHER | 15 to 45 h | 1 A100 80 GB or 1 RTX PRO 6000 96 GB | 15 to 20 | 25 to 40 |
| E91-4 | Part A audit rows, 12 model classes times 6 operators | EITHER, cloud recommended for the batch | 50 to 65 h for the 8B-and-smaller rows; 32B not local | 2 RTX PRO 6000 96 GB, or 2 H100 80 GB | 40, or 18 | 170, or 120; plus 200 to 400 API for the frontier rows |
| E91-5 | Diagnostic arm: priors under outcome rewards | LOCAL | 30 h | none | 0 | 0 |
| E91-6 | GroundLive generator and de-duplication | LOCAL generation; CLOUD storage box for full-corpus hashing | 15 h | CPU-only pod with a 2 to 4 TB volume | 24 to 48 CPU-hours | 20 to 50 plus volume |
| E91-7 | K3 human validation, 1,000 items times 3 annotators | no GPU | 1 h for sampling | none | 0 | about 500 |
| E91-8 | K4 predictive validity and baselines | LOCAL | 20 to 30 h | 1 H100 80 GB (for speed) | 8 to 10 | 30 |
| E91-9 | Training edit bank, 20k images | CLOUD recommended | 70 to 75 h, three days, possible | 1 A100 80 GB or 1 H100 80 GB, plus a 500 GB volume | 40, or 28 | 65, or 92, plus 35 per month for the volume |
| E91-10 | GRPO arm (c) at 4B, 3 seeds | CLOUD | not possible | 8 H100 SXM 80 GB node (4 train, 1 judge, 3 for a second seed) | 25 wall, 100 GPU-h | 350 to 450 |
| E91-11 | K5 direct arm, four cheap counterfactuals at equal compute | CLOUD | not possible | 8 H100 SXM 80 GB node | 40 to 50 wall, 300 to 400 GPU-h | 1,000 to 1,300 |
| E91-12 | Evaluation of trained arms, judge B, forgetting add-on | LOCAL for spot checks; training node for the batch | 75 h for all arms | run on the training node before release | 8 wall on 4 cards | about 100 |

Local-only path to the audit decision: E91-0, E91-1, E91-2, about 40 to 50 hours of 5090 time, about 300 dollars of human checking and 50 to 150 dollars of API calls. The design document DESIGN-kill-infra.md (v0.3) supersedes the E91-1 and E91-2 lines where they differ: the gate runs at the resolution the models see, the control edit is a matched object removal, and the human check is 300 items with three annotators. The metric decision (E91-3, E91-4 subset, E91-7, E91-8) adds about 500 dollars of annotation and either four days of local GPU time or about 150 dollars of cloud time.

## 3. Experiments in order

### E91-0 · Setup · LOCAL

Purpose. Everything below shares the editing stack, the judge service and the relation code.

Day one: request SAM 3 access (gated checkpoint, custom licence that governs any released edit bank or instance set). Install SAM 3, big-LaMa, one 4B instruction editor (FLUX.2 klein 4B, Apache-2.0, or Mage-Flow-Edit-Turbo 4B, MIT), vLLM, and the judge models. Write the relations as code from `../team/landscape/r3-history.md` Section 7: invariance, necessity, necessity-plus with the hole masked from the judge's crop, sufficiency with the text-only prior, text-side controls, the removal-region rule (SAM 3 mask when box-to-mask IoU is at least 0.5, else a rectangular hole flagged `box-inpaint`), the duplicates rule (REDUNDANT, UNRESOLVED), and the always-edited design (every image, including positives, carries a matched control edit). Write the judge-lineage rule into the service config: judge A Qwen3.5-9B for rewards and the crop judge, judge B Gemma4-12B held out, the GroundLive generator of a different lineage from the judge, the edit verifier different from the reward judge.

Data to download (about 60 GB core): GroundingME (1,005 items), OpenRef, Ref-Adv 2026 (COCO images), Ref-L4 annotations with COCO train2014 and val2014 (19 GB) and the Objects365 test images it uses (about 2 GB), gRefCOCO no-target annotations, COCO-Search18 target-present and target-absent image lists (COCO images; non-commercial terms, evaluation only), VisMin, two or three SA-1B shards (20 to 30 GB), an OpenImages subset (10 to 20 GB). Later: Charades videos for the diagnostic arm (55 GB), LISA-Grounding.

Models to download (about 70 GB core): Qwen3-VL-8B 17.6 GB, Molmo2-8B 17 GB, Qwen3.5-9B 19 GB, Gemma4-12B 25 GB, SAM 3 about 2 GB, big-LaMa 0.2 GB, the 4B editor 8 to 16 GB, Grounding DINO 0.7 GB, TruFor and DINOv2-B under 1 GB. Later, for E91-4: Qwen3-VL-4B, Molmo2-4B, InternVL3.5-8B, GRIT, TreeVGR, Rex-Thinker, DeepEyes, Chain-of-Focus (about 100 GB more), and Qwen3-VL-32B on the cloud card only.

Local spec. GPU RAM any; CPU RAM 32 GB fine; storage 200 GB free for the core, 400 GB to hold every model in E91-4. Time: 1 to 1.5 person-weeks, most of it the relation code and the control-edit sampler.

### E91-1 · K1 artifact gate (KE-2) · LOCAL

Decides. Whether the editor leaves a global signature that a policy could learn as "edited implies absent."

Stop rule. Hidden-location AUROC above 0.6 on either hole type: change the editor or the control design; no downstream consumer touches the bank until this passes.

Procedure. From the COCO-free pool (SA-1B or OpenImages), 5k referent-removed and 5k control-edited images, separately for SAM 3 mask holes and rectangular holes, every image edited. Train ResNet-18 and ViT-S/16 (ImageNet init, 5 epochs) to separate them with the referent location hidden by random crops; report AUROC hidden and shown. Report a TruFor-class forensic detector and a fine-tuned DINOv2-B as the strong adversary without a hard gate. Add the frequency-domain check (high-pass residual energy inside versus outside the masks) once per editor.

Data. 10k pool images with SAM 3 instances. Models. SAM 3, big-LaMa, ResNet-18, ViT-S/16, TruFor, DINOv2-B.

Local spec. GPU RAM up to 16 GB; CPU RAM 12 GB; storage 8 GB of edited images; wall-clock 6 to 9 hours (SAM 3 and LaMa about 1 hour, probe training 2 to 3 hours, DINOv2 fine-tune 2 to 3 hours, forensic detector under 1 hour).

Output. The gate result per hole type, quoted in every table that uses edited images; the editor and control settings frozen for E91-2, E91-3 and E91-9.

### E91-2 · K2 removal test with a frontier subset and an edit-free row (KE-3) · LOCAL plus API

Decides. Whether current models keep emitting the same box when the referent is gone, and whether the edits inform rather than confuse.

Stop rules. Team: `none` on more than 60% of removals means no rejection headroom; a box on more than 80% of removals together with abstention on more than 30% of controls means the edits confuse, fix the editor; same-box re-prediction below 20% means keep only the rejection half. Added reading: same-box re-prediction on human-verified clean removals below 5% on the frontier subset means the audit headline will be "boxes are load-bearing"; plan for the metric-only paper and let E91-7 and E91-8 decide it.

Procedure. Build 2k REMOVE plus CONTROL pairs on GroundingME and OpenRef images with SAM 3 masks and LaMa; verify every removal with Gemma4-12B as the edit verifier; send 200 to people (about 100 dollars). Run Qwen3-VL-8B and Molmo2-8B (point prompt, point-in-box scoring) with native prompts on originals, removals and controls, plus the text-side operators (null expression, attribute swap, head swap). Record `none` on removals, box on removals (reported on human-verified clean removals only, editor-remnant responses as their own row), abstention on controls, and same-box re-prediction where the original box was correct. Frontier subset: 500 pairs through Qwen3-VL-235B and Gemini 3 Pro by API with model version, prompt and query date recorded. Edit-free row: COCO-Search18 target-absent photos with category prompts, present and absent side by side, Molmo2 as the cleaner lineage.

Data. GroundingME, OpenRef, COCO-Search18 images (3,101 plus 3,101). Models. SAM 3, LaMa, Qwen3-VL-8B, Molmo2-8B, Gemma4-12B, API models.

Local spec. GPU RAM 17 to 25 GB, one model at a time (policies first, then the verifier in a second pass); CPU RAM 12 GB; storage 10 GB; wall-clock 15 to 30 hours (edits under 1 hour, about 30k policy prompts at 3 to 5 per second, 4k verifier calls, the COCO-Search18 row).

Cash. About 100 dollars for the human check; 50 to 150 dollars of API calls for about 2k image requests.

Output. The K2 table; the seed of the evaluation bank; removal success by referent size, which sets the edit floor for IDEA-92's necessity edits and IDEA-11's edit-pair negatives; the box-on-clean-removal rate that IDEA-92 quotes as its motivating number.

### E91-3 · Evaluation edit bank, about 8k items times 4 edits · EITHER

Purpose. The bank every audit row and the metric read from: for each item on GroundingME, OpenRef, Ref-Adv 2026 and the Ref-L4 subsets, a removal of the referent's SAM 3 mask, a matched-size control removal elsewhere, an outside edit (distractor insertion or background edit) for invariance, and an attribute swap with the 4B editor, every edit verified by the edit verifier and keyed by image and instance; plus the crop-region removal recipe for crop models.

Data. The four benchmarks (about 8k items). Models. SAM 3, LaMa, the 4B editor, Gemma4-12B as verifier.

Local spec. GPU RAM up to 20 GB, stages run one after another (SAM 3 and LaMa, then the editor, then the verifier); CPU RAM 12 GB; storage 15 GB; wall-clock 15 to 45 hours (bottom-up: 16k LaMa edits under 1 hour, 8k swaps at about 2 seconds each, 32k verifier calls at about 0.3 seconds; the team's 15 H100-hour figure converts to the upper end).

Cloud alternative. 1 A100 80 GB (about 1.5 dollars an hour) or 1 RTX PRO 6000 96 GB (about 2 dollars an hour), 64 GB host RAM, 100 GB disk, 15 to 20 hours, 25 to 40 dollars; both the editor and the verifier fit at once, which removes the model swapping.

Output. The evaluation bank with per-edit verifier outcomes and the K1 gate result per hole type attached.

### E91-4 · Part A audit rows, about 12 model classes times 6 operators · EITHER, cloud recommended for the batch

Purpose. The table the paper is remembered for. Plain grounders (Qwen3-VL-4B, 8B, 32B; Molmo2-4B, 8B; InternVL3.5-8B; Grounding DINO plus SAM 3 as the specialist), box-in-trace reasoners (GRIT, TreeVGR, Rex-Thinker; every box in the trace anchored, answer and box both reported), crop models (DeepEyes, Chain-of-Focus; the crop region anchored, plus the observation-replacement comparison), frontier rows by API (Qwen3-VL-235B, Gemini 3 Pro on the same bank, 500-item drift re-query at submission). Outputs per model: invariance, necessity, necessity-plus, sufficiency, text pass rates, the score, and the right-box-wrong-reason rate on the human-validated subset (after E91-7), stratified by referent size, expression type, instance count and model class; the RefCOCO-family rows as a separated memorization-probe row only.

Data. The E91-3 bank, natural pairs (COCO-Search18, video exits from SA-V or Ego4D frames, multi-instance photos, VisMin). Models. All of the above; judge A on crops for sufficiency.

Local spec for the 8B-and-smaller rows. GPU RAM up to 25 GB, policy passes and judge passes alternating; CPU RAM 16 GB; storage 100 GB for the models plus 20 GB of outputs; wall-clock 50 to 65 hours (about 500k prompts at 3 to 5 per second plus about 100k crop judgments). Iterate on one or two models locally, then batch.

Cloud spec for the batch. 2 RTX PRO 6000 96 GB (policy and judge resident together, about 40 hours, about 170 dollars) or 2 H100 SXM 80 GB (about 18 hours, about 120 dollars); host RAM 128 GB; disk 300 GB. The Qwen3-VL-32B rows need a 96 GB or larger card in bf16 (about 8 hours on a PRO 6000, about 15 dollars) or run locally in 4-bit as a labelled preliminary. Frontier rows: about 100k API image calls, 200 to 400 dollars, no GPU.

Output. The audit table, one row per model and item, with flags; the cross-idea column for IDEA-11 (read-out probe IoU on the shared Qwen3-VL sizes).

### E91-5 · Diagnostic arm: priors under outcome rewards · LOCAL

Purpose. Whether current RL grounders solve items with a box prior: centre and median-size prior, head-noun prior (SAM 3 on the head noun, largest or most central instance), training-set location histogram. Run Visual-RFT, VLM-R1 and Time-R1 class models on iid versus OOD (Charades-CD with TimeLens re-annotations, both Ref-Adv sets, null-expression inputs). Decision rule: an iid-to-OOD gap above 10 points on any split, or a null-expression hit rate within 5 points of the box-prior model, promotes a prior-relative advantage to a section of the training half; otherwise one diagnostic table.

Data. Charades videos (55 GB) with Charades-CD splits and TimeLens re-annotations, Ref-Adv 2020 and 2026, the audit sets. Models. Visual-RFT 2B or 7B, VLM-R1 3B, Time-R1 7B, SAM 3.

Local spec. GPU RAM up to 20 GB; CPU RAM 16 GB (video decoding); storage 60 GB more; wall-clock about 30 hours.

Output. One table, or one section if the rule fires.

### E91-6 · GroundLive generator and de-duplication · LOCAL generation, CLOUD storage for full-corpus hashing

Purpose. A refreshable post-cutoff evaluation set: Creative-Commons images uploaded after the latest documented cutoff among all audited models (after release date where a closed model's cutoff is undocumented), perceptually hashed against COCO, Objects365, OpenImages, SA-1B and accessible LAION-derived sets, with reverse-image spot checks; expressions written by Gemma4-12B and judged by Qwen3.5-9B, the reverse split as a check; stratified to GroundingME's referent-size and SAM 3 instance-count distribution; a 200-item human anchor per refresh. The paper releases the generator, not a fixed set.

Local spec for generation. GPU RAM up to 25 GB, writer and judge in two passes; CPU RAM 12 GB; storage 10 GB per refresh; wall-clock about 15 hours for a few thousand images including SAM 3 counts for stratification.

Cloud spec for de-duplication. The COCO index (about 120k images, 19 GB) builds locally in an hour. Objects365 (about 365 GB), OpenImages train (about 560 GB) and SA-1B (about 11 TB) do not fit this machine. Rent a CPU-only pod with a 2 to 4 TB network volume, download Objects365 and OpenImages once and the SA-1B shards actually used for training, compute perceptual hashes (24 to 48 CPU-hours, 20 to 50 dollars plus the volume for the month), and keep only the hash lists (a few GB). If that is out of scope, restrict de-duplication to COCO and the pools you train on, and state the coverage in the paper.

Output. The generator code, the hash lists, and the first GroundLive batch used by E91-8.

### E91-7 · K3 human validation · no GPU

Purpose. 1,000 stratified items from the audit table, three annotators, labels "box correct", "box correct but expression ambiguous", "box wrong"; targets kappa at least 0.6 between sufficiency and "box correct", AUROC at least 0.85 of the score for human "box correct" among IoU-correct items. Human evidence marks from AiR, VQA-MHUG, DeFacto-1.5K and COCO-Search18 fixations are used only to validate the score.

Stop rule. Score AUROC below 0.75 or kappa below 0.5: no metric contribution; the paper is the audit.

Spec. About 1 hour of local GPU time for stratified sampling and crop export; about 500 dollars of annotation; two weeks of turnaround, so launch it as soon as E91-4 has rows for six models. Fold IDEA-92's 300-claim set into the same batch if it runs.

### E91-8 · K4 predictive validity and the baselines it must beat · LOCAL

Decides. Whether the score is more than the judge.

Stop rules. Item-level AUROC of the score for correctness at Ref-L4 IoU 0.75 or on GroundingME below 0.75; the score does not beat the judge alone by 0.03 overall or 0.05 on the judge-wrong subset; the score does not beat a two-seed or two-temperature disagreement predictor by 0.05; secondary, the unit-level Spearman between the score on GroundLive and Ref-L4 Acc at 0.75 (Objects365-test half for Qwen3-VL bases) with a cluster-bootstrap CI lower bound below 0.5.

Procedure. From the audit table, compute AUROC and ECE for the judge alone, necessity-plus and invariance alone, and the score; rerun the six grounders at two temperatures on the labelled sets for the disagreement predictor; run verification-based zero-shot REC and OpenRef's consistency checker as item-level baselines; the unit-level design waits for training checkpoints from E91-10 and is reported on base models and sizes only until then.

Data. The audit table, GroundLive batch, Ref-L4 IoU 0.75 labels, GroundingME. Models. The six locally runnable grounders; baseline code from 2509.09958 and OpenRef.

Local spec. GPU RAM up to 25 GB; CPU RAM 12 GB; storage 10 GB; wall-clock 20 to 30 hours for the two-temperature reruns; the analysis is CPU.

Cloud alternative. 1 H100 SXM 80 GB, 64 GB host RAM, 100 GB disk, 8 to 10 hours, about 30 dollars.

Output. Figure 2 of the paper; the metric verdict.

### E91-9 · Training edit bank, 20k images · CLOUD recommended

Purpose. The reward lookup for E91-10 and E91-11, shared with IDEA-92 if it runs: 20k COCO-free images (SA-1B, OpenImages; Objects365 train conditional), about 8 SAM 3 instances each, one LaMa removal per instance, two attribute swaps per image with the 4B editor, one copy-paste distractor insertion, matched controls for everything, every edit verified by Qwen3.5-9B as the edit verifier (a different model from the reward judge in the arm where Gemma4 judges, and the reverse), keyed by image and instance; LaMa fallback for off-bank boxes at reward time.

Data. 20k pool images (two SA-1B shards plus an OpenImages subset). Models. SAM 3, LaMa, the 4B editor, Qwen3.5-9B.

Local spec. Possible: GPU RAM up to 20 GB in stages; CPU RAM 16 GB; storage 150 to 250 GB (about 440k edited images at 300 to 500 KB); wall-clock 70 to 75 hours, about three days of continuous editing and verification. Not recommended because the bank must then be uploaded to the training node's volume anyway.

Cloud spec. 1 A100 80 GB (about 40 hours, about 65 dollars) or 1 H100 SXM 80 GB (about 28 hours, about 92 dollars); host RAM 64 GB; a 500 GB network volume (about 35 dollars a month) that E91-10 and E91-11 mount. Editor and verifier fit together, so no swapping.

Output. The training bank with verifier outcomes and K1's gate result attached; the same bank serves IDEA-92's arm A if it is staffed.

### E91-10 · GRPO arm (c) at 4B, three seeds, Molmo2-4B replicate · CLOUD

Runs only after E91-1 to E91-4 exist and E91-7 has not stopped the metric; nothing is trained before the audit.

Setup. EasyR1, Qwen3-VL-4B policy, 20k prompts from the bank, rewards as bank lookups (the relations on the instance whose mask best overlaps the predicted box; UNRESOLVED items contribute zero), the positive-rate constraint against always-abstaining, judge A (Qwen3.5-9B) serving sufficiency on its own card, LaMa on the same card for off-bank boxes; three seeds; the forgetting add-on before the node is released.

Cloud spec. One 8 H100 SXM 80 GB node: 4 cards for training, 1 for the judge and inpainter, 3 for a second seed's training in parallel with the judge shared; host RAM 256 GB or more; the 500 GB volume from E91-9 plus 200 GB for checkpoints (keep three of 9 GB per seed); about 100 GPU-hours, 25 wall-hours with two seeds side by side then the third; 350 to 450 dollars secure. If H100 SXM stock is out: 4 H200 for the training half at about the same dollar figure, or 5 RTX PRO 6000 96 GB over PCIe at about 1.5 times the wall-clock. The Molmo2-4B replicate is another 50 to 100 GPU-hours (200 to 350 dollars).

Output. Arm (c) checkpoints with intermediate saves (the unit-level design in E91-8 uses 5 per run), evaluated in E91-12.

### E91-11 · K5 direct arm: four cheap counterfactuals at equal compute · CLOUD

Decides. Whether realistic edits with control edits earn the training contribution.

Stop rule. For any of gray mask (DeFacto-style), blur-ghosting with the visibility and geometric-invariance rewards (VISE-style), or the attention-masked blind state (VIGIL-style), the 95% CI of the (c) minus arm difference on natural-pair necessity-plus or on GroundingME rejection includes 2 points: drop contribution 3 and report the comparison as a negative result in the appendix. PAPO's corrupted-image KL runs as the fourth arm.

Cloud spec. One 8 H100 SXM 80 GB node; host RAM 256 GB or more; the E91-9 volume plus 400 GB for checkpoints; 300 to 400 GPU-hours, 40 to 50 wall-hours with two arms side by side; 1,000 to 1,300 dollars secure, 850 to 1,100 community. Run it only if E91-10's arm (c) beats the base on GroundingME rejection and natural-pair necessity-plus by more than the item-level CI; otherwise there is nothing to compare against.

Output. Table 2 of the paper, with the CI of (c) minus each arm printed.

### E91-12 · Evaluation of trained arms, held-out judge, forgetting add-on · LOCAL for spot checks, training node for the batch

Purpose. GroundingME (four dimensions), OpenRef F1 and N3R, gRefCOCO no-target, both Ref-Adv sets, Ref-L4 Acc at 0.5, 0.75 and 0.9 and mean accuracy, RefCOCO-family for non-regression only, natural-pair necessity-plus, sufficiency under judge A, judge B and the human subset, box shifts under control edits as the invariance monitor, and MMMU, MMBench and POPE before and after every arm; the judge-gaming criterion (gain under judge A exceeding the gain under judge B by more than 5 points).

Local spec. A 4B checkpoint evaluates on the 5090 at up to 20 GB; one arm's full evaluation is about 8 to 10 hours; all arms of E91-10 and E91-11 are about 75 hours, so spot-check locally and batch on the node.

Cloud spec. About 8 wall-hours on 4 cards of the training node before it is released, about 100 dollars.

Output. The final tables and the forgetting table.

## 4. Assets and storage

Models (download sizes, bf16): Qwen3-VL-4B 9 GB; Qwen3-VL-8B 17.6 GB; Qwen3-VL-32B 66 GB (cloud volume only); Molmo2-4B 9 GB; Molmo2-8B 17 GB; InternVL3.5-8B 17 GB; Qwen3.5-9B 19 GB; Gemma4-12B 25 GB; SAM 3 about 2 GB; big-LaMa 0.2 GB; FLUX.2 klein 4B 8 to 16 GB or Mage-Flow-Edit-Turbo 4B about 8 GB; Grounding DINO 0.7 GB; TruFor and DINOv2-B under 1 GB; GRIT, TreeVGR, Rex-Thinker, DeepEyes, Chain-of-Focus about 15 GB each; Visual-RFT, VLM-R1, Time-R1 class models 5 to 15 GB each. Core for E91-0 to E91-3: about 110 GB. Everything for E91-4 and E91-5: about 250 GB.

Data: GroundingME 1 GB; OpenRef 2 to 5 GB; COCO train2014 plus val2014 19 GB (Ref-L4, Ref-Adv, gRefCOCO, COCO-Search18 all draw on it); Objects365 test subset 2 GB; SA-1B two to three shards 20 to 30 GB; OpenImages subset 10 to 20 GB; VisMin a few GB; Charades videos 55 GB (E91-5 only). Core about 60 GB; with the diagnostic arm about 120 GB.

Edit banks: evaluation bank 15 GB (local); training bank 150 to 250 GB (cloud volume).

Local disk: 1 TB NVMe covers the core models, the core data, the evaluation bank and outputs with room; 2 TB if every audited model and the Charades videos live locally at once.

## 5. Order and calendar (one person, local first)

1. Week 1: E91-0. SAM 3 access on day one; relation code and the control-edit sampler are the long pole; downloads in the background.
2. Week 2: E91-1 (one day) then E91-2 (two days of GPU, human check and API rows in parallel). Decision: does the audit have headroom at 8B, and does the frontier subset show any same-box re-prediction on clean removals?
3. Week 3: E91-3 locally or on one A100 (about 30 dollars); iterate E91-4 on two models locally; start E91-6 generation.
4. Week 4: E91-4 batch on two cloud cards (about 120 to 170 dollars, plus 200 to 400 dollars of frontier API calls); launch E91-7 annotation (about 500 dollars); E91-5 locally as filler.
5. Week 5: E91-8 locally once labels return; the CPU hashing box for E91-6 if full-corpus de-duplication is wanted. Decision: metric paper or audit paper.
6. Week 6 onward, only if the audit and metric hold: E91-9 (about 100 dollars plus the volume), E91-10 (350 to 450 dollars), E91-12; E91-11 (1,000 to 1,300 dollars) only if arm (c) clearly beats the base.

Total cloud spend to the audit-plus-metric decision: about 300 to 350 dollars of GPU time plus 700 to 1,000 dollars of API calls and annotation. With the training contribution: about 1,800 to 2,500 dollars more. Total local GPU time to the metric decision: about 130 to 200 hours.
