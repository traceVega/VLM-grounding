# IDEA-11 experiments: local RTX 5090 versus cloud

Date: 2026-09-02. Local machine: one RTX 5090 (32 GB VRAM) with 32 GB CPU RAM. Cloud: RunPod-class rental; prices below were queried on 2026-09-02.

Source: `../team/ideas/IDEA-11-latent-grounding-gap.md` Sections 3, 6 and 7 and `../team/PLAN.md`. H100-hour budgets are the team's. Memory peaks, 5090 wall-clock, cloud machine specs and dollar figures are my estimates, assuming bf16 weights, batch one for Hugging Face passes and vLLM for serving; treat them as within about 30%.

## 0. Rules of thumb for this machine

What fits on the 5090 (32 GB VRAM):
- Any model up to about 12B in bf16 for inference through vLLM (Qwen3-VL-8B 17.6 GB, Molmo2-8B about 17 GB, InternVL3.5-8B about 17 GB, Qwen3.5-9B about 19 GB, Gemma4-12B about 25 GB), one at a time, with `gpu_memory_utilization` at most 0.9, `max_model_len` 4096 and `swap_space` 2.
- Qwen3-VL-8B through Hugging Face with eager attention plus gradient read-outs, if `gradient_checkpointing` is on and parameters do not require grad: peak about 22 to 24 GB at up to 2.5k visual tokens. Without checkpointing the peak is 27 to 36 GB and GUI screenshots will not fit.
- Qwen3-VL-32B only in 4-bit (NF4 or a pre-quantized AWQ checkpoint, about 19 GB) with checkpointing: peak about 24 GB, roughly 2 to 3 times slower than bf16 on the same card, and quantization is a confound. Use it as a preliminary read, confirm on cloud.

What never fits locally:
- Any 27B or larger model in bf16 (Qwen3-VL-32B 66 GB, UI-Venus-2-27B 54 GB).
- Two 8B models resident at once (activation patching needs 35 GB).
- GRPO or SFT of a 4B or larger policy with EasyR1 (optimizer states alone exceed 32 GB; vLLM rollouts are colocated).

CPU RAM at 32 GB:
- Load with `low_cpu_mem_usage=True` and `device_map="cuda"`; safetensors shards stream and the 66 GB 32B checkpoint quantizes on the fly without needing 66 GB of RAM.
- Keep dataloader workers at 4 or fewer and stream read-out outputs to disk per item; a thousand raw attention tensors do not fit in RAM.
- Keep 100 GB of free NVMe for shard downloads and per-item caches; 1 TB total is comfortable for everything below.

Software for Blackwell (sm_120): PyTorch wheels built for CUDA 12.8 or newer, a vLLM build with Blackwell support, bitsandbytes 0.45 or newer for NF4, eager attention for the hooks (flash-attention is not needed and returns no weights anyway).

Conversion: 5090 wall-clock is about 3 times the H100-hour figure for these prefill-heavy inference jobs, about 2 times for decode-bound sampling, and 4 to 5 times for training, which is not done locally. An RTX PRO 6000 Blackwell 96 GB runs at about 5090 speed with three times the memory, which makes it the cheap cloud card for memory-bound jobs.

## 1. Cloud options (RunPod, queried 2026-09-02, USD per GPU-hour)

| GPU | VRAM | Secure | Community | Stock | Use it for |
|---|---|---|---|---|---|
| RTX PRO 6000 Blackwell, server edition | 96 GB | 2.09 | 1.69 | high | 32B inference and attention read-outs, two-model patching, any memory-bound single-card job; same software stack as the 5090 |
| H200 SXM | 141 GB | 4.59 | 3.59 | medium | 32B gradient read-outs on one card; fastest single card |
| H100 SXM | 80 GB | 3.29 | 2.69 | low | EasyR1 training nodes, throughput batches |
| H100 PCIe or NVL | 80 or 94 GB | 2.89 or 3.19 | 1.99 or 2.59 | low | single-card throughput when SXM is out |
| A100 80 GB, PCIe or SXM | 80 GB | 1.39 or 1.59 | 1.19 or 1.39 | low | cheapest single-card inference batches; no FP8 |

Dollar lines below use secure prices; community is about 20% lower. RunPod pods come with host RAM that scales with GPU count (roughly 100 to 250 GB per H100 or H200); ask for a network volume for anything that must outlive the pod.

## 2. Summary

| ID | Experiment | Where | Local wall-clock (5090) | Cloud machine | Cloud hours | About USD |
|---|---|---|---|---|---|---|
| E11-0 | Setup, harness, hooks | LOCAL | engineering, about 1 person-week | none | 0 | 0 |
| E11-1 | Rescue-rate kill at 8B | LOCAL | about 26 h at the 2.4 Mpx setting with read-outs on every item (design D4, D13) | 1 H100 80 GB (optional, for speed) | 6 to 8 | 20 to 26 |
| E11-2 | Matched-label control heads | LOCAL | 1 to 2 h | none | 0 | 0 |
| E11-3 | Rescue-rate kill at 32B | CLOUD | about 100 h at 4-bit, preliminary only | 1 H200 141 GB (the PRO 6000 is about 5 times slower here) | 17 to 35 | 90 to 170 plus about 10 a month for the volume |
| E11-4 | Audit rows, other 8B backbones and GUI sets | LOCAL, long | 45 to 60 h | 1 H100 80 GB (for speed) | 15 to 20 | 50 to 65 |
| E11-5 | GUI right-box-wrong-reason column | LOCAL plus cloud or API for the 27B row | 12 to 15 h | 1 RTX PRO 6000 96 GB for UI-Venus-2-27B | 2 to 3 | 5 to 7, plus 100 human check |
| E11-6 | Layer-wise probes at 8B; 32B probes on cloud | LOCAL, 32B on CLOUD | 15 to 25 h | 1 H200 or 1 RTX PRO 6000 for the 32B pass | 5, or 12 | 23, or 25 |
| E11-7 | Gate 1 headroom, Gate 2 presence probe | LOCAL | 5 to 6 h | none | 0 | 0 |
| E11-8 | 8B IoU-GRPO copy for patching | CLOUD | not possible | 4 H100 SXM 80 GB (or 4 H200, or 4 RTX PRO 6000) | 15 wall, 60 GPU-h | 200 (H100), 250 to 330 (PRO 6000) |
| E11-9 | Activation patching base versus GRPO copy | CLOUD | not possible | 1 H100 80 GB or 1 RTX PRO 6000 96 GB | 10, or 25 | 33, or 52 |
| E11-10 | Stage 3 test-time exploitation | LOCAL | 15 to 30 h | none | 0 | 0 |
| E11-11 | Stage 5 selective grounding, AURC | LOCAL | 30 to 45 h | 1 H100 80 GB (for speed) | 10 to 15 | 33 to 50 |
| E11-12 | Stage 4 gated training minimum at 4B | CLOUD | not possible | 8 H100 SXM 80 GB node (5 used: 4 train, 1 teacher and judge) | 25 wall, 100 GPU-h | 350 to 450 |
| E11-13 | Forgetting add-on per trained checkpoint | LOCAL for 4B and 8B checkpoints | 6 h per checkpoint | run on the training node instead | 2 per checkpoint | 7 per checkpoint |

Local-only path to a decision: E11-0, E11-1, E11-2, then E11-7 and E11-6, about 55 to 70 hours of 5090 time and no cloud spend. The 32B confirmation (E11-3) costs about 90 to 170 dollars and one to two days on an H200 and is the first cloud job worth buying. The design document DESIGN-kill-infra.md (v0.3) supersedes the E11-1 and E11-3 lines below where they differ: the kill runs at a fixed 2.4 Mpx, read-outs cover every item for the chance rows, and the 32B budget is derived from the measured 8B seconds per item.

## 3. Experiments in order

### E11-0 · Setup, harness and read-out hooks · LOCAL

Purpose. Everything below reads the same tables. Build the evaluation harness with one coordinate convention per model (0 to 1000 for Qwen3-VL, absolute pixels for Qwen2.5-VL, 0 to 100 for Molmo2, `<locXXXX>` for PaliGemma) fixed in a config, never a best-IoU-over-conventions oracle in a reported number; the contamination lines of `../team/landscape/r2-practical.md` Section 8.14; per-layer eager-attention hooks that keep only the rows of the probe token and the first coordinate token; the entropy-gradient read-out with one fixed box rule (largest connected component above a fixed mass quantile, chosen once); the 4 by 4 block-ablation map with control blocks.

Data to download (about 55 GB): GroundingME (1,005 items), OpenRef, Ref-L4 annotations plus COCO train2014 and val2014 images (about 19 GB) and the Objects365 test images it uses (about 2 GB for 3,233 images), gRefCOCO no-target annotations (COCO images), ScreenSpot-Pro (1,581 items), OSWorld-G (564 items), Ref-Adv 2026.

Models to download (about 18 GB now, more later): Qwen3-VL-8B-Instruct. Later: Qwen3-VL-32B-Instruct (66 GB, cloud only, or a 19 GB AWQ checkpoint locally), Molmo2-8B, InternVL3.5-8B, Gemma4-E4B, Qwen3-VL-4B.

Local spec. GPU RAM any; CPU RAM 32 GB is fine; storage 100 GB free. Time: about one person-week of engineering.

### E11-1 · Rescue-rate kill at 8B (KE-1, part 1) · LOCAL

Decides. Whether internal read-outs of the same forward pass localize the target where the emitted box misses it, at 8B.

Stop rule. Rescue rate below 15% and 0.15-IoU-gap rate below 20% at 8B is half of the kill; the idea stops only if E11-3 shows the same at 32B.

Procedure. Run REC with the native prompt over GroundingME positive items (exclude rejection items), OpenRef single-target items (multi-target scored separately with a set-aware read-out) and the Ref-L4 small-object bin with the COCO and Objects365-test halves kept apart. On every emitted-box failure (IoU below 0.5) compute R_grad0 (zero labels) and R_att1k (heads and threshold chosen on 1,000 disjoint Ref-L4 pairs). Report rescue rate, 0.15-gap rate, agreement A, stratified by size and GroundingME dimension.

Data. GroundingME, OpenRef, Ref-L4 small bin (subsample to 5k items if the bin is larger), 1k held-out Ref-L4 pairs.

Models. Qwen3-VL-8B-Instruct, Hugging Face, eager attention, gradient checkpointing on, parameters frozen.

Local spec. GPU RAM peak 22 to 24 GB (17.6 GB weights plus checkpointed backward at up to 2.5k visual tokens); CPU RAM about 12 GB in use; storage 15 GB for per-item read-out masks and the 1k-item attention rows used for head selection (about 12 MB per item); wall-clock 8 to 12 hours (about 15k prompts of inference, then 5k failures times two read-outs at 2 to 3 s each).

Cloud alternative. 1 H100 SXM 80 GB, 64 GB host RAM, 80 GB disk, about 3 hours, about 10 dollars. Only worth it if the local card is busy.

Output. The failure table with both read-outs per item; the size-binned gap table that IDEA-21 pre-registers against; hooks reused by every later experiment.

### E11-2 · Matched-label control heads (KE-1, part 2) · LOCAL

Decides. Whether a supervised head trained on the same 1,000 labels matches the read-out, in which case the finding is "a small probe beats the text decoder" and Stage 4 is dropped.

Stop rule. Head within 2 IoU points of the best read-out on the E11-1 failure set: drop Stage 4; the paper is the audit and the decoder question handed to IDEA-21.

Procedure. From the E11-1 forward passes, store projected visual tokens and the mid-layer hidden state of the presence-probe token for the 1k training pairs and all evaluation items. Train a coordinate-regression head and a GUI-Actor-style attention head, each on the 1k labels, on frozen features. Score them on the same failure items as the read-outs.

Data. The 1k pairs and the E11-1 item set. Models. Frozen Qwen3-VL-8B features; two heads of about 100M parameters.

Local spec. GPU RAM up to 16 GB; CPU RAM 16 GB; storage 10 to 20 GB of cached features (8 to 20 MB per item); wall-clock 1 to 2 hours.

Output. The control column of Table 1.

### E11-3 · Rescue-rate kill at 32B (KE-1, part 3) · CLOUD

Decides. Whether the gap survives scale. The kill fires only if both 8B and 32B fail the thresholds.

Procedure. Same as E11-1 on Qwen3-VL-32B-Instruct in bf16, same item set, same box rule, R_att1k re-selected on the same 1k pairs; add layer-wise probes at the presence token while the passes run (feeds E11-6).

Cloud spec, option A. 1 H200 SXM 141 GB; GPU RAM 66 GB weights plus about 30 GB activations without checkpointing (with checkpointing about 72 GB, so an 80 GB card is tight and a 96 or 141 GB card is comfortable); host RAM 128 GB or more; disk 150 GB (66 GB weights, 55 GB data, outputs); about 6 hours; about 28 dollars.

Cloud spec, option B. 1 RTX PRO 6000 Blackwell 96 GB; same memory need, 2 to 3 times slower; 15 to 20 hours; 35 to 40 dollars; high stock.

Local preliminary. Qwen3-VL-32B in NF4 or AWQ (about 19 GB) with checkpointing: peak about 24 GB, about 40 hours, quantization confound; use only to get an early read while waiting for a cloud card.

Output. The 32B rows of Table 1; the two-scale kill verdict.

### E11-4 · Audit rows for the other backbones and the GUI sets · LOCAL, long

Purpose. Stage 1 across lineages: Molmo2-8B (no RefCOCO annotations in its documented mixture; points, so REC is scored point-in-box; overlapping crops need overlap-weighted un-tiling), InternVL3.5-8B (tiles stitched with per-tile normalization), Gemma4-E4B, plus Qwen3-VL-8B on ScreenSpot-Pro and OSWorld-G at about 2.5k visual tokens, and Ref-Adv 2026 for all. Gradient read-outs additionally on one Qwen3.5 hybrid (no attention matrix in its linear layers).

Data. As E11-1 plus ScreenSpot-Pro, OSWorld-G, Ref-Adv 2026. Models. Molmo2-8B, InternVL3.5-8B, Gemma4-E4B, Qwen3.5-4B or 9B (about 60 GB of downloads).

Local spec. GPU RAM up to 26 GB with checkpointing; CPU RAM 16 GB; storage 60 GB; wall-clock 45 to 60 hours, so two to three days of continuous runs.

Cloud alternative. 1 H100 SXM 80 GB, 64 GB host RAM, 200 GB disk, 15 to 20 hours, 50 to 65 dollars; sensible if the local card is needed for E11-6 and E11-7 at the same time.

Output. The remaining rows of Table 1, including the clean-lineage row on Molmo2.

### E11-5 · GUI right-box-wrong-reason column with IDEA-91's operators · LOCAL plus cloud or API for one row

Purpose. The lead placed the GUI audit here. On ScreenSpot-Pro and OSWorld-G, remove the predicted element by flat-UI pixel inpainting, add a matched control edit on a non-target element, apply the text-side controls, and report necessity, necessity-plus, invariance and text pass rates next to the gap column; no sufficiency judge on GUI. Run IDEA-91's K1 gate on flat-UI holes first (cheap on flat UI). Tag relative-position expressions and mark items UNRESOLVED when the descriptor no longer resolves after removal.

Data. ScreenSpot-Pro, OSWorld-G with element boxes; a 200-item human plausibility check of removals (about 100 dollars).

Models. Qwen3-VL-8B, GUI-Actor-7B, MolmoPoint-GUI-8B locally; UI-Venus-2-27B (54 GB bf16) on a cloud card or by API; OpenCV or LaMa for the inpainting; ResNet-18 for the gate.

Local spec. GPU RAM up to 20 GB; CPU RAM 8 GB; storage 10 GB; wall-clock 12 to 15 hours.

Cloud spec for the 27B row. 1 RTX PRO 6000 96 GB (or 1 H100 80 GB), 64 GB host RAM, 100 GB disk, 2 to 3 hours, 5 to 7 dollars.

Output. The GUI column of Table 1; code credited to IDEA-91.

### E11-6 · Stage 2 layer-wise probes on SA-1B with SAM 3 instances · LOCAL, with the 32B pass on CLOUD

Purpose. Where in the network the location is decodable. Train linear probes from the presence-probe token's hidden state at every layer to the SAM 3 box, on a clean pool, and evaluate on the audit sets by size bin.

Procedure. Take 20k SA-1B images (two shards, about 20 GB), run a captioner for a 5 to 10 noun list per image, run SAM 3 concept prompts (100k to 200k forwards), keep instances as proxy boxes, run Qwen3-VL-8B with the presence-probe prompt and store the probe token's hidden state at every layer (36 layers times 4096 times 2 bytes, about 0.3 MB per item, 6 GB for 20k), fit ridge probes per layer, evaluate on E11-1's items.

Data. SA-1B shards (SAM 3 licence covers derived instances), the audit sets. Models. SAM 3 (gated checkpoint, about 2 GB), Qwen3-VL-8B, a captioner (Qwen3-VL-8B itself is fine).

Local spec. GPU RAM up to 20 GB; CPU RAM 16 GB; storage 35 GB (shards, instances, states); wall-clock 15 to 25 hours.

Cloud spec for the 32B probes. Same card as E11-3 (1 H200 or 1 RTX PRO 6000), 150 GB disk, about 5 hours on H200 or 12 on the PRO 6000, about 25 dollars either way; best run in the same pod session as E11-3.

Output. Probe IoU per layer by size at both scales; the SA-1B pool with SAM 3 instances that E11-7 and E11-12 reuse.

### E11-7 · Gate 1 headroom and Gate 2 presence-probe check · LOCAL

Decides. Whether Stage 4 is allowed to run at all.

Gate 1. On the E11-6 pool, headroom g equals max(0, IoU(read-out, proxy box) minus IoU(emitted, proxy box)); Stage 4 runs only if the mean is at least 0.05 IoU; report the same on the human-labelled audit sets, since the two can differ.

Gate 2. On gRefCOCO no-target and GroundingME rejection items against matched positives, the statistic is the referent-mask presence drop minus a matched-size control-mask drop, from a 4 by 4 block-ablation map (16 masked prefills per prompt, controls are the other blocks); AUROC for groundability must reach 0.7, else the necessity term and the presence-head labels are replaced by IDEA-91's edit-pair negatives.

Data. The E11-6 pool; gRefCOCO no-target; GroundingME. Models. Qwen3-VL-8B.

Local spec. GPU RAM up to 20 GB; CPU RAM 8 GB; storage 5 GB; wall-clock 5 to 6 hours (about 40k prefills at 0.5 s).

Output. The two gate values; the block-map cache format used by E11-12's reward.

### E11-8 · 8B IoU-GRPO copy for the cause test · CLOUD

Purpose. Activation patching needs an IoU-GRPO copy of the same 8B base (VLM-R1 recipe, about 300 steps). Not a kill, but the cause test in Stage 2 cannot run without it. Under the contamination protocol, prefer the SA-1B pool with SAM 3 proxy boxes from E11-6 over RefCOCO-family training data, and say which was used.

Cloud spec. 4 H100 SXM 80 GB (EasyR1, FSDP plus colocated vLLM); host RAM 256 GB; disk 250 GB on a network volume (pool 30 GB, five checkpoints of 17.6 GB, logs); about 15 wall-hours, 60 GPU-hours; about 200 dollars secure, 160 community. If H100 SXM stock is out: 4 H200 (same hours, about 275 dollars) or 4 RTX PRO 6000 96 GB over PCIe (30 to 40 wall-hours, 250 to 330 dollars). Add the forgetting add-on (E11-13) on the same node before it is released.

Output. The GRPO checkpoint and its evaluation on RefCOCO (appendix), LISA-Grounding, Ref-Adv 2026 and GroundingME.

### E11-9 · Activation patching and the OOD probe-versus-GRPO comparison · CLOUD

Decides. H1 versus H2: does a mid-layer probe or the zero-label read-out match IoU-GRPO out of distribution, and does GRPO reduce the emitted-minus-read-out gap (decoder learned to read the map) or raise read-out IoU (perception changed)?

Procedure. Load base and GRPO copy, swap layer blocks following 2602.12395, score on LISA-Grounding, Ref-Adv 2026 and GroundingME; in-domain RefCOCO is not used for this claim.

Cloud spec. 1 H100 SXM 80 GB or 1 RTX PRO 6000 96 GB (two 8B models resident, 35 GB, plus hooks); host RAM 64 GB; disk 100 GB; about 10 hours on H100 (33 dollars) or 25 on the PRO 6000 (52 dollars). Local is not viable: 35 GB of weights exceeds the card, and swapping layers from a memory-mapped second checkpoint through 32 GB of host RAM is possible but not worth the engineering for a 30-dollar job.

Output. The Stage 2 cause-test table.

### E11-10 · Stage 3 test-time exploitation, label-free · LOCAL

Purpose. Read-out-constrained coordinate decoding (a logits processor biasing coordinate tokens toward the read-out window) and agreement-based selection among N sampled boxes, against ACS-Free and SD-RPN's RoI predictor; step-0 reported, no oracle selection.

Data. GroundingME, OpenRef, Ref-L4 small bin. Models. Qwen3-VL-8B; ACS-Free and SD-RPN code and checkpoints.

Local spec. GPU RAM up to 24 GB; CPU RAM 12 GB; storage 10 GB; wall-clock 15 to 30 hours (N equals 8 samples per item through vLLM, then read-outs on the chosen).

Output. The label-free gain table.

### E11-11 · Stage 5 selective grounding and AURC · LOCAL

Purpose. Actions box, multiple, none. Signals: coordinate-token likelihood (free, from vLLM logprobs), verbalized confidence, sample dispersion (8 samples), agreement A, presence head, a Brier-trained confidence, and the best logit, sampling and hidden-state families from the 27-method GUI uncertainty benchmark. Metric AURC and risk-coverage on GroundingME, OpenRef, Ref-Adv 2026 and ScreenSpot-Pro, thresholds calibrated on a 10k subsample of Ref-L4. The join with IDEA-91's audit table (necessity-plus failures as a second abstention target) is inference-free once that table exists.

Stop rule. Agreement does not beat token likelihood on AURC on two of four benchmarks.

Local spec. GPU RAM up to 24 GB; CPU RAM 12 GB; storage 15 GB; wall-clock 30 to 45 hours (about 5k items times 8 samples plus read-outs plus calibration runs).

Cloud alternative. 1 H100 SXM 80 GB, 64 GB host RAM, 100 GB disk, 10 to 15 hours, 33 to 50 dollars.

Output. The AURC table and the risk-coverage curves; the negative-result variant (does IoU-GRPO degrade calibration) uses E11-8's checkpoint.

### E11-12 · Stage 4 gated training minimum at 4B · CLOUD

Runs only if E11-2 did not stop it and both gates in E11-7 passed.

Arms. (a) label-free reward: agreement with the frozen teacher's precomputed read-out, control-masked necessity from the cached 4 by 4 block map, crop sufficiency online, anti-inflation penalty, format penalty; two seeds. (b) SD-RPN-style RoI head on the same read-out pseudo-labels; one seed. (e) the pseudo-label incumbent: SFT then IoU-GRPO on SAM 3 boxes at equal data; two seeds. Policy Qwen3-VL-4B, teacher Qwen3-VL-8B, 20k prompts from the E11-6 pool, about 500 steps of 128 prompts.

Cloud spec. One 8 H100 SXM 80 GB node: 4 cards for EasyR1, 1 card for the frozen teacher serving online sufficiency calls through vLLM, the rest idle or a second seed in parallel; host RAM 256 GB or more; disk 400 GB on a network volume (pool 20 GB, block-map cache 30 GB, checkpoints 9 GB each, keep at most 3 per run); about 100 GPU-hours, 25 wall-hours with two seeds side by side; 350 to 450 dollars secure. Precompute the teacher read-outs and the block map before the pod is rented (13 to 15 H100-hours, or about 40 hours locally on the 5090, which is the cheaper route).

Output. Acc at 0.5, 0.75 and 0.9 on Ref-L4 by size, box-area drift, achieved gain against the Gate 1 bound, the bypass comparison, and the forgetting table.

### E11-13 · Forgetting add-on for every trained checkpoint · LOCAL for 4B and 8B checkpoints

Purpose. MMMU, MMBench and POPE before and after every RL run; the team reports this as a contribution because no grounding-RL paper does.

Local spec. GPU RAM up to 20 GB; CPU RAM 8 GB; storage 5 GB per benchmark set; about 6 hours per checkpoint on the 5090, or 2 GPU-hours on the training node before it is released (about 7 dollars per checkpoint), which is the better place for it.

## 4. Assets and storage

Models (download sizes, bf16 unless noted): Qwen3-VL-8B 17.6 GB; Qwen3-VL-4B 9 GB; Qwen3-VL-32B 66 GB (cloud) or a 19 GB AWQ build (local preliminary); Molmo2-8B 17 GB; InternVL3.5-8B 17 GB; Gemma4-E4B about 16 GB; Qwen3.5-9B 19 GB; SAM 3 about 2 GB; GUI-Actor-7B 15 GB; MolmoPoint-GUI-8B 17 GB; UI-Venus-2-27B 54 GB (cloud). Core set for E11-0 to E11-7: about 80 GB.

Data: GroundingME 1 GB; OpenRef 2 to 5 GB; COCO train2014 plus val2014 19 GB; Objects365 test subset 2 GB; Ref-L4, gRefCOCO, Ref-Adv annotations under 1 GB; ScreenSpot-Pro 1.5 GB; OSWorld-G 1 GB; SA-1B two shards 20 GB; LISA-Grounding a few GB. Core about 55 GB.

Outputs: read-out masks, features and hidden states 50 to 80 GB across the local experiments.

Local disk: 1 TB NVMe is comfortable; 500 GB is workable if the 32B checkpoint stays on the cloud volume.

## 5. Order and calendar (one person, local first)

1. Week 1: E11-0. Start downloads on day one; the hooks are the long pole.
2. Week 2: E11-1 and E11-2 locally (two days of GPU); rent one H200 or PRO 6000 session for E11-3 and the 32B half of E11-6 (one afternoon, about 55 dollars). Decision: is the idea alive, and is Stage 4 already dead?
3. Weeks 3 to 4: E11-6, E11-7, E11-5 locally; E11-4 as background runs; write the audit table.
4. Week 5: E11-8 then E11-9 on cloud (about 250 dollars total); E11-10 and E11-11 locally in parallel.
5. Week 6 onward: E11-12 only if the gates passed; E11-13 on the training node.

Total cloud spend to the full core without Stage 4: about 350 dollars. With Stage 4: about 800 dollars. Total local GPU time: about 150 to 200 hours.
