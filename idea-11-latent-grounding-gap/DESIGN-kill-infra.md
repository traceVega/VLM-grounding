# Design: infrastructure for the IDEA-11 kill experiments

Status: v1.0, 2026-09-02. Review converged after three rounds with three reviewers (Section 12); the conditions attached to the final verdicts are applied as written. Scope: E11-0 to E11-3 in `EXPERIMENTS.md` (setup, the 8B rescue-rate kill, the matched-label control heads, the 32B confirmation). Section 2 is the pre-registration text; the B0 tag makes it binding under the freeze rule D19.

Sources: `../team/ideas/IDEA-11-latent-grounding-gap.md` Sections 3.0, 3.1, 6 and 7; `../team/landscape/r2-practical.md` Sections 5.1, 5.2, 8.1, 8.11 and 8.14; `../team/PLAN.md` S1, S3, S5; `EXPERIMENTS.md` in this folder for machine specs; `../shared/harness/SPEC.md` v1.2 for the harness contract.

## 1. Purpose, scope and success criteria

Purpose. Build the smallest set of code, data and environment that computes the IDEA-11 kill statistics (rescue rate and 0.15-gap rate of internal read-outs on emitted-box failures) at 8B on the local RTX 5090 and at 32B on one rented card, plus the matched-label control that decides whether Stage 4 is dead, such that the stop rule can be evaluated with no choice made after seeing results.

In scope: environment, data pipeline, evaluation harness (shared with IDEA-91), read-out extraction, control heads, analysis, cloud parity for the 32B run, reproducibility.

Out of scope: Stage 2 probes on SA-1B, activation patching, the GUI column, other backbones, Stages 3 to 5.

Acceptance checks (all under non-kill run ids, see D19):
1. Harness validity. The vLLM server for 1a and 1b runs Qwen3-VL-8B-Instruct at the processor's default resolution with `max_model_len` at least 20,480 and default multimodal processor arguments; its arguments and its rejected-request count go into the manifest, and the count must be zero. (1a) The GroundingME repository's own evaluator against that server reproduces the paper's Table 3 overall score for the model (the reviewer's read is 31.0; the executor records the exact cell, paper version and repository commit, O6). If 1a misses by more than 2 points, the miss is attributed to the paper's undisclosed settings, the delta is printed in every kill table, and 1a does not gate. (1b) Our harness on the backend that serves the kill (Hugging Face generation for this idea), same model, same items, same default-resolution settings, must match 1a's number within 2 points; this is the gate, and the 1b manifest's `backend` field is `hf`. A vLLM-versus-HF parity record at the default override set is optional and recorded if made. (1c) Our harness at the kill resolution D4 is reported with the score delta and the change in failure count.
2. Hook validity. The rows captured by the registered attention function and the rows from `output_attentions=True`, both taken after the attention function's cast to bf16 and upcast to fp32 for comparison, differ by at most 4e-3 in absolute value, and the B0 boxes derived from both are identical, on five D2(b) items, in the exact kill configuration (gradients enabled, checkpointing on, train mode, vision encoder on SDPA).
3. Grid-mapping validity. On 100 synthetic images (one colored rectangle, random size 24 to 256 px on the longer side, random aspect 1:2 to 2:1, random position, plain background), 50 on a 1,920 by 1,280 canvas and 50 on 1,280 by 1,920 (exactly 2,457,600 px, so no resize occurs at D4 and `image_grid_thw` is (1, 80, 120) or (1, 120, 80) before merging), both read-outs reach IoU at least 0.5 times the item's grid ceiling (D5). A unit test with a deliberately transposed grid map must fail this check, and a 3:2 unit test checks the per-axis pixel mapping.
4. Regeneration. Every number in the kill table regenerates from the configs and the result store with one command, including the 32B rows from the synced files alone (the head-selection table, `heads.parquet` with the H1 and H2 boxes on F_c, the six head checkpoints, per-item read-out records and a sha256 list of the raw captures; raw captures are recomputable and not stored); manifests carry commit SHAs for code, model and dataset revisions, all seeds including the bootstrap seed, package versions, GPU name, the pre-registration version, and sha256 of every file in the run directory, written last.

## 2. Pre-registered definitions

Every constant below is fixed at v1.0. Dev slices (D2) are disjoint from kill sets by perceptual hash of the image.

| ID | Decision | Value | Why |
|---|---|---|---|
| D1 | Kill item sets | GroundingME items outside the Rejection dimension whose ground truth is one box; OpenRef positive items with exactly one ground-truth box (O1); every Ref-L4 small-bin item (D3) on an image not reserved by D2(a), COCO half and Objects365-test half kept apart. If a Ref-L4 half exceeds 6,000 eligible small-bin items, a seed-0 subsample of 6,000 is used and the deviation is stated. Count tables per set and bin, before and after the D2(a) reservation, are committed before freeze | The idea file's sets; multi-box items go to a set-aware appendix; the cap bounds compute and is declared |
| D2 | Dev slices | Reservation first: (a) 1,000 labelled Ref-L4 items drawn with seed 0 from images that carry no GroundingME or OpenRef item (by pHash), stratified by size bin to the histogram of the candidate pool (GroundingME single-box positives, OpenRef single-target positives and every Ref-L4 small-bin item, computed from ground truth alone), so small-bin heavy; 900 train, 100 early-stop; the images of (a) are reserved and excluded from D1. (b) 200 Ref-L4 COCO-half items in the medium and large bins on images disjoint from D1 and from (a), for smoke runs, parity checks and pipeline sanity only. Disjointness is asserted on pHash (Hamming distance at most 4) across every D1 set | (a) is the 1k budget shared by the attention read-out and the control heads and matches the distribution where the kill is decided; drawing it before D1 makes both constructible |
| D3 | Size bins | By ground-truth longer side in original pixels: tiny under 32, small 32 to 64, medium 64 to 128, large 128 to 256, xl 256 and above. Small bin for the kill is tiny plus small | Ref-L4 spans 30 to 3,767 px |
| D4 | Input resolution | A run-level override, recorded in the manifest, not a per-model field: `min_pixels = max_pixels = 2,457,600` for every kill run at both scales, so every image is resized to about 2.4 Mpx and about 2,400 visual tokens on Qwen3-VL's 32 px grid; the resize is anisotropic by up to a few percent because sides round to multiples of 32, so coordinates map through per-axis factors taken from `image_grid_thw`, and the scalar `downscale_factor_d4` is descriptive only. The processor's defaults are used only in acceptance check 1a and 1b. This setting differs from IDEA-91's, which caps at the same value without a minimum and so never upscales; the setting for the later joint audit table is decided in the audit design, not here | A fixed grid across items; the largest fixed setting that keeps the 8B gradient pass under about 26 GB and E11-1 near one day locally |
| D5 | Grid ceiling and F_c | Per item, `ceiling_iou_d4` is the best IoU with the ground truth of any box whose edges lie on the 32 px token grid of the resized image, computed from the ground truth alone at D4 with the per-axis mapping, before freeze; a table of the share of D1 items with ceiling at least 0.5, per set and bin, is committed with the count table. F_c is the subset of F with ceiling at least 0.5 | Below about 80 px on the resized side no grid-aligned box reaches IoU 0.5, so a rescue rate on unrestricted F is bounded by geometry |
| D6 | Prompt and decoding | GroundingME: its own prompt and parser verbatim. OpenRef and Ref-L4: "Locate the {expr} in this image and output its bbox coordinates in JSON format." Greedy, `max_new_tokens 64`, first `bbox_2d` parsed, coordinates 0 to 1000 relative to the resized image, mapped to original pixels per axis. Literal prompt strings live in `configs/prompts/` and their hashes in the manifest | One convention per model in config; never a best-IoU oracle |
| D7 | Failure sets | V-failure: unparseable output, abstention, or IoU(V, GT) below 0.5. F is the set of V-failures over D1; F_c per D5. Parse failures and abstentions carry IoU(V) equal to 0; their share is reported and every statistic is also given excluding them | The idea file's definition plus a guard against a parser bug |
| D8 | Anchor tokens | Primary: the last token of the assistant prefix in the presence prompt "Is there {expr} in this image? Answer yes or no.", the position whose logits produce the first answer token, one forward pass, no generation. Secondary: the position that produces the first digit of x1 in the grounding output, teacher-forced with the model's own output. The kill statistic uses the primary; the secondary is descriptive | The idea file names both |
| D9 | R_grad0 | At the primary anchor, `p_yes` is the softmax over the logits of the pinned token ids for "Yes" and "No" (the ids of " Yes", " No", "Yes", "No" under the pinned tokenizer are listed in `configs/tokens.yaml`; the two-way softmax uses the larger-probability variant of each); H is the binary entropy of `p_yes`. Gradient of H with respect to `pixel_values` after the processor. Layout: `pixel_values` has one row per 16 by 16 patch, channels times two temporal copies, rows ordered so that four consecutive rows form one 2 by 2 merge group; the relevance of a visual token is the L1 norm of the gradient over its four rows after per-channel normalization by the processor's std. The share of F whose argmax first token is neither pinned id is reported | Model-agnostic; no dependence on where DeepStack injects features |
| D10 | Box rule B0 | Min-max normalize the token map to [0, 1]; threshold by Otsu's method; keep the largest 4-connected component; box = its bounding rectangle in token coordinates, mapped to original pixels per axis. If the foreground covers more than 50% of tokens, fall back to B1. The share of items hitting the fallback is reported | Zero constants to tune |
| D11 | Box rule B1 | Threshold at the 85th percentile of token relevance; same component and box rule | Descriptive only |
| D12 | R_att1k | For every text-decoder layer and head, the attention row from the primary anchor to the visual tokens, reshaped to the grid. On the 1,000 dev pairs, score each head by the mean IoU of its B0 box with the ground truth; keep the top 3; the read-out map is their mean; B0 gives the box. Budget 1,000 labels. Re-selected independently at 32B | The few-heads paper's budget and k |
| D13 | Statistics | Rescue on i in F: IoU(R_i, GT_i) at least 0.5. Gap-0.15: IoU(R_i, GT_i) minus IoU(V_i, GT_i) at least 0.15. Chance rows: read-outs are run on every D1 item; the table reports Acc@0.5 of each read-out over all D1 items, the reverse rescue rate (V at IoU at least 0.5 on items where the read-out fails), and a permuted-box null in which each read-out's B0 boxes, in normalized image coordinates, are permuted among items within the same set and size bin and then scored | Chance rows show whether rescue exceeds what an independent localizer of the same accuracy would give; they do not alter D15 |
| D14 | Best read-out | For each statistic, the larger of R_grad0 and R_att1k; the bootstrap CI of the best re-selects the best inside every resample; both read-outs reported | Two pre-declared candidates |
| D15 | Stop rule | STOP the idea if, on F_c pooled over D1 at 8B, rescue(best) is below 0.15 and gap15(best) is below 0.20, and the same holds at 32B. Point estimates decide. Reported with them: 95% bootstrap CIs (2,000 resamples, clustered by image), the same statistics on unrestricted F, per-set and per-bin rows, per-set shares of F and F_c, and a set-balanced secondary rate. Secondary quantities (B1, the secondary anchor, the projector-gradient variant if run, the parse-excluded variant, the chance rows) are descriptive and cannot overturn the verdict. 32B fallback, pre-registered: if the pod's 50-item timing projects the full 32B read-out pass above 30 hours, the 32B read-outs run on every item of F plus a seed-0 stratified 2,000-item subsample of D1 for the chance rows, and the table says so | The idea file's rule on the items where rescue is geometrically possible |
| D16 | Control heads | Frozen features. H1 coordinate head: query is the primary anchor's hidden state at layer depth/2 (18 of 36 at 8B, 32 of 64 at 32B); keys and values are the LLM-input visual tokens; two cross-attention blocks (width 512, 8 heads) and an MLP to four normalized coordinates; L1 plus GIoU loss; about 6M parameters. H2 attention head: the same blocks producing a per-token map trained with a soft-mask BCE against the ground-truth box, box by B0. Training on D2(a): 900 pairs, 100 for early stopping, AdamW lr 1e-4, weight decay 0.05, at most 50 epochs, 3 seeds, mean reported | Same 1k label budget and distribution as R_att1k's selection set |
| D17 | Stage 4 stop | Let R* be the read-out with the larger mean IoU on F_c. Stage 4 is dropped if max over H1 and H2 of mean IoU(head, GT) on F_c is at least mean IoU(R*, GT) on F_c minus 0.02. Reported overall and per bin | "Within 2 IoU points" made computable |
| D18 | 32B settings | Identical D1 to D17; bf16; eager attention on the text decoder with capture, SDPA in the vision encoder; gradient checkpointing on; parameters frozen; `logits_to_keep=1`; the same 1,000 dev pairs; its own top-3 head selection | Scale is the only change |
| D19 | Freeze rule | v1.0 is frozen before any model runs on any D1 item, and after the ground-truth-only computations (counts, D2 reservation, D5 ceilings) exist. Before freeze only those computations and runs on D2(b) are allowed. Acceptance check 1 and the smoke run use non-kill run ids that analysis refuses to join into kill tables; the smoke run uses D2(b) only. After freeze no value in D1 to D18 changes except through the two declared contingencies: O1 (OpenRef absent), with its consequence written in Section 8, and the 32B timing fallback of D15, triggered only by the pod's 50-item timing; any other change forks the document to a new version and both versions' results are reported | The reason the document exists |

## 3. System overview

```
data/raw/<set>                     downloads, untouched, revision SHAs recorded
data/prepared/<set>/items.parquet  one row per item (D1, D2, D3, D5)
        |
        v
shared/harness  --run model_cfg item_set --override res-->  results/<run_id>/outputs.parquet
        |
        v
idea11/readouts --run model_cfg item_set-->  results/<run_id>/readouts/<item_id>.npz
        |
        v
idea11/heads    --train/eval-->              results/<run_id>/heads.parquet
        |
        v
idea11/analysis --kill run_ids-->            tables/kill_8b.md, tables/kill_32b.md, figures/
```

Code homes: `shared/` (environment, data pipeline, harness, result store, judge service later) is shared with IDEA-91; `idea11/` holds read-outs, heads and analysis. The harness contract is `../shared/harness/SPEC.md` v1.2 with `schema.py`; this document carries no copy and lists what it uses and adds in Section 5.

## 4. Components

### 4.1 Environment and runtime

- Runtime. The local host is Windows 11. Two supported runtimes: native Linux (dual boot) or WSL2 with Docker. Under WSL2: `.wslconfig` with `memory=28GB` and `swap=16GB` on NVMe; data, results and model caches on the WSL ext4 volume, never under `/mnt/c`; containers run with `--gpus all`. Idle VRAM taken by the display is measured at startup and the memory envelope is checked against the remaining amount. The Dockerfile in `shared/env/` (RunPod PyTorch base image, `uv` lockfile) builds the same image locally and on the pod.
- Python 3.11; PyTorch built for CUDA 12.8 or newer; transformers pinned to one commit that supports Qwen3-VL, the dict form of `attn_implementation`, and the attention-function registry; accelerate; timm; scikit-image and scipy; pyarrow; a vLLM build with Blackwell support, required for acceptance check 1 and for IDEA-91's Molmo2 path, optional as a harness backend after the parity rule in SPEC Section 5.
- Model loading: `torch_dtype=bfloat16`, `low_cpu_mem_usage=True`, `device_map="cuda"`; `attn_implementation={"text_config": "eager_capture", "vision_config": "sdpa"}` so the vision encoder never materializes attention probabilities; `gradient_checkpointing_enable()` on the whole model and `model.train()` during read-out passes (attention dropout is 0.0, asserted in code); parameters `requires_grad=False`; `logits_to_keep=1` on every read-out forward; `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True, warn_only=True)` with the tolerance in SPEC Section 6.
- Memory envelope at D4 (about 2,400 visual tokens plus about 60 text tokens) on 32 GB: 8B weights 17.6 GB; checkpointed layer inputs about 0.7 GB; one recomputed text layer during backward including its eager attention probabilities about 1.2 GB; vision encoder on SDPA with about 9,600 patches and no saved probabilities, provided SDPA selects a fused kernel on sm_120 (B5's peak test covers it); CUDA context and workspace about 2 GB; expected peak 22 to 26 GB. The 32B pass on a 96 or 141 GB card: 66 GB weights plus under 8 GB.
- The `output_attentions=True` fallback capture path needs about 14 GB of attention tensors at D4 at 8B and does not fit the 5090 next to the weights; if it is ever needed it runs at half resolution under a non-kill run id, or on the cloud card, and says so.

### 4.2 Data pipeline

| Set | Source and pin | Size | Prepared as |
|---|---|---|---|
| GroundingME | Hugging Face `lirang04/GroundingME` at a recorded revision SHA; repository `github.com/lirang04/GroundingME` at a recorded commit for its evaluator | about 3 GB; images 1,500 to 7,680 px | items with dimension and single-box flag; per-item D4 resize factors recorded |
| OpenRef | release of arXiv 2605.25706, revision SHA recorded; format and licence confirmed at download (O1) | 2 to 5 GB | single-target positives for D1 |
| Ref-L4 | Hugging Face annotations plus its bundled `images.tar.gz` (CC BY-NC 4.0), revision SHA recorded; no separate COCO or Objects365 download | about 21 GB | D3 bins, D2 reservation, D1 small bin |
| Synthetic grid-check set | generated per acceptance check 3 | under 0.1 GB | check 3 |

`items.parquet` follows SPEC Section 3. Splits are materialized with seed 0 and committed. `data/LICENSES.md` (single file, shared with IDEA-91) carries one row per asset: licence, URL, date read, allowed uses (evaluate, publish numbers, release images), attribution duty; the rows for Ref-L4 (CC BY-NC 4.0, re-ships Objects365 images), GroundingME (research use under SA-1B and HR-Bench terms) and OpenRef (to be read) exist before B2 closes.

### 4.3 Evaluation harness

Contract: SPEC v1.2 (model config, run contract, manifest, schemas, parity rule, determinism tolerance). This idea uses the ORIGINAL condition only and the run-level overrides `min_pixels`, `max_pixels` (D4) and `abstain_protocol` (primary). Prompts are versioned files whose hashes enter the manifest. Hugging Face batched generation with left padding is the default backend for Qwen3-VL; vLLM may serve a kill run only with a parity record per SPEC Section 5 under this idea's override set on dev slice (a).

### 4.4 Read-out extraction

Attention capture. An attention function `eager_capture` registered with the transformers registry wraps eager attention; when a capture context is active it stores the anchor row of the bf16 attention weights per text-decoder layer on the host and filters on the text attention class so the vision encoder is never captured. Capture is idempotent per (layer, item): the recompute pass under checkpointing overwrites rather than appends, and the item's record is asserted to hold exactly one row per layer. Visual token positions, the grid shape and the per-axis pixel factors come from `image_grid_thw` and the resized size. Acceptance check 2 runs in the exact kill configuration.

Gradient capture. `pixel_values.requires_grad_(True)`; forward with the presence prompt and `logits_to_keep=1`; H per D9 from the logits at the primary anchor; `backward`; aggregate per D9 into the token grid. The vision encoder is inside the backward pass.

Secondary anchor. Teacher-force the model's own grounding output truncated at the position before the first digit of x1, so that position is the last one in the sequence and `logits_to_keep=1` returns its logits; attention row there and the gradient of the entropy over the digit-token logits at that position; a second backward pass.

Box rules per D10 and D11, unit-tested on synthetic maps including the fallback.

Coverage. Read-outs run on every D1 item (D13), subject to the 32B fallback in D15.

Storage. Per item about 30 KB. Per dev pair, all layer-head rows for head selection: about 5.5 MB per item at 8B (5.5 GB total), about 20 GB total at 32B.

Throughput on the 5090 at D4: presence forward with capture about 0.8 s; gradient pass with checkpointing about 3 s; secondary anchor about 3 s; about 7 s per item; about 12,000 D1 items: about 23 hours, plus the plain inference over D1 (about 1.5 hours) and the dev pairs with full capture (about 1.5 hours). B5's 20-item test records the measured seconds per item, and every later budget is recomputed from it.

### 4.5 Matched-label control heads

Features are recomputed on the fly from the frozen model with an LRU cache; only the 1,000 dev pairs are cached to disk (about 20 GB at 8B, about 25 GB at 32B). Heads per D16; training takes minutes; evaluation on F_c is one forward per item. Results per seed and per bin feed D17.

### 4.6 Analysis

`idea11/analysis/kill.py` reads the run ids in `kill.yaml` (with `bootstrap_seed`), refuses run directories whose manifest lacks checksums or seeds or that are marked non-kill, joins outputs, read-outs and heads on `item_id`, computes D7, D13, D14, D15, D17 with bootstrap CIs clustered by image, and writes `tables/kill_<scale>.md` with: rescue and gap rates for R_grad0, R_att1k, best, H1, H2 on F_c and on F; the chance rows of D13; the ceiling table; the parse-failure share and the parse-excluded variant; per-set and per-bin rows and shares; the set-balanced rate; agreement A between V and each read-out; the acceptance-check 1 numbers and any 1a miss; the STOP verdict as one line from D15; and a pre-registered reading: if the reverse rescue rate is at or above the rescue rate, the result reads as two noisy localizers, not a latent gap. Figure 1: per-item scatter of IoU(V, GT) against IoU(R, GT) on the small bin and GroundingME, one panel per scale, rescue region shaded, the heads' operating points marked.

### 4.7 Cloud parity for E11-3

- Budget derivation. The 32B pass costs about 4 times the 8B pass per item (64 layers of width 5,120 against 36 of width 4,096). With the 8B pass measured at about 7 s per item on the 5090, and an H200 at about 5 times the 5090 on dense bf16 (an RTX PRO 6000 at about 1 time), the full 32B read-out pass over about 12,000 items is about 17 to 35 hours on the H200 (about 80 to 160 dollars of GPU time, 90 to 170 with staging) and about 90 to 100 hours on the PRO 6000 (about 200 dollars). The H200 is the recommended card; if H200s are out of stock, the fallback is an H100 NVL 94 GB at about H100 speed (roughly 20 to 40 hours, 65 to 130 dollars), and the PRO 6000 is the last resort. The pod's 50-item timing is written into the manifest before the full run and triggers the D15 fallback if it projects above 30 hours.
- Volume first. Create a 250 GB network volume in a datacenter that lists both H200 SXM and H100 NVL. Start a CPU pod on it: pull Qwen3-VL-32B-Instruct at the pinned revision (66 GB), rebuild `data/prepared` from the public sources with the B2 script, and assert `image_sha256` equality with the committed `items.parquet`; no bulk upload from the home uplink.
- GPU pod: 1 H200 SXM 141 GB (about 4.59 dollars an hour, medium stock); container from `shared/env/Dockerfile`; the same commands as local with `configs/qwen3vl-32b.yaml`; results synced back into local `results/` under the same run id with checksums verified after sync. Synced: `outputs.parquet`, per-item read-out records, the head-selection table, `heads.parquet`, the six head checkpoints, the manifest and a sha256 list of the raw captures; not synced: the raw captures and cached features, which are recomputable.
- Expected: 17 to 35 hours plus 1 to 2 hours of staging; about 90 to 170 dollars plus about 10 dollars a month for the volume.

### 4.8 Reproducibility

Immutable `results/<run_id>/` with `manifest.json` written last and carrying sha256 of every file in the directory; analysis verifies checksums before reading. Determinism per SPEC Section 6. The pre-registration version is in every manifest.

## 5. Interfaces

This idea reads and writes `items.parquet` and `outputs.parquet` exactly as SPEC v1.2 defines them. It adds:

`readouts/<item_id>.npz`: `grid_hw`, `axis_factors`, `anchor_primary_pos`, `anchor_secondary_pos`, `r_grad0_map`, `r_att_sel_map`, `r_grad0_secondary_map`, `r_att_secondary_map`, `boxes` (rule and read-out to `xyxy_px`), `ceiling_iou`, `otsu_fallback_used`, `out_of_pair_mass`.

`heads.parquet`: `run_id`, `item_id`, `head_id` (H1, H2), `seed`, `box_xyxy_px`, `iou_gt`.

`head_selection.parquet`: `run_id`, `layer`, `head`, `mean_iou_dev`, `selected`.

`tables/kill_<scale>.md`: the fields in 4.6 plus the pre-registration version and the run ids.

## 6. Storage and compute budget

| Item | Local 5090 | Cloud |
|---|---|---|
| Models | Qwen3-VL-8B 17.6 GB | Qwen3-VL-32B 66 GB on the volume |
| Data | about 26 GB prepared (Ref-L4 bundle, GroundingME, OpenRef) | rebuilt on the volume from public sources |
| Outputs | about 30 GB (dev captures 5.5 GB, dev features 20 GB, per-item read-outs under 1 GB) | about 50 GB on the volume; under 3 GB synced |
| GPU time | E11-1 about 26 h; E11-2 1 to 2 h | E11-3 17 to 35 h on H200 |
| Cash | 0 | 90 to 170 dollars plus about 10 a month for the volume |

## 7. Build plan

| Step | Deliverable | Effort | Test |
|---|---|---|---|
| B1 | `shared/env`: runtime set-up (Linux or WSL2 per 4.1), Dockerfile, lockfile | 1 to 2 days | one Qwen3-VL-8B forward at D4 in the container locally and on a pod; idle VRAM logged |
| B2 | `shared/data`: downloads with revision SHAs, `items.parquet`, the D2(a) reservation and D2(b), pHash disjointness assert, `LICENSES.md` rows | 2 days, mostly waiting | row counts committed; the assert passes; sha256 of every image recorded |
| B3 | `shared/harness`: SPEC v1.2 implemented, `schema.py`, configs, prompts, parsers, scoring, manifest; the ground-truth-only resize and ceiling code | 3 to 4 days | convention round-trip tests; parser fuzz; schema validation on write; the D5 table produced from ground truth alone |
| B0 | Freeze v1.0: commit the D1 and D2 count tables and the D5 ceiling table; create the pre-registration tag | half a day | tag created; no model run on D1 items exists |
| B4 | acceptance check 1 (1a, 1b, 1c) under non-kill run ids with the server settings of check 1 | half a day | numbers and the rejected-request count in the manifest |
| B5 | `idea11/readouts`: capture function, gradient pass, box rules, storage | 4 to 5 days | acceptance checks 2 and 3; capture idempotence assert; peak memory at D4 at most 26 GB on a 20-item run with backward; measured seconds per item recorded |
| B6 | `idea11/heads` | 1 to 2 days | overfits 20 items to IoU above 0.9; early stop triggers |
| B7 | `idea11/analysis` | 1 day | recomputes a hand-built toy table exactly, including the max re-selection in the bootstrap and the permuted-box null |
| B8 | smoke run on 50 D2(b) items end to end at 8B, non-kill run id | half a day | tables produced; manifest complete with checksums |
| B9 | E11-1 and E11-2 full runs | about 1.5 days of GPU | Section 2 verdicts written |
| B10 | E11-3 volume staging, pod 50-item timing, full run or the D15 fallback | 1 to 2 days | the 32B rows join the table; checksums verified after sync |

About 3.5 person-weeks to the 8B verdict, 4 to both scales.

## 8. Risks and mitigations

- Transformers internals move: pin one commit; acceptance check 2 is a permanent test; the `output_attentions=True` fallback exists at half resolution or on the cloud card.
- Checkpointing silently off in eval mode: train mode is enforced in the read-out pass and the dropout-is-zero assertion guards it.
- Otsu degenerates on flat maps: D10's fallback is fixed and its share reported.
- The grid ceiling: D5's F_c is the primary set; F is reported; tiny and small are separate bins.
- A parser bug inflates F: D7's parse-excluded variant.
- OpenRef unavailable or unsuitable (O1, the sole declared contingency): the kill runs on GroundingME plus Ref-L4, the count tables are re-committed without OpenRef, and every table says so.
- Acceptance check 1a misses the paper's number: the delta is printed; 1b gates.
- The 32B pass is slower than budgeted: the D15 fallback is pre-registered and triggered by the pod's 50-item timing.
- GroundingME images are downscaled at D4: expected; the per-item factors and the ceiling table make it visible.
- WSL2 memory cap or slow `/mnt/c` I/O: the runtime rules in 4.1; B1's test runs inside the container.

## 9. Open items

- O1 (open): OpenRef's release format and licence; whether single-target positives are derivable.
- O2 to O4 (closed in v0.2).
- O5 (open, low priority): a 5k-label column for the heads; not in scope for the kill.
- O6 (open until B4): the exact GroundingME Table 3 cell for Qwen3-VL-8B-Instruct and the paper version.
- O7 (new, deferred to the audit design): the resolution setting for the joint audit table shared with IDEA-91.

## 10. Not designed here on purpose

The SA-1B probe pool, the block-ablation cache, the IoU-GRPO copy, activation patching, Stage 3 decoding, Stage 5 signals, the GUI column and the Molmo2 and InternVL un-tiling.

## 11. Glossary

V: the emitted box. R: a read-out box. GT: ground truth. F, F_c: failure sets (D7, D5). B0, B1: box rules. H1, H2: control heads. Rescue, gap-0.15: D13. R*: D17.

## 12. Review log

Round 1, 2026-09-02. Three reviewers: systems (RS), methods (RM), data and operations (RD). Verdict before revision: NOT CONVERGED. Responses (v0.2):

- RS-11-1 (MAJOR, gradient pass OOM): ACCEPT. Dict-form attention implementation with SDPA in the vision encoder, train mode with a dropout-is-zero assertion, idempotent capture, a 26 GB peak test in B5; check 2 in the kill configuration.
- RS-11-2 (MAJOR, check 1 resolution): ACCEPT. Check 1 split into 1a, 1b, 1c under non-kill run ids.
- RS-11-3 (MAJOR, synthetic check): ACCEPT. Random positions, rectangles, both canvases, ceiling-relative criterion, transposed-map unit test.
- RS-11-4 (MINOR, tolerance): ACCEPT. fp32 comparison; `CUBLAS_WORKSPACE_CONFIG`.
- RS-11-5 (MINOR, pixel layout and token ids): ACCEPT. D9.
- RS-11-6 (MINOR, cost and volume): ACCEPT.
- RS-11-7 (MINOR, D17 and parity): ACCEPT. R* defined; parity in SPEC.
- RM-11-1 (BLOCKING, grid ceiling): ACCEPT. D5 and F_c; D4 raised to 2.4 Mpx. Partial REBUT on "the largest max_pixels the envelope allows": 2.4 Mpx chosen as the cost-benefit point (accepted by both reviewers in round 2).
- RM-11-2 (MAJOR, freeze before check 1): ACCEPT. D19 and B0.
- RM-11-3 (MAJOR, chance level): ACCEPT. Read-outs on all D1 items; chance rows; the pre-registered reading; D15 unchanged (accepted in round 2).
- RM-11-4 (MAJOR, dev-slice tilt): ACCEPT. D2(a) stratified; D17 per bin.
- RM-11-5 to RM-11-8 (MINOR): ACCEPT.
- RD-11-1 (MAJOR, check 1 executable): ACCEPT. Repository evaluator against our vLLM server; cell and commit recorded (O6).
- RD-11-2 (MAJOR, Ref-L4 bundle): ACCEPT. No COCO or Objects365 in this idea; sizes corrected; revisions pinned.
- RD-11-3 (MAJOR, reproducibility): ACCEPT. SHAs, seeds, checksums, parity, tolerance.
- RD-11-4 (MAJOR, staging): ACCEPT. Volume and CPU-pod staging.
- RD-11-5 (MAJOR, Windows runtime): ACCEPT. 4.1.
- RD-11-6, RD-11-7 (MINOR): ACCEPT.
- Cross-document RS-X-1, RS-X-2, RS-X-3, RM-X-1, RM-X-3, RM-X-4, RD-X-1, RD-X-2, RD-X-3, RD-X-4: ACCEPT. One contract in `shared/harness/SPEC.md`, one schema file, resolution as a run-level override, one validity run cited by both, one `LICENSES.md`, reconciled downloads.

Round 2, 2026-09-02. Verdicts on v0.2: RM CONVERGED (two minor), RS CONVERGED subject to minor fixes, RD NOT CONVERGED (two major). Responses (v0.3):

- RM-11-9 (MINOR, freeze clause): ACCEPT. D19 covers D1 and D2 with O1 as the sole contingency.
- RM-11-10 and RS-11-11 (MINOR, permuted null on differing grids): ACCEPT RS's form. D13 permutes B0 boxes in normalized coordinates within set and bin.
- RS-11-3 residual and RS-11-8 (MINOR, canvases and per-axis mapping): ACCEPT. Canvases are 1,920 by 1,280 and 1,280 by 1,920; per-axis factors from `image_grid_thw`; the scalar factor is descriptive (check 3, D4, D5, D10).
- RS-11-4 residual (MINOR, like-for-like tolerance): ACCEPT. Both rows compared after the bf16 cast, upcast to fp32, tolerance 4e-3 (check 2).
- RS-11-9 (MINOR, 1a server settings): ACCEPT. `max_model_len` at least 20,480, default processor arguments, server arguments and a zero rejected-request count in the manifest (check 1).
- RS-11-10 and RD-11-N1 (MINOR and MAJOR, 32B budget): ACCEPT. 4.7 derives the 32B budget from the measured 8B seconds per item times 4 and a stated speed ratio (17 to 35 hours on H200, 90 to 100 on the PRO 6000); the H200 is recommended; `logits_to_keep=1`; the D15 fallback is pre-registered and triggered by the pod's 50-item timing; Section 6 and `EXPERIMENTS.md` restated.
- RS-11-11 residual (MINOR, `output_attentions` fallback memory): ACCEPT. The fallback runs at half resolution under a non-kill id or on the cloud card (4.1, Section 8).
- RD-11-N2 (MAJOR, D2(a) reservation): ACCEPT. D2(a) is drawn first with seed 0, stratified to the candidate-pool histogram, its images reserved; D1 is defined on the remainder; both count tables committed at B0.
- RD-11-N3 (MINOR, 32B regeneration): ACCEPT. The head-selection table, six head checkpoints and a sha256 list of captures are synced; raw captures are recomputable (check 4, 4.7, Section 5).
- RD-11-N4 (MINOR, build order and the 1a-miss rule): ACCEPT. Build order is B1, B2, B3, then the B0 freeze, then B4; a 1a miss is printed and 1b gates (check 1, Section 7, Section 8).
- RS-X-4 and RM-X-5 (MINOR, "the same setting"): ACCEPT. D4 states the difference from IDEA-91's cap; the joint audit setting is deferred (O7); 4.3 cites SPEC Section 5 for parity without restating thresholds; SPEC v1.1 names the shared parity slice.

Round 3, 2026-09-02. Verdicts on v0.3: RS CONVERGED, RM CONVERGED conditional on RM-11-11 applied as written, RD CONVERGED. Responses (v1.0):

- RM-11-11 (MAJOR, backend of check 1b): ACCEPT as written. 1b runs on the Hugging Face backend that serves the kill, at the default resolution, against 1a's number within 2 points, and its manifest records `backend` equal to `hf`; a vLLM-versus-HF parity record at the default override set is optional (check 1).
- RM-11-12 (MINOR): ACCEPT. D19 lists both contingencies, O1 and the D15 timing fallback.
- RM-11-13 (MINOR): ACCEPT. The secondary-anchor forward is truncated at the anchor so it is the last position (4.4).
- RS-11-12 (MINOR): ACCEPT. `heads.parquet` is in the synced set (check 4, 4.7).
- RS-X-5 (MINOR, SPEC wording and `effective_resolution`): ACCEPT. SPEC v1.2 Section 2 says the YAML carries no override values and the manifest records `effective_resolution` read back from the processor.
- RD-11-N5 (MINOR, fallback card): ACCEPT. 4.7 names the H100 NVL fallback with hours and cash and creates the volume in a datacenter listing both cards; the cost wording is made consistent.
- Verdict after application: all three reviewers converged; v1.0.
