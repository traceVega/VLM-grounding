# Pins: name, revision, file sha256

Design 4.1: every addition is "pinned by name, revision and file sha256". A row
with `PIN_REQUIRED` is not yet obtainable here; `shared/harness/model_config.py`
and `shared/harness/prompts.py` refuse to serve an unpinned artefact to a run
that is not `--non-kill`, so an unpinned row cannot reach a kill table.

Last updated 2026-09-02.

## Runtime

| Item | Version | Notes |
|---|---|---|
| Host | Windows 11 Pro 26200, WSL2 Ubuntu (kernel 6.6.114.1) | `.wslconfig` memory=24GB, swap=8GB (see DEVIATIONS D-3) |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MB, driver 591.86, sm_120 | idle VRAM with the display attached: ~3,466 MB, measured by `shared/env/probe.py` |
| Python | 3.12.13 (uv-managed, `~/vlmg-env`) | DEVIATIONS D-1 |
| torch | 2.14.0+cu130 | Blackwell wheels from the pytorch cu130 index |
| timm | 1.0.29 | P6 classifiers |
| kornia | 0.8.3 | imported by LaMa's `spatial_transform` module; the generator itself does not use it |
| transformers | 5.16.1 | ships `Sam3Model`/`Sam3Processor` natively |
| numpy / pyarrow / pandas | 2.5.2 / 25.0.1 / 3.0.5 | |
| opencv-python-headless | 5.0.0.93 | window, compositing, JPEG ladders |
| scikit-image / scipy | 0.26.0 / 1.18.1 | |
| pycocotools | installed | RLE for `edits/index.parquet` |
| vLLM | PIN_REQUIRED for this env | 0.28.0 is proven on this host in `~/ptr1-env`; pin the build used for the K2 runs |

## Editing stack

LaMa generator code is vendored at `advimman/lama@786f5936` under `idea91/edits/vendor/` (Apache-2.0); see that directory's `NOTICE.md`.

| Item | Source | Revision | sha256 | Status |
|---|---|---|---|---|
| SAM 3 | HF `facebook/sam3` | `3c879f39826c281e95690f02c7821c4de09afae7` | per-file, in the HF cache | **downloaded and verified 2026-09-03.** `Sam3Model` + `Sam3Processor` from transformers 5.16.1 (the repo config declares `Sam3VideoModel`, but `Sam3Model` loads the same weights for the image path). 3.44 GB safetensors; `sam3.pt` excluded. Measured: concept prompt IoU 0.99 on a synthetic target, 0 instances for an absent concept, box prompt IoU 0.99, ~0.2 s per prompt, **peak 2.13 GB VRAM**. Licence tag `other`; design O7 still wants the text read before anything derived is released |
| SAM 3.1 | HF `facebook/sam3.1` | `daa63191845a41281374e725f4c9e51c7a824460` | -- | **not pinned**: ships only `sam3.1_multiplex.pt` (3.50 GB), no `model.safetensors`, so `Sam3Model.from_pretrained` cannot read it. See DEVIATIONS D-16 |
| SAM 2.1 checkpoint | `~/sam2_ckpts/sam2.1_hiera_large.pt` | on disk | PIN_REQUIRED (compute at first use) | the design's O6 fallback for class-agnostic masks and its SAM 3 contingency; 898,083,611 bytes |
| GroundingME dataset | HF `lirang04/GroundingME` | `78e3c7974b2b1db0ea52266969e40f664a38e330` | per-file, in the HF cache | **downloaded 2026-09-03.** 1,005 test items, 2.9 GB: 804 positive single-box, 201 `Rejection` with a null `bbox`. `bbox` is absolute `xyxy` in original pixels; `detection_type` is the head noun. Licence `other` -- images are research-licensed and never published (P17, P11) |
| GroundingME evaluator | `github.com/lirang04/GroundingME` | `6867f7a045596d29cd77a14a01d22d4a68d64fb6` (2026-08-04) | -- | **read 2026-09-03**, not vendored. Source of P12's primary instruction, copied byte for byte from the `PROMPT` constant at `lmms_eval_task/groundingme/utils.py:95` into `configs/prompts/grounding_qwen3vl_primary.txt`. Its scorer's best-of-four coordinate decode is deliberately *not* adopted; see OPEN-QUESTIONS Q-3 |
| Qwen3-VL-8B-Instruct | HF `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | per-file, in the HF cache | **downloaded 2026-09-03**, 17 GB. Coordinate convention still `PIN_REQUIRED` (Q-3): absent from the card and both config files, so it needs a GPU probe |
| Molmo2-8B | HF `allenai/Molmo2-8B` | `e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b` | per-file, in the HF cache | **downloaded 2026-09-03**, 33 GB. Pointing instruction and abstention forms still to read (Q-2) |
| Gemma4-12B (edit verifier, P10) | HF `google/gemma-4-12b-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | per-file, in the HF cache | **downloaded 2026-09-03**, 23 GB. Pinned in `shared/judges/judges.yaml` as `judge_b` |
| Qwen3.5-9B (reward/crop judge, parser) | HF `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | per-file, in the HF cache | **downloaded 2026-09-03**, 19 GB. Pinned in `shared/judges/judges.yaml` as `judge_a`; the lineage rule keeps it off the edit-verifier role |
| big-LaMa | `big-lama.zip` from the URL in the upstream README (`huggingface.co/smartywu/big-lama`) | archive sha256 `f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6` | `best.ckpt` sha256 `fccb7adffd53ec0974ee5503c3731c2c2f1e7e07856fd9228cdcc0b46fd5d423` | **downloaded and verified 2026-09-03.** Apache-2.0. FFC-ResNet, 51.1 M params, loaded from the original Lightning checkpoint (no TorchScript). Measured: 0.029 s per 512x512 fill after load, **0.45 GB VRAM**. `VLMG_LAMA_DIR` points at the unpacked `big-lama` directory; the checkpoint sha256 goes into every index row |
| OpenCV Telea | opencv 5.0.0.93 | -- | -- | **not kill-grade**; pipeline exercise and a known-dirty editor for the gate |

## Models under test and judges

| Item | HF path | Revision | Status |
|---|---|---|---|
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | PIN_REQUIRED | coordinate convention also PIN_REQUIRED (OPEN-QUESTIONS Q-3) |
| Molmo2-8B | `allenai/Molmo2-8B` | PIN_REQUIRED | crop policy read back from the processor (P19); Molmo2-4B is already in the host HF cache |
| Qwen3.5-9B (judge A) | `Qwen/Qwen3.5-9B` | PIN_REQUIRED | reward/crop judge, noun-phrase parser |
| Gemma4-12B (judge B) | `google/gemma-4-12b-it` | PIN_REQUIRED | the edit verifier of P10 |

## Gate probes and adversaries (P6)

| Item | Source | Status |
|---|---|---|
| ResNet-18 | timm `resnet18.a1_in1k` | ImageNet weights, downloaded on first use |
| ViT-S/16 | timm `vit_small_patch16_224.augreg_in21k_ft_in1k` | created with `dynamic_img_size` + `dynamic_img_pad` (DEVIATIONS D-9) |
| DINOv2-B | timm `vit_base_patch14_dinov2.lvd142m` | strong adversary, one seed, reported |
| Forensic detector | PIN_REQUIRED | TruFor if obtainable under research terms, else PSCC-Net or MVSS-Net class; CAT-Net is excluded (it reads JPEG DCT streams; the windows are PNG) |

## Benchmarks and pools

| Item | Source | Revision | Status |
|---|---|---|---|
| Open Images (K1 pool) | validation split: `validation-images-with-rotation.csv` (2018_04), `validation-annotations-bbox.csv` + `class-descriptions-boxable.csv` (v5), images from the CVDF S3 mirror | pinned 2026-09-03 | **metadata downloaded (38 MB), pool selected (10,000 images, mean 4.12 class labels each, 84.6% with >= 2 distinct classes)**. All 41,620 validation images are CC BY 2.0. Images (~3 GB) not yet fetched. See DEVIATIONS D-19 |
| SA-1B shards | Meta release | PIN_REQUIRED | K1 fallback pool; makes that bank non-releasable |
| GroundingME | HF `lirang04/GroundingME` | PIN_REQUIRED | K2 (P8); its evaluator repo commit is also needed for Q-1 |
| OpenRef | release of arXiv 2605.25706 | PIN_REQUIRED | design O1: format and licence unread |
| Ref-L4 dev slice (a) | IDEA-11 `shared/data` | PIN_REQUIRED | the parity run of check 4 |
| gRefCOCO no-target + COCO train2014 | annotations + images | PIN_REQUIRED | the `p12` dev slice |
| COCO-Search18 | dataset's own images and present/absent lists | PIN_REQUIRED | P18; COCO 2017 instance annotations for the secondary statistic |
