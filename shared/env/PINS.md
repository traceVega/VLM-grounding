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
| numpy / pyarrow / pandas | 2.5.2 / 25.0.1 / 3.0.5 | |
| opencv-python-headless | 5.0.0.93 | window, compositing, JPEG ladders |
| scikit-image / scipy | 0.26.0 / 1.18.1 | |
| pycocotools | installed | RLE for `edits/index.parquet` |
| vLLM | PIN_REQUIRED for this env | 0.28.0 is proven on this host in `~/ptr1-env`; pin the build used for the K2 runs |

## Editing stack

| Item | Source | Revision | sha256 | Status |
|---|---|---|---|---|
| SAM 3 (repo) | PIN_REQUIRED | PIN_REQUIRED | -- | gated; design O7 (licence text behind the gate) |
| SAM 3 checkpoint | PIN_REQUIRED | PIN_REQUIRED | PIN_REQUIRED | gated |
| SAM 2.1 checkpoint | `~/sam2_ckpts/sam2.1_hiera_large.pt` | on disk | PIN_REQUIRED (compute at first use) | the design's O6 fallback for class-agnostic masks and its SAM 3 contingency; 898,083,611 bytes |
| big-LaMa (TorchScript `big-lama.pt`) | PIN_REQUIRED | PIN_REQUIRED | PIN_REQUIRED | `VLMG_LAMA_PT` points at it; `idea91/edits/inpaint.py` hashes it into every index row |
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
| OpenImages subset | official downloader by image id + images CSV | PIN_REQUIRED | K1 pool (P1); the CSV carries the attribution columns |
| SA-1B shards | Meta release | PIN_REQUIRED | K1 fallback pool; makes that bank non-releasable |
| GroundingME | HF `lirang04/GroundingME` | PIN_REQUIRED | K2 (P8); its evaluator repo commit is also needed for Q-1 |
| OpenRef | release of arXiv 2605.25706 | PIN_REQUIRED | design O1: format and licence unread |
| Ref-L4 dev slice (a) | IDEA-11 `shared/data` | PIN_REQUIRED | the parity run of check 4 |
| gRefCOCO no-target + COCO train2014 | annotations + images | PIN_REQUIRED | the `p12` dev slice |
| COCO-Search18 | dataset's own images and present/absent lists | PIN_REQUIRED | P18; COCO 2017 instance annotations for the secondary statistic |
