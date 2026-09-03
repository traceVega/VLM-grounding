# r2-practical.md — Practical landscape for VLM grounding research
Author: Researcher2 (Builder / Experimentalist). Date: 2026-09-01. Task T1.3.
Budget assumption throughout: 1–8 rentable H100/A100-class GPUs for weeks. Evidence labels per CHARTER.md.

Cost anchor: H100 rental in 2026 is roughly $2–4 per GPU-hour (RunPod H100 PCIe $1.99 community / $2.39 secure, H100 SXM $2.99; Lambda $3.29–3.99; Vast.ai from $1.49) [VERIFIED: https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison , https://www.runpod.io/gpu-models/h100-pcie]. So 200 GPU-hours ≈ $400–800 and a 1,000 GPU-hour project ≈ $2–4k. Everything below is sized against that.

## 0. TL;DR — the ten practical facts that matter most

1. The open-model frontier moved a lot in 2026: Qwen3.5 (Feb, natively multimodal, 4B/9B/27B dense + MoE, Apache-2.0), Gemma 4 (Apr, Apache-2.0, E2B–31B), Qwen3.6 (Apr), MolmoPoint (Mar), Muse Glimmer-30B (Meta, Aug), Qwen3.8-27B (Aug). Qwen2.5-VL is now two generations old; any paper built on it will be asked "does this hold on Qwen3.5 / Gemma 4?" [VERIFIED: sources in §1].
2. RefCOCO/+/g is saturated and contaminated: InternVL3.5-241B avg 92.4, Qwen3.5-27B avg 90.9, even 4B models ≥88; 14% / 24% / 5% of RefCOCO / + / g test labels are wrong (Ref-L4 paper); Qwen3-VL's report explicitly lists RefCOCO/+/g as training data [VERIFIED: arXiv 2508.18265, HF Qwen3.5-27B card, arXiv 2406.16866, arXiv 2511.21631]. Reporting RefCOCO gains as a headline is no longer a paper.
3. Still-discriminative benchmarks for open ≤10B models: ScreenSpot-Pro (single-pass: Qwen3-VL-8B 54.6, MolmoPoint-GUI-8B 61.1, Qwen3-VL-235B 62.0; with zoom-in in tech reports ≈ 73; community leaderboard with multi-step zoom agents ≈ 81–83; the 85–88 numbers on llm-stats are self-reported and unverified), OSWorld-G (Qwen3-VL-8B 58.2; MolmoPoint-GUI-8B 70.0; MAI-UI-32B + zoom 70.9), RefSpatial-Bench (2B RFT 52/54/42; Gemini-2.5-Pro placement 24), MMSI-Bench (open ≈ 30%, humans 97%), Point-Bench (MolmoPoint-8B 70.7 vs human 89.1), HallusionBench (0.70), Charades-STA R@0.7 (best ≈ 53, specialist; MLLMs lag at boundary precision), Ref-L4 (≈ 82 for 2024–25 models, one third-party 90.3), HumanRef, GSEval, RefBench-PRO [VERIFIED per §4 after Verifier corrections V-077..V-098].
4. RL for grounding is commoditised: EasyR1 (Qwen2/2.5/3-VL, 7B GRPO on 4×40 GB), TRL GRPOTrainer (Qwen2.5-VL, Gemma 3, SmolVLM2, vLLM colocate), ms-swift 4.0 (300+ multimodal models incl. Qwen3-VL, InternVL3.5, GLM-4.5V, Gemma 4). A 3B–8B GRPO run of ~1k steps is a 1–2 day job on 4–8 H100s [VERIFIED READMEs; time estimate LIKELY].
5. Coordinate conventions are a minefield: Qwen2-VL 0–1000, Qwen2.5-VL absolute pixels in the smart_resize'd frame, Qwen3-VL back to 0–1000, InternVL 0–1000 with `<box>`, PaliGemma `<locXXXX>` (1024 bins), Molmo percent floats in XML, Gemini `[ymin,xmin,ymax,xmax]` 0–1000, Gemma 4 JSON on a 1000×1000 grid, GLM normalized xyxy inside special tokens, Rex-Omni 0–999 tokens, MolmoPoint patch/subpatch/location tokens [VERIFIED/LIKELY per §5.1]. Both VLM-R1 and LLaMA-Factory have shipped grounding baselines with rescaling bugs.
6. Evaluation-protocol drift is as large as method effects: Qwen2.5-VL-7B on Charades-STA is 43.6 mIoU (vendor), 39.3 (TimeLens re-run), 29.46 (lmms-eval issue #857) [VERIFIED]. Any video/GUI number without the exact protocol is noise.
7. Reward hacking in grounding RL is documented and specific: hit-rewards shrink boxes, IoU-rewards inflate them (GUI-G1); Gaussian rewards (GUI-G2) and box-size constraints are the standard fixes; long CoT does not help GUI grounding [VERIFIED: arXiv 2505.15810, 2507.15846].
8. The cleanest "uncontaminated" open bases for controlled studies are the Molmo family (PixMo-only data) and possibly Gemma 4 (training data undisclosed); verify before relying on it [LIKELY].
9. Small test sets dominate the field: ReasonSeg test 779, RefSpatial-Bench 277, Where2Place 100, OSWorld-G 564, Point-Bench 982, HallusionBench 1,129. A 2-point gain is often inside the 95% CI; seeds and CIs are mandatory.
10. Out of reach at our scale: any pretraining, SFT on ≥5M grounding samples (Rex-Omni 22M, UGround 10M, RefSpatial 20M), full fine-tuning ≥27B, RL on ≥32B, proprietary data, and large human annotation (>10k items). Everything in §6 fits in ≤200 GPU-hours.

## 1. Open models usable for grounding research

Legend: FT = fine-tuning. "RefCOCO in train?" = whether the RefCOCO family was used in the model's own training (contamination).

### 1.1 General VLMs (2025–2026)

| Model (release) | Sizes | Licence | Coordinate output | RefCOCO in train? | Key grounding numbers | FT feasibility at our budget |
|---|---|---|---|---|---|---|
| Qwen2.5-VL (Feb 2025) | 3B / 7B / 32B / 72B | 7B, 32B Apache-2.0; 3B Qwen Research licence; 72B Qwen licence [LIKELY] | Absolute pixels in the smart_resize'd image, JSON `{"bbox_2d":[x1,y1,x2,y2]}` / `point_2d` [VERIFIED: https://github.com/QwenLM/Qwen2.5-VL/issues/866 ; https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/13] | Undisclosed: the report names RefCOCO only in evaluation tables and describes grounding data as "publicly available datasets and proprietary data" plus Grounding DINO / SAM synthesis [VERIFIED: Verifier V-004, arXiv 2502.13923] | 7B: RefCOCO val/testA/testB 90.0/92.5/85.4 [VERIFIED: search snippet of arXiv 2502.13923] | 3B/7B full FT on 2–4 H100; 32B LoRA on 2; 72B LoRA on 4+ |
| Qwen3-VL (Oct–Nov 2025) | 2B, 4B, 8B, 32B dense; 30B-A3B, 235B-A22B MoE; Instruct + Thinking | Apache-2.0 [VERIFIED: https://github.com/QwenLM/Qwen3-VL] | Normalized [0,1000] ("relative coordinates"), boxes and points; reverted from 2.5-VL's absolute pixels [VERIFIED: https://arxiv.org/html/2511.21631] | Yes: COCO, Objects365, OpenImages, RefCOCO/+/g + Grounding-DINO/Qwen2.5-VL synthetic + PixMo points [VERIFIED: arXiv 2511.21631 data section] | RefCOCO-avg (Instruct / Thinking): 235B 91.9 / 92.1; 32B 91.9 / 91.1; 30B-A3B 89.7 / 89.3; 8B 89.1 / 88.2; 4B 89.0 / 88.2; 2B 85.6 / 84.8; GPT-5 (high) 66.8 and Gemini 2.5 Pro 74.6 in the same table. ScreenSpot-Pro: 235B 62.0 / 61.8, 32B 57.9, 8B 54.6, 4B 59.5. OSWorld-G: 235B 66.7 / 68.3, 32B 65.1, 8B 58.2. Charades-STA mIoU: 235B 64.8 / 63.5, 8B-Thinking 59.9. ODinW-13 48.6 mAP (235B); Omni3D 3D boxes beat Gemini-2.5-Pro on SUN RGB-D by 5.2 [VERIFIED: Verifier V-005..V-007 from the PDF text of 2511.21631; no per-split RefCOCO table exists] | 2B/4B/8B full FT feasible; 32B LoRA |
| Qwen3.5 (Feb 16 2026) | 4B, 9B, 27B dense; 35B-A3B, 122B-A10B, 397B-A17B MoE | Apache-2.0 [VERIFIED: https://huggingface.co/Qwen/Qwen3.5-4B , https://huggingface.co/Qwen/Qwen3.5-27B] | Not stated on the cards; read from chat template / tokenizer (open item) | Not disclosed [SPECULATION: almost certainly yes given the Qwen3-VL mix] | 4B: RefCOCO avg 88.1, CountBench 96.3, ERQA 54.0, RefSpatial-Bench 54.6, ScreenSpot-Pro 60.3, OSWorld-Verified 35.6. 27B: RefCOCO avg 90.9, ODinW-13 41.1, CountBench 97.8, ERQA 60.5, RefSpatial-Bench 67.7, ScreenSpot-Pro 70.3, OSWorld-Verified 56.2, AndroidWorld 64.2 [VERIFIED: HF cards] | 4B/9B full FT ok; 27B LoRA only. Hybrid Gated-DeltaNet + MoE architecture, natively multimodal (early fusion): check framework support (EasyR1 via EasyVideoR1 claims Qwen3.5 support [VERIFIED: arXiv 2604.16893 abstract]; TRL/ms-swift unverified) |
| Qwen3.6 (Apr 2026) / Qwen3.8-27B (Aug 13–14 2026) | 3.6: 27B, 35B-A3B; 3.8: 27B (image + video) | Apache-2.0 [VERIFIED: https://huggingface.co/Qwen/Qwen3.6-27B ; https://www.latent.space/p/ainews-qwen-38-max24t-and-27b-new] | as Qwen3.5 [SPECULATION] | not disclosed | Qwen3.6-27B tops a third-party CountBench board at 97.8 [LIKELY: benchlm.ai]. Qwen3.8-Max open weights are text-only under a custom licence [VERIFIED: latent.space snippet] | LoRA only at 27B |
| InternVL3.5 (Aug 2025) | 1B, 2B, 4B, 8B, 14B, 38B, 241B-A28B | weights Apache-2.0 / code MIT [LIKELY; the CC-BY-4.0 in the paper is the arXiv licence] | Normalized 0–1000, `<box>[[x1,y1,x2,y2]]</box>` [LIKELY: InternVL2.5/3 convention] | Yes [LIKELY: InternVL2.5/3 data lists include RefCOCO] | RefCOCO-series avg: 1B 81.4, 2B 85.5, 4B 89.4, 8B 89.7, 14B 90.1, 30B-A3B 90.9, 38B 89.1, 241B 92.4. ScreenSpot-v2: 4B 85.1, 8B 86.2, 241B 92.9. OSWorld-G: 4B 33.9, 8B 36.4, 241B 53.2 [VERIFIED: https://arxiv.org/html/2508.18265v1] | 1B–8B full FT; 14B/38B LoRA. No InternVL4 as of 2026-09; InternVL-U (Mar 2026) is a 4B unified understanding + generation model [VERIFIED: search] |
| Molmo 2 (Dec 2025) | 4B, 8B (Qwen3 LLM); Molmo2-O 7B (Olmo, fully open) | Apache-2.0 with third-party academic/non-commercial dataset caveats [VERIFIED: https://allenai.org/blog/molmo2] | Points as text (x, y percentages in XML); video points with timestamps; track IDs | Not mentioned; Molmo 1 was PixMo-only [LIKELY] | Largest gains on Point-Bench, PixMo-Count, CountBenchQA; beats Gemini 3 Pro on video tracking (blog claim) | 4B/8B full FT feasible; 9 new datasets (>9M examples: Molmo2-VideoPoint 300k queries / 160k videos, Molmo2-VideoTrack 15k queries, Molmo2-MultiImagePoint 470k) [VERIFIED] |
| MolmoPoint (Mar 18 2026) | 8B, GUI-8B, Vid-4B | open (models, code, data) [VERIFIED: https://allenai.org/blog/molmopoint] | No text coordinates: coarse-to-fine `<PATCH>`, `<SUBPATCH>`, `<LOCATION>` tokens selecting visual features (3 tokens per point instead of 8) | as Molmo 2 | Point-Bench 70.7 (Molmo 2: 68.7); PixMo-Points F1 89.2 (85.2); ScreenSpot-Pro 61.1 and OSWorld-G 70.0 (claimed open SOTA at release); MolmoPoint-GUISyn 36k screenshots / 2M points [VERIFIED: blog] | 8B full FT on 4 H100; the non-text head is a natural testbed for coordinate-representation studies |
| Gemma 4 (Apr 2 2026) | E2B (2.3B eff.), E4B (4.5B eff.), 12B unified encoder-free, 26B-A4B MoE, 31B dense | Apache-2.0 [VERIFIED: https://huggingface.co/blog/gemma4] | Native JSON detection; coordinates on a 1000×1000 grid relative to the input | Not disclosed | No RefCOCO / ScreenSpot numbers in the launch blog → a cheap first-eval contribution | E2B/E4B/12B full FT feasible; TRL/PEFT/bitsandbytes supported; variable image token budget 70–1120 |
| PaliGemma 2 (Dec 2024) | 3B / 10B / 28B at 224/448/896 px | Gemma licence [LIKELY] | `<locXXXX>` 1024-bin tokens; `<segXXX>` mask tokens | Yes [LIKELY] | — | 3B full FT on 1 GPU; classic controlled-study base |
| Florence-2 (2024) | 0.23B / 0.77B | MIT [LIKELY] | 1000-bin location tokens | Yes via FLD-5B [LIKELY] | — | trivially cheap; good for probing studies |
| GLM-4.1V-9B-Thinking / GLM-4.5V (106B-A12B) / GLM-4.6V | 9B; 106B-A12B; 4.6V size unverified | MIT [LIKELY] | normalized xyxy box wrapped in `<|begin_of_box|>` tokens [VERIFIED: snippet of arXiv 2507.01006] | trained with RLCS on grounding; reports Ref-L4 and TreeBench [VERIFIED snippet] | claims ~10% over open SOTA incl. Ref-L4-test (4.5V) [LIKELY] | 9B full FT on 4 H100; 106B LoRA marginal |
| Kimi-VL-A3B (16B total) / Kimi K2.5 (Jan 2026, ~1T MoE) | A3B; K2.5 too large | MIT [LIKELY] | GUI-focused grounding data [VERIFIED snippet] | unknown | — | A3B LoRA feasible; K2.5 inference-only |
| MiMo-VL-7B (2025) / MiMo-V2-Flash (309B-A15B, Dec 2025) | 7B; 309B | MIT [LIKELY] | strong GUI grounding [VERIFIED snippet] | unknown | — | 7B feasible; V2 no |
| Muse Glimmer-30B (Meta, Aug 2026) | 29.6B dense + 1.8B ViT-G/14 encoder | Apache-2.0 [VERIFIED: https://huggingface.co/meta-models/Muse-Glimmer-30B ; VentureBeat] | unknown | unknown (distilled from closed Muse Spark) | ScreenSpot-Pro 75.4 = best open score on a third-party board [VERIFIED: https://llm-stats.com/benchmarks/screenspot-pro ; provenance of the number unknown] | LoRA only |
| LLaVA-OneVision (0.5B/7B/72B) / OneVision-1.5 | 0.5B, 7B, 72B | Apache-2.0 [LIKELY] | 0–1 floats [LIKELY] | Yes in the single-image stage [LIKELY] | — | cleanest fully-open data recipe for "we control the mix" studies |
| SmolVLM2 (256M / 500M / 2.2B) | tiny | Apache-2.0 [LIKELY] | none native [LIKELY] | unknown | — | supported by TRL GRPO [VERIFIED: TRL docs]; ideal for many-seed sweeps |
| ZAYA1-VL-8B-A1B (May 2026) | 8B MoE, 1B active | unverified | pointing in XML during pretraining → 0–1000 boxes/points in SFT [VERIFIED: https://arxiv.org/html/2605.08560] | unknown | RefCOCO avg 84.3 (val 88.0 / testA 91.0 / testB 83.5); Point-Bench 58.0 (= MolmoE-8B-A1B) [VERIFIED] | cheap (1B active) |

### 1.2 Grounding-specialist open models

| Model | Base | What it adds | Numbers | Notes |
|---|---|---|---|---|
| Sa2VA (Jan 2025; 1B/4B/8B/26B) | InternVL2.5/3 + SAM2 | `[SEG]` token → SAM2 masks for images and video | LSVOS/MeViS 2025 winners built on it (SaSaSa2VA, Sa2VA-i) [VERIFIED: arXiv 2509.16972, 2509.19082] | ByteDance repo also hosts SAMTok (CVPR-26) [VERIFIED: github.com/bytedance/Sa2VA] |
| Seg-Zero-7B (Mar 2025) | Qwen2.5-VL-7B + SAM2 | pure GRPO (no CoT data), decoupled reasoner / segmenter | ReasonSeg zero-shot gIoU 57.5 (+18% vs LISA-7B) [VERIFIED: arXiv 2503.06520 abstract]; trained on RefCOCOg-9K [VERIFIED: repo] | 7B training needs "4×80 GB or 8×46 GB" [VERIFIED: repo]; follow-up VisionReasoner (multi-object, multi-task, May 2025) |
| Rex-Omni (Oct 2025, 3B) | Qwen2.5-VL-3B [LIKELY] | 0–999 quantized coordinate special tokens; 22M SFT samples then GRPO with geometry-aware rewards (box accuracy, de-duplication) | zero-shot COCO/LVIS comparable to Grounding DINO (claim) [VERIFIED: arXiv 2510.12798 abstract] | the 22M SFT is out of our reach; the RL stage is not |
| RexSeek (ICCV 2025) | MLLM + detector | multi-instance person referring; HumanRef benchmark | SOTA on HumanRef [VERIFIED: github.com/IDEA-Research/RexSeek] | |
| UniVG-R1 (May 2025) | Qwen2.5-VL-7B [LIKELY] | GRPO with cold-start + difficulty-aware weighting | best average on RefCOCO/+/g among compared, biggest gain on RefCOCOg [VERIFIED snippet: arXiv 2505.14231] | |
| Perception-R1 (Apr 2025) | Qwen2-VL-2B [LIKELY] | RL for perception policies | RefCOCO 89.1 / + 81.7 / g 85.7 [VERIFIED snippet: arXiv 2504.07954] | |
| VLM-R1 (Apr 2025) | Qwen2.5-VL-3B/7B, InternVL | GRPO for REC; RL beats SFT out-of-domain on LISA-Grounding [VERIFIED: repo] | | last major updates Aug 2025 |
| RoboRefer-2B/8B (NeurIPS 2025) | NVILA-2B/8B [LIKELY] | depth-aware encoder + SFT then RFT for spatial referring; RefSpatial 20M QA, 31 relations, ≤5 reasoning steps | see §4 | 2B RFT beats 8B SFT on Unseen [VERIFIED: arXiv 2506.04308 Table 2] |
| TimeLens-7B/8B (Dec 2025) | Qwen2.5-VL-7B / Qwen3-VL-8B | re-annotated VTG data (TimeLens-100K) + recipe | Charades-STA R1@0.5 55.6 (7B) / 63.0 (8B), mIoU 48.8 / 55.2; ActivityNet R1@0.5 51.0 / 58.4 [VERIFIED: https://arxiv.org/html/2512.14698] | base Qwen2.5-VL-7B measured at R1@0.5 37.8, mIoU 39.3 |
| GUI-Actor-2B/7B (2025) | Qwen2-VL / Qwen2.5-VL | attention-based coordinate-free action head | ScreenSpot-Pro 40.7 (Qwen2-VL) / 44.6 (Qwen2.5-VL backbone) [VERIFIED snippet: microsoft.github.io/GUI-Actor] | |
| GUI-G1 / GUI-G2 (2025) | Qwen2.5-VL-3B/7B | reward-design fixes for grounding RL | see §5.4 | |
| Qwen3-VL-Seg (May 2026) | Qwen3-VL | open-world referring segmentation | RefCOCO 82.3/83.7/79.1, + 76.2/80.2/70.8, g 78.2/78.1 (mask cIoU) [VERIFIED snippet: arXiv 2605.07141] | |
| Parallel Tube Decoding (Aug 2026, 4B) | 4B backbone | parallel spatial box decoding for STVG; 79× lower latency, 92× throughput vs autoregressive on VidSTG (claim) [VERIFIED: arXiv 2608.28192 abstract] | | |

### 1.3 Fine-tuning memory rules of thumb
- LLaMA-Factory reference table (text LLMs; add 20–60% for image tokens [SPECULATION]): 7B full FT AdamW bf16 ≈ 120 GB, pure-bf16 ≈ 60 GB, LoRA ≈ 16 GB, QLoRA-4bit ≈ 6 GB; 14B: 240/120/32/12 GB; 30B: 600/300/64/24 GB [VERIFIED: https://github.com/hiyouga/LLaMA-Factory README].
- EasyR1 reference: Qwen2.5-VL-7B GRPO full FT bf16 ≈ 4×40 GB; LoRA ≈ 2×32 GB [VERIFIED: https://github.com/hiyouga/EasyR1].
- Community: Qwen2.5-VL-7B LoRA SFT with image grids needs ≥18 GB and runs on a 24 GB RTX 4090 at ~20 GB [VERIFIED: labellerr / spheron blog posts; low authority].
- Seg-Zero-7B (VLM + SAM2 + rollouts): 4×80 GB or 8×46 GB [VERIFIED: repo].
- Implication: on 1×H100 you can LoRA-train anything ≤14B and full-FT ≤4B; on 4×H100 you can GRPO a 7–9B model with 8 rollouts; 27B+ is LoRA-only; 72B+ RL is out.

## 2. Training frameworks for SFT and RL on VLMs

| Framework | SFT | RL algorithms | VLMs supported | Rollout engine | Maturity / gotchas |
|---|---|---|---|---|---|
| EasyR1 (verl fork, LLaMA-Factory team) | none (explicitly not provided) | GRPO, DAPO, GSPO, CISPO, Reinforce++, ReMax, RLOO | Qwen2-VL / Qwen2.5-VL / Qwen3-VL (+ Qwen3.5 via the EasyVideoR1 contribution); InternVL / Kimi-VL not listed | vLLM ≥ 0.8.3 (Docker image pins vLLM 0.11.0, torch 2.8, CUDA 12.9) | De-facto engine behind Seg-Zero, VisionReasoner, GUI-G1, EasyVideoR1 and many 2025–26 grounding-RL papers [LIKELY]. VLMs incompatible with Ulysses sequence parallelism; padding-free training and checkpoint resume supported; transformers ≥ 4.54, flash-attn ≥ 2.4.3 [VERIFIED: https://github.com/hiyouga/EasyR1] |
| verl (upstream) | via recipes | full menu | Qwen2.5-VL and others [LIKELY] | vLLM / SGLang | heavier config surface; use EasyR1 unless you need Megatron or multi-turn tool agents [SPECULATION] |
| TRL GRPOTrainer (Hugging Face) | SFTTrainer handles VLMs | GRPO (+ DPO etc.) | tested: Qwen2-VL, Qwen2.5-VL, Gemma 3, SmolVLM2, LLaVA-NeXT; "compatibility with all VLMs is not guaranteed" | vLLM colocate or server | Simplest path for ≤4B on 1–2 GPUs; works with PEFT + 4-bit; slower per step than EasyR1 [LIKELY]. Dataset = prompt + PIL image(s) [VERIFIED: https://huggingface.co/docs/trl/main/en/grpo_trainer]. Gemma 4 explicitly supported in TRL/PEFT [VERIFIED: HF Gemma 4 blog] |
| ms-swift 4.0 (ModelScope, Mar 3 2026) | LoRA / QLoRA / full; multimodal packing (+100% speed claim); mixed text/image/video/audio | GRPO with vLLM colocate; RLHF family | "300+ multimodal models" incl. Qwen3-VL, Qwen3-Omni, InternVL3.5, Ovis2.5, GLM-4.5V, Gemma 4, LLaVA, Phi-4 | vLLM / SGLang / LMDeploy | Broadest model coverage and Megatron multimodal training; fast-moving API, partly Chinese docs; GRPO less battle-tested for grounding than EasyR1 [features VERIFIED: https://github.com/modelscope/ms-swift ; assessment LIKELY] |
| LLaMA-Factory | LoRA / QLoRA / full; DPO / KTO | no native VLM GRPO (points to EasyR1) | Qwen2-VL/2.5-VL/3-VL, QVQ, Qwen3-Omni, InternVL2.5–3.5, Intern-S1-mini, Gemma 3, PaliGemma 1/2, Llama 3.2-V / Llama 4, LLaVA-1.5/NeXT/NeXT-Video, GLM-4.1V/4.5V, Kimi-VL, MiMo, MiniCPM-o/V, Pixtral | — | Best SFT ergonomics; Megatron backend since Oct 26 2025 [VERIFIED: README]. Grounding gotcha: Qwen2.5-VL boxes must be rescaled to the resized image and special tokens handled (issue #9279) [VERIFIED: issue title] |
| VLM-R1 (OmLab) | SFT baselines | GRPO | Qwen2.5-VL, InternVL | HF generate / xllm | REC recipes and RefCOCO/+/g annotation files ready-made; SFT baselines once used mismatched pixel configs (fixed); last major update Aug 2025 → reference code, not a platform [VERIFIED: https://github.com/om-ai-lab/VLM-R1] |
| Open-R1-Multimodal, R1-V, Visual-RFT repos | — | GRPO | Qwen2-VL / 2.5-VL | HF generate | early-2025 code; superseded by EasyR1 [LIKELY] |
| Evaluation: lmms-eval | — | — | tasks `refcoco_bbox_val/test`, `refcocog_bbox_*`; model `qwen2_5_vl` etc. [VERIFIED: search] | — | Known discrepancy: Qwen2.5-VL-7B Charades-STA mIoU 29.46 in lmms-eval vs 43.6 reported (issue #857) [VERIFIED: issue title] |
| Evaluation: VLMEvalKit | — | — | 220+ LMMs, 80+ benchmarks [VERIFIED: README] | — | grounding coverage thinner than lmms-eval [LIKELY] |

Realistic throughput ([LIKELY] where derived from repo defaults, [SPECULATION] where extrapolated):
- GRPO, 7B, EasyR1, 8×H100, 8 rollouts, ≤1k-token completions, ~1M-pixel images: roughly 1–3 min per optimisation step of 128 prompts, so 500–1,000 steps in 12–36 h ≈ 100–300 GPU-hours [SPECULATION, consistent with Seg-Zero / GUI-G1 setups that train ≤1k steps on ≤8 GPUs].
- GRPO, 3B, 4×H100: about half that per step; a 500-step run ≈ 30–60 GPU-hours [SPECULATION].
- SFT/LoRA, 7B, 100k grounding samples of ~1.5k tokens (≈150M tokens) at ~2k tokens/s/GPU ≈ 20 GPU-hours; full FT ≈ 2× [SPECULATION].
- vLLM inference: Qwen3.5-4B over ScreenSpot-Pro (1,581 hi-res images) well under an hour on 1 H100 [SPECULATION].

## 3. Datasets by task (size, licence, known label problems)

### 3.1 REC / RES / detection-style
| Dataset | Size | Licence | Quality / contamination notes |
|---|---|---|---|
| RefCOCO / RefCOCO+ / RefCOCOg | ≈142k / 141k / 95k expressions on ≈20k / 20k / 26k COCO train2014 images [LIKELY] | COCO images CC-BY-4.0; annotations Apache/BSD-style [LIKELY] | Manual audit: 14% (RefCOCO), 24% (RefCOCO+), 5% (RefCOCOg) of test labels are wrong (non-unique expressions, bad boxes, misaligned targets); cleaned versions released with Ref-L4 [VERIFIED: arXiv 2406.16866, https://github.com/JierunChen/Ref-L4]. Images are COCO train2014 → inside almost every VLM pretraining/SFT mix; RefCOCO itself is in LLaVA-1.5 SFT, VILA, Qwen3-VL [VERIFIED: 2511.21631; snippet of arXiv 2411.03823]. One analysis found 48.8% of RefCOCOg val images (634/1,300) in a typical training mix, but only 0.4% verbatim expressions [LIKELY: search snippet, source paper unidentified]. RefCOCOg has UMD vs Google splits; papers mix them [LIKELY]. |
| Ref-L4 (CVPR 2025) | 45,341 annotations, 365 categories, instance scale 30–3,767 px, avg 24.2 words, 22,813 vocab; built from COCO + Objects365 [VERIFIED: 2406.16866] | as sources [LIKELY] | The best "same task, harder, cleaner" REC target; already reported by GLM-4.5V → will drift into training loops |
| gRefCOCO (GRES / GREC / GREG; CVPR 2023, IJCV 2026 GREx) | 278,232 expressions, 80,022 multi-target, 32,202 no-target, 60,287 instances, 19,994 images [VERIFIED: github.com/henghuiding/gRefCOCO] | as RefCOCO | No-target ("reject") cases are the discriminative part; metrics gIoU / cIoU / N-acc; 2026 SOTA numbers not collected (see §9) |
| ReasonSeg (LISA) | ≈1,218 image-instruction pairs (239 train / 200 val / 779 test) [LIKELY] | CC-BY-NC [LIKELY] | Tiny; val vs test confusion common; reasoning is often shallow; best trained gIoU ≈ 68 val / 67 test (Rea2Seg, AnchorSeg); RESAnything's 74.6 is a training-free pipeline on val only [VERIFIED: Verifier V-096, V-021, V-022]; Seg-Zero 57.5 zero-shot |
| HumanRef (ICCV 2025) | 6,000 test expressions in 6 subsets (attribute, position, interaction, reasoning, celebrity, rejection); avg 9.6 people / image [VERIFIED: arXiv 2503.08507] | check repo | Multi-instance + rejection; most VLMs fail because they emit one box [VERIFIED claim] |
| D3 (Described Object Detection, NeurIPS 2023) | ≈422 descriptions / ≈10.6k images [LIKELY] | unknown | absence-aware, multi-instance; small |
| RefCOCOm (MRES, CVPR 2024) | 70k part expressions, 391 part categories; MRES-32M train set (32.2M masks, 1M images) [VERIFIED: arXiv 2312.08007] | as RefCOCO | part-level; UniRES baseline |
| GroundingSuite / GSEval (ICCV 2025) | GSTrain 9.56M expressions; GSEval 3,800 images: stuff 1,000, part 500, multi-object 800, single 1,500 [VERIFIED: arXiv 2503.10596] | check repo | VLM-agent auto-annotated train set (noise unquantified); GSEval is a good multi-granularity target |
| RefBench-PRO (Dec 2025) | perception vs reasoning axes; 6 tasks incl. reject; automated generation pipeline; Ref-R1 with dynamic-IoU GRPO [VERIFIED: arXiv 2512.06276 abstract] | unknown | new enough to be uncontaminated |
| KnowDR-REC (Aug 2025) | knowledge-dependent REC [VERIFIED title: arXiv 2508.14080] | unknown | |
| Objects365 / OpenImages / LVIS / COCO | 1.7M / 9M / 118k / 118k images [LIKELY] | O365 non-commercial; OI CC-BY; COCO CC-BY-4.0 [LIKELY] | sources for synthetic REC; all inside Qwen3-VL training |

### 3.2 Grounded captioning / phrase grounding
| Dataset | Size | Licence | Notes |
|---|---|---|---|
| Flickr30k Entities | 31,783 images, 244k coreference chains, 276k boxes [LIKELY] | Flickr terms, research only [LIKELY] | standard phrase grounding; saturated (Recall@1 > 90 for detectors) [LIKELY] |
| GRIT (KOSMOS-2) | ≈20M image-text pairs with noun-phrase boxes mined from COYO/LAION [LIKELY] | web images, URL-only | noisy pseudo-labels; too large for us except sampled subsets |
| GroundingSuite GSTrain | 9.56M [VERIFIED] | check | pixel-level |
| PixMo-Cap / Molmo2-Cap | 104k videos + 431k clips dense captions (Molmo2-Cap) [VERIFIED: blog] | ODC-BY [LIKELY] | not box-grounded |
Grounded-description benchmarks with region attribution are thin; most papers build their own with GPT/Gemini judges [LIKELY].

### 3.3 Pointing / spatial referring
| Dataset | Size | Licence | Notes |
|---|---|---|---|
| PixMo-Points | ≈2.3M points on ≈428k images [LIKELY] | ODC-BY [LIKELY] | the reason Molmo points well |
| Point-Bench (PointArena) | 982 image-query pairs, 5 categories (Spatial, Affordance, Counting, Steerable, Reasoning); metric = point-in-mask [VERIFIED: github.com/pointarena/pointarena] | check | Molmo-72B ≈ Gemini-2.5-Pro at launch (0.43 pt apart, n.s.) [VERIFIED: arXiv 2505.09990]; MolmoPoint 70.7, Molmo 2 68.7, ZAYA1-VL-8B 58.0 [VERIFIED] |
| RoboPoint / Where2Place | Where2Place 100 real images (free-space referring) [VERIFIED: 2506.04308] | Apache [LIKELY] | RoboRefer-2B-RFT 71.0, Gemini-2.5-Pro 61.9 [VERIFIED] |
| RefSpatial / RefSpatial-Bench (RoboRefer) | train 20M QA, 31 relations, ≤5 reasoning steps; bench 277 (Location 100 / Placement 100 / Unseen 77), points normalized 0–1 [VERIFIED: HF dataset card, arXiv 2506.04308] | Apache-2.0 (bench) [VERIFIED] | Bench is tiny → ±5 pt CIs; Qwen3.5 already reports it |
| Molmo2-VideoPoint / MultiImagePoint / VideoTrack | 300k queries / 470k samples / 15k queries [VERIFIED: blog] | Apache + third-party caveats | video pointing + tracking training data exists now |

### 3.4 GUI grounding
| Dataset | Size | Licence | Notes |
|---|---|---|---|
| ScreenSpot-v2 | ≈1,272 instructions, mobile/desktop/web [LIKELY] | check | saturated: GUI-Cursor 93.9, InternVL3.5-241B 92.9, UI-TARS-1.5-7B 89.0 [VERIFIED snippets] |
| ScreenSpot-Pro | 1,581 instructions, 23 apps, 5 industries, 3 OS, 4K screenshots [VERIFIED: arXiv 2504.07981] | check | still discriminative (see §4) |
| OSWorld-G (NeurIPS 2025 spotlight) | 564 samples, 32 element types, incl. refusal subset [VERIFIED: arXiv 2505.13227] | check | Jedi-7B 54.1, InternVL3.5-241B 53.2, MolmoPoint-GUI-8B 70.0 |
| UGround (ICLR 2025 oral) | 10M elements, 1.3M screenshots, ≈95% web (Web-Hybrid ≈8M / 775k) [VERIFIED: github.com/OSU-NLP-Group/UGround] | check | synthetic referring expressions |
| OS-Atlas | 13M+ elements across mobile/desktop/web [VERIFIED snippet: arXiv 2410.23218] | Apache [LIKELY] | |
| Jedi | 4M examples from 4 pipelines (icons, components, doc/spreadsheet/slides) [VERIFIED] | check | raised OSWorld task success 5% → 27% for a general model |
| Aguvis | ≈110k grounding samples (mobile) per one summary [VERIFIED snippet, low confidence] | check | |
| ShowUI, GUI-Actor data | ShowUI data ≈256k? [SPECULATION] | check | |
| MolmoPoint-GUISyn | 36k screenshots, 2M points [VERIFIED] | open | |
Licence caveat: most GUI corpora are screenshots of third-party software / websites; redistribution rights are rarely explicit [LIKELY].

### 3.5 Video temporal and spatio-temporal grounding
| Dataset | Size | Licence | Notes |
|---|---|---|---|
| Charades-STA | ≈16k moment-sentence pairs on 6,672 videos [LIKELY] | Charades non-commercial [LIKELY] | legacy annotations noisy: TimeLens re-annotation flipped conclusions about closed models [VERIFIED: arXiv 2512.14698]; protocol drift up to 14 mIoU points (§5.3) |
| ActivityNet Captions | ≈20k videos, 100k sentences [LIKELY] | YouTube links (rot) | same noise issue |
| QVHighlights | ≈10k videos [LIKELY] | | TimeLens-8B R1@0.5 71.6 [VERIFIED] |
| VidSTG (from VidOR) | ≈99.9k sentences on 10k videos [LIKELY] | | STVG; new parallel-decoding results Aug 2026 |
| HC-STVG v1 / v2 | ≈5.7k / 16.5k clips, human-centric [LIKELY] | | DEViL 43.1 / 42.5 m_vIoU [LIKELY: search snippet of arXiv 2512.06673]; SpaceVLLM 39.3 (+20.2 vs Qwen2.5-VL) [VERIFIED snippet: 2503.13983] |
| TimeLens-100K | re-annotated union of legacy VTG corpora [VERIFIED] | check | best available VTG SFT set |
| MeViS / Ref-YouTube-VOS | referring video segmentation [LIKELY] | | Sa2VA family |

### 3.6 3D / embodied / spatial reasoning
| Dataset | Size | Licence | Notes |
|---|---|---|---|
| ScanRefer | 51,583 descriptions, 800 ScanNet scenes [LIKELY] | ScanNet ToS (form) | SOTA Acc@0.25 / 0.5 ≈ 62.0 / 53.6 (MCM-VG) [LIKELY: search snippet]; PSGF 58.2 / 48.4 [VERIFIED snippet] |
| Nr3D / Sr3D (ReferIt3D) | 41,503 / 83,572 [LIKELY] | ScanNet ToS | SOTA ≈ 57–58% [LIKELY snippets] |
| SpatialRGPT-Bench | ≈1.4k QA with depth [LIKELY] | | no 2026 leaderboard found (see §9) |
| MMSI-Bench (ICLR 2026) | 1,000 MCQ over 120k images, 300+ expert hours [VERIFIED: github.com/InternRobotics/MMSI-Bench] | check | best open ≈ 30%, o3 ≈ 40%, humans 97% [VERIFIED: README] |
| ERQA (Gemini Robotics) | ≈400 MCQ [LIKELY] | | Qwen3.5-27B 60.5 [VERIFIED] |
| Omni3D (ARKitScenes, Hypersim, SUN RGB-D) | | | Qwen3-VL emits 3D boxes [VERIFIED] |
| RefSpatial | see §3.3 | | |

### 3.7 Counting
| Dataset | Size | Notes |
|---|---|---|
| CountBench | 540 images, counts 2–10 [LIKELY] | saturated: Qwen3.5-27B 97.8, Qwen3.5-4B 96.3 [VERIFIED] |
| PixMo-Count / CountBenchQA | | Molmo-family strength |
| TallyQA | simple / complex splits [LIKELY] | Youtu-VL 85.1 / 74.4 [VERIFIED snippet: arXiv 2601.19798] |
| FSC-147 | 6,135 images, 147 classes, point labels [LIKELY] | class-agnostic; VLMs weak at >30 objects [LIKELY] |
| HoloCount (Jul 2026) | 3-level hierarchy [VERIFIED title: arXiv 2607.06420] | new |

### 3.8 Hallucination as grounding failure
| Benchmark | Size | Status |
|---|---|---|
| POPE | ≈9k yes/no, 3 sampling regimes [LIKELY] | easiest; ≥88 F1 for most 7B models; saturated [LIKELY] |
| AMBER | ≈15k, existence/attribute/relation, generative + discriminative [LIKELY] | "most comprehensive" per recent surveys [VERIFIED snippet] |
| HallusionBench | 346 images, 1,129 questions; qAcc / fAcc / aAcc [VERIFIED: github.com/tianyi-lab/HallusionBench] | best 0.700 (Qwen3.5-27B) on a third-party board [VERIFIED: llm-stats]; still the hardest |
| Object HalBench (CHAIR on descriptions) | ≈300 images [LIKELY] | |
| ReactBench (May 2026) | cause-driven [VERIFIED title: arXiv 2605.29579] | argues classic sets are saturated |

## 4. Benchmark status (best reported numbers, saturation, discriminativeness)

| Benchmark | Metric | Best open (size) | Best closed | Saturation verdict | Source |
|---|---|---|---|---|---|
| RefCOCO/+/g (REC) | Acc@0.5, avg over 8 splits | InternVL3.5-241B 92.4; Qwen3-VL-235B 92.1 (Thinking) / 91.9; Qwen3-VL-32B 91.9; InternVL3-78B 91.4; Qwen3.5-27B 90.9 (card); InternVL3.5-30B-A3B 90.9; Ovis2.5-9B 90.1; InternVL3.5-8B 89.7; Qwen3-VL-8B 89.1; Qwen3.5-4B 88.1 | Gemini 2.5 Pro 74.6 avg; GPT-5 (high) 66.8 avg (as measured by Qwen; output-format mismatch, not perception) | SATURATED on val/testA (top-5 spread 1–2 pts, inside 5–24% label noise); RefCOCO+ testB still 84–87; contaminated for Qwen3-VL / InternVL, undisclosed for Qwen2.5-VL / Qwen3.5. Sanity check only | [VERIFIED: Verifier A1, V-005..V-007, V-078; 2508.18265; 2511.21631; HF cards] |
| Ref-L4 | Acc@0.5 (also 0.75 / 0.9 / mAcc) | CogVLM-Grounding-17B 81.70 (paper); VLM-R1-3B 82.56, Qwen3-VL-8B 81.70, Qwen2.5-VL-7B 81.24, Ovis2.5-9B 90.29 (all from RefBench-PRO Table 3, single third-party source, protocol unstated) | not reported | NOT saturated (≈ 82 for 2024–25 models; the 90.3 outlier needs replication). Neither Qwen report publishes Ref-L4. The COCO half is image-contaminated for Qwen3-VL / InternVL; the Objects365 half is also in their mixes | [VERIFIED: Verifier V-015, V-016; 2406.16866; 2512.06276] |
| ReasonSeg | gIoU / cIoU | test (trained): AnchorSeg-13B 67.7 / 68.1; Rea2Seg-3B 66.6 / 65.5; Seg-Zero-7B 57.7 / 54.4. val: Rea2Seg 68.4 / 70.0 (trained); RESAnything 74.6 / 72.5 is a training-free pipeline (Pixtral-12B or Claude + SAM + CLIP) on **val**, not test | — | NOT saturated (trained SOTA ≈ 67–68 gIoU); val ≈ 200 and test 779 items so 1–2 pt deltas are noise; never mix training-free val numbers with trained test numbers | [VERIFIED: Verifier V-021, V-022, V-083, V-096; 2604.18562; 2606.09303; 2505.02867] |
| gRefCOCO (GRES) | gIoU / N-acc | ReLA baseline family; 2026 SOTA not collected | — | No-target subset still hard [LIKELY] | |
| HumanRef | recall/precision per subset | RexSeek | — | discriminative (multi-instance + rejection) | [VERIFIED claim] |
| GSEval | gIoU by granularity | GroundingSuite baseline | — | discriminative (stuff/part) | |
| Point-Bench | point-in-mask acc | MolmoPoint-8B 70.7; Molmo2-8B 68.7; Poivre-7B 67.5 (Qwen2.5-VL-7B + RL self-refinement); Molmo-72B 63.8; Qwen3-VL-235B 58.3; Qwen2.5-VL-7B 56.3 | Gemini-2.5-Pro 62.8 (≈ Molmo-72B at launch, n.s.) | NOT saturated (human 89.1); 982 samples → ±3 pt CIs | [VERIFIED: Verifier V-037; MolmoPoint Table 1, 2603.28069; 2509.23746; 2505.09990] |
| RefSpatial-Bench | point-in-mask acc (Loc / Place / Unseen) | RoboRefer-2B-RFT 52.0 / 54.0 / 41.6; Qwen3.5-27B 67.7 (metric aggregation unclear) | Gemini-2.5-Pro 47.0 / 24.2 / 27.1 | Discriminative, esp. Placement and Unseen; 277 samples | [VERIFIED: 2506.04308 Table 2; HF card] |
| Where2Place | acc | RoboRefer-2B-RFT 71.0 | Gemini-2.5-Pro 61.9 | 100 samples → noisy | [VERIFIED] |
| ScreenSpot-v2 | acc | MAI-UI-32B 96.5 (report); UI-Venus-72B 95.28 (community leaderboard); Holo2-30B-A3B 94.89; GUI-Cursor 93.9; InternVL3.5-241B 92.9; Jedi-7B 91.7 | — | SATURATED (> 95; remaining errors are annotation-level) | [VERIFIED: Verifier V-098; 2512.22047; 2505.13227] |
| ScreenSpot-Pro | acc | Single-pass: Qwen3-VL-235B 62.0, UI-Venus-72B 61.9, MolmoPoint-GUI-8B 61.1, Qwen3.5-4B 60.3 (card), Qwen3-VL-4B 59.5, Qwen3-VL-8B 54.6, GUI-C2-7B 50.8, SE-GUI-7B 47.2, GUI-Actor-7B 44.6. With zoom / multi-step: MAI-UI-32B 73.5 (report), UI-Venus-2-27B 74.1 (README), community leaderboard Indeed-UI-32B-zoomin 82.7, Holo2-235B 81.5, KV-Ground + Qwen3.5-27B router 80.9. Qwen3.5-27B 70.3 (card; single-pass or not unstated) | Gemini-3-Pro 72.7, Seed1.8 73.1 (MAI-UI report). The 84–88 numbers for Claude Opus 4.8 / GPT-5.2 / Qwen3.8-Max on llm-stats are self-reported with 0 verified | NOT saturated but climbing fast; single-pass and agentic numbers are no longer comparable; icons lag text | [VERIFIED: Verifier V-026, V-027, V-079, V-085, V-099; 2504.07981; 2511.21631; 2512.22047; MolmoPoint blog] |
| OSWorld-G | acc | MAI-UI-32B + zoom 70.9; UI-Venus-72B 70.4; MolmoPoint-GUI-8B 70.0 (blog); Qwen3-VL-235B 68.3 / 66.7; MAI-UI-32B 67.6; Qwen3-VL-32B 65.1; GTA1-32B 65.2; Qwen3-VL-8B 58.2; Jedi-7B 54.1; InternVL3.5-241B 53.2; InternVL3.5-8B 36.4 | Seed1.5-VL 62.9; Gemini-2.5-Pro 45.2 (Jedi Table 5) | NOT saturated (≈ 71); 564 items incl. 54 refusal cases; UI-Venus-2-27B's 79.1 is on the different OSWorld-G-R split. Agent-level link verified: with a GPT-4o planner OSWorld success 5.0 → 24.0 (Jedi-3B) → 27.0 (Jedi-7B); 51.0 with an o3 planner | [VERIFIED: Verifier A1, V-084; 2505.13227; 2511.21631; 2512.22047] |
| Charades-STA | R@0.5 / R@0.7 / mIoU | VITAL-7B 72.0 / 46.7 / 59.9 (Qwen2.5-VL-7B, RL, trained on Charades-STA train); SG-DETR specialist 71.1 / 52.8 / 60.7; AVI training-free agent 69.0 / 37.6 / 60.0; Qwen3-VL-235B mIoU 64.8; TimeLens-8B 63.0 / 35.2 / 55.2 (SOTA only within its own comparison set); TAR-7B mIoU 61.1 [LIKELY snippet] | Gemini-2.5-Pro 61.1 / 34.0 / 52.8; GPT-5 42.0 / 22.0 / 40.5 (TimeLens Table 1) | NOT saturated at R@0.7 (best ≈ 53, specialist); MLLMs match specialists at R@0.5 but lag on boundary precision; protocol-fragile (same model 43.6 vs 39.3 vs 29.5 mIoU across harnesses); legacy labels noisy | [VERIFIED: Verifier V-097, A1; 2508.04416; 2410.01615; 2511.14446; 2512.14698] |
| ActivityNet Captions | R1@0.5 | TimeLens-8B 58.4 | — | same caveats; video rot | [VERIFIED] |
| HC-STVG v1 / v2 | m_vIoU | DEViL 43.1 / 42.5 [LIKELY]; SpaceVLLM 39.3 | — | Far from ceiling; MLLM-native STVG is young | [snippets] |
| VidSTG | m_vIoU | PTD (4B) claims SOTA, numbers not extracted | — | open | [2608.28192] |
| ScanRefer / Nr3D | Acc@0.25/0.5 ; acc | ≈62 / 54 ; ≈58 | — | Far from ceiling; requires ScanNet + 3D tooling | [LIKELY snippets] |
| MMSI-Bench | acc | ≈30% open | o3 ≈ 40% | Wide open (humans 97%) | [VERIFIED README] |
| CountBench | acc | Qwen3.5-27B 97.8 | — | SATURATED | [VERIFIED] |
| POPE | F1 | ≥88–90 broadly | — | SATURATED | [LIKELY] |
| HallusionBench | aggregate | Qwen3.5-27B 0.700 | — | Hard but leaderboard mixes protocols | [VERIFIED: llm-stats] |
| AMBER | CHAIR / F1 | — | — | moderately saturated per surveys | [LIKELY] |

Practical rule: for an academic paper in 2026 the credible primary targets are ScreenSpot-Pro / OSWorld-G (GUI), RefSpatial-Bench + Point-Bench + Where2Place (pointing/spatial), Ref-L4 + HumanRef + RefBench-PRO + GSEval (REC/RES beyond RefCOCO), Charades-STA under a fixed protocol + HC-STVG (video), MMSI-Bench (spatial reasoning). RefCOCO, ScreenSpot-v2, CountBench, POPE are for tables in the appendix.

## 5. Pitfalls

### 5.1 Coordinate conventions (each model family differs)
| Family | Convention | Source |
|---|---|---|
| Qwen2-VL | 0–1000 normalized `<box>` | [LIKELY] |
| Qwen2.5-VL | absolute pixels of the smart_resize'd image (multiples of 28); JSON bbox_2d / point_2d; evaluation must rescale to original size; results depend on min_pixels / max_pixels | [VERIFIED: Qwen2.5-VL issue #866; HF discussion #13] |
| Qwen3-VL | back to [0,1000] relative | [VERIFIED: 2511.21631] |
| Qwen3.5 / 3.6 / 3.8 | undocumented on cards; must inspect | open |
| InternVL 2.5 / 3 / 3.5 | 0–1000 `<box>[[x1,y1,x2,y2]]</box>`, `<ref>` tags | [LIKELY] |
| PaliGemma 1/2 | 1024 `<locXXXX>` tokens, y-first order [LIKELY] | |
| Florence-2 | 1000-bin location tokens | [LIKELY] |
| Molmo 1/2 | `<point x="12.3" y="45.6">` percent floats; `<points …>` for multiple | [VERIFIED: Molmo blog / paper] |
| MolmoPoint | patch / subpatch / location special tokens (no numbers) | [VERIFIED] |
| Gemini 2.x / 3 | `[ymin, xmin, ymax, xmax]` integers 0–1000; points `[y, x]` | [VERIFIED: https://ai.google.dev/gemini-api/docs/image-understanding] |
| Gemma 4 | JSON, 1000×1000 grid | [VERIFIED: HF blog] |
| GLM-4.1V / 4.5V / 4.6V | normalized xyxy inside `<|begin_of_box|>` … `<|end_of_box|>` | [VERIFIED snippet] |
| Rex-Omni | 0–999 special tokens | [VERIFIED] |
| Claude (computer use) | pixel click coordinates on the provided screenshot | [LIKELY] |
Consequences: (a) SFT data must be converted per model; a wrong convention silently costs 10–40 points; (b) RL reward code must parse the model's native format and rescale before IoU; (c) cross-model tables compiled from papers mix conventions and resolutions.

### 5.2 Resolution handling
- Qwen-family dynamic resolution: min_pixels / max_pixels change both the token count and (for 2.5-VL) the coordinate frame; report both. Community reports of "Qwen2.5-VL grounding is worse than Qwen2-VL" often trace to this [VERIFIED: issue #866 title; attribution LIKELY].
- ScreenSpot-Pro is 4K; a model with a 1–2M-pixel budget sees tiny icons at ≤10 px. Zoom-in / crop-then-ground (ScreenSeekeR raised OS-Atlas-7B from 18.9 to 48.1; GUI-Eyes, RegionFocus) is a resolution fix, not a reasoning fix [VERIFIED: 2504.07981; LIKELY for others].
- Gemma 4 lets you pick 70–1120 image tokens; PaliGemma 2 ships 224/448/896 checkpoints; use them for controlled resolution studies.

### 5.3 Evaluation-protocol inconsistencies
- REC Acc@0.5 on boxes vs RES cIoU / gIoU on masks vs "point-in-mask" for pointing; papers report "RefCOCO" for all three.
- "RefCOCO avg" is sometimes over 8 splits, sometimes 3 vals; RefCOCOg UMD vs Google splits.
- ReasonSeg val (200) vs test (779); many papers report val only.
- Charades-STA for Qwen2.5-VL-7B: 43.6 mIoU (vendor) vs 39.3 (TimeLens) vs 29.46 (lmms-eval #857) — frame sampling, fps, prompt wording and timestamp parsing each move the number [VERIFIED].
- GUI: ScreenSpot-Pro accuracy = predicted point inside GT box; some methods output boxes and take the centre; a few report IoU-based success [LIKELY].
- Multi-instance (HumanRef, gRefCOCO, D3): precision/recall/F1 with rejection; single-box models get scored generously if only "first box" is used [LIKELY].
- Third-party leaderboards (llm-stats, benchlm) mix vendor-reported and self-run numbers and mislabel open weights (Qwen3.5-27B shown as closed) [VERIFIED: llm-stats page]. Cite primary sources.

### 5.4 Reward hacking and format rewards in grounding RL
- GUI-G1 (NeurIPS 2025): hit-reward → smaller boxes (accuracy up, IoU down); IoU-reward → larger boxes; fix with a box-size constraint, a fast-thinking template (no long CoT), difficulty weighting and removing length normalisation [VERIFIED: arXiv 2505.15810].
- GUI-G2: Gaussian point + coverage rewards give dense, scale-aware signal; error shrinks monotonically (290 px → 150 px) vs binary rewards [VERIFIED: arXiv 2507.15846].
- Rex-Omni adds de-duplication rewards because multi-instance policies learn to spam boxes [VERIFIED abstract].
- Format rewards saturate within ~50 steps and then are pure noise in the advantage; unparseable outputs must get a defined reward (usually 0), otherwise the policy learns to break the parser [LIKELY: common practice].
- With KL-free GRPO variants (DAPO / GSPO) the policy can drift to degenerate short outputs; monitor mean IoU and box-size distribution, not only accuracy [SPECULATION].
- RL on RefCOCO train with a contaminated base mostly re-elicits memorised boxes; out-of-domain evals (LISA-Grounding, Ref-L4, HumanRef) are the only honest signal [LIKELY: VLM-R1 finding].

### 5.5 Data leakage between benchmarks and base-model training
- Every COCO-derived benchmark (RefCOCO family, gRefCOCO, RefCOCOm, POPE, AMBER, Object HalBench, Flickr30k-adjacent) shares images with COCO captions / VQAv2 / GQA / LLaVA mixes; Qwen3-VL lists RefCOCO/+/g, COCO, Objects365, OpenImages directly [VERIFIED: 2511.21631]; LLaVA-1.5 and VILA include RefCOCO in SFT [VERIFIED snippet: arXiv 2411.03823]. "Both Text and Images Leaked!" (arXiv 2411.03823) is the reference for systematic multimodal contamination analysis.
- Vendor-reported grounding numbers (Qwen, InternVL, GLM) are on benchmarks whose train splits are in their mixes; treat as in-distribution.
- Newer benchmarks that are probably clean today: RefBench-PRO (Dec 2025), GSEval (Mar 2025), HumanRef (Mar 2025), RefSpatial-Bench (Jun 2025; but Qwen3.5 already reports it, so its train set is likely in the mix), MMSI-Bench, HoloCount, ReactBench.
- Candidate "clean" base models: Molmo 2 / MolmoPoint (PixMo + Molmo2 data; no RefCOCO mention) [LIKELY], Gemma 4 (unknown), SmolVLM2 (unknown), Qwen3.5 (unknown, presumably contaminated). Nobody publishes decontamination reports for grounding; an n-gram/image-hash overlap audit is itself a cheap contribution.

### 5.6 Evaluating closed models
- Gemini 3 / 3.1 Pro return boxes and points via prompt (0–1000); Claude computer-use returns click coordinates; GPT-5.x box output is undocumented and inconsistent [LIKELY]. Vendors control resolution and preprocessing; versions change silently; refusals on people-centric data (HumanRef celebrity subset) bias results.
- Cost is manageable: 1.5k–5k images per benchmark ≈ $10–100 per model per run at 2026 prices [SPECULATION].
- Third-party numbers for closed models on GUI benchmarks come partly from vendor blogs; ScreenSpot-Pro 87.9 for Claude Opus 4.8 is vendor-adjacent [LIKELY].

### 5.7 Statistical power
95% CI half-width for accuracy near 60% is ≈ ±4.3 pts at n=500, ±3.0 at n=1,000, ±9.6 at n=100. ReasonSeg test (779), RefSpatial-Bench (277), Where2Place (100), OSWorld-G (564), Point-Bench (982) and ScreenSpot-Pro (1,581) therefore cannot resolve 1–3 point claims without multiple seeds and paired tests.

## 6. Menu of cheap but convincing experiments (each ≤ 200 GPU-hours)

E1. Contamination-controlled REC fine-tuning (≈ 60 GPU-h). Bases: Molmo2-4B (assumed clean), Gemma4-E4B (unknown), Qwen3.5-4B (assumed contaminated). Identical LoRA SFT on 20k RefCOCO-train expressions, identical convention conversion. Evaluate RefCOCO (cleaned Ref-L4 version), Ref-L4, RefBench-PRO, HumanRef. Signal: the RefCOCO-vs-Ref-L4 gap per base isolates memorisation from skill; also gives the first Gemma 4 grounding numbers. 1×H100 per base, ~10–20 h each.

E2. Coordinate-representation ablation (≈ 80 GPU-h). One base (Gemma4-E4B or Qwen3.5-4B), five output formats: absolute pixels, 0–1000 ints, 0–1 floats, 1000 quantised special tokens (Rex-Omni style), MolmoPoint-style patch tokens (requires a small head). Same 50k-sample mix (REC + pointing + GUI). Evaluate REC, Point-Bench, ScreenSpot-Pro, plus small-object and resolution-shift splits. Prior work compares across models; this holds everything else fixed.

E3. GRPO reward-design ablation for REC at 3–4B (≈ 160 GPU-h). EasyR1, Qwen3-VL-4B, 5k RefCOCOg + 2k HumanRef prompts, rewards ∈ {IoU, hit, Gaussian (GUI-G2), IoU + box-size (GUI-G1), dynamic-IoU (Ref-R1)}, 500 steps × 8 rollouts; 4×H100 ≈ 8 h per config. Track box-size drift, duplicate rate, OOD (Ref-L4, LISA-Grounding). Clean, publishable negative results likely.

E4. Resolution / zoom probing with no training (≈ 10 GPU-h). Qwen3.5-4B / 27B and Gemma4-E4B on ScreenSpot-Pro and Ref-L4-small-objects with a max_pixels sweep and a two-stage crop-then-ground. Quantifies how much of "reasoning" gains in the literature are resolution.

E5. Synthetic-data transfer study (≈ 40 GPU-h). Render HTML/GUI (Jedi-style) or composited object scenes with exact boxes/points; SFT a 2–4B model on 0 / 10k / 100k synthetic + fixed 5k real; measure real-benchmark slope. Tells the team whether a data-synthesis idea is worth building.

E6. Linear probes for location knowledge (≈ 5 GPU-h). Freeze Qwen3-VL-8B / Molmo2-8B; train linear probes from hidden states at the referent's last token to the box centre / mask. If probes beat the model's own emitted coordinates, the bottleneck is decoding, not perception → motivates non-autoregressive heads (MolmoPoint, GUI-Actor, PTD).

E7. Protocol-variance audit for video grounding (≈ 20 GPU-h). Five open models × Charades-STA under three published protocols (vendor, lmms-eval, TimeLens). Report the variance table; propose a fixed protocol. Cheap, citable, and needed.

E8. Spatial-referring RFT at 2–4B (≈ 100 GPU-h). Reproduce the RoboRefer RFT stage on a 100k RefSpatial subset with Qwen3.5-4B or Molmo2-4B; evaluate RefSpatial-Bench + Where2Place + Point-Bench-Spatial. Checks whether the 2B-RFT > 8B-SFT result transfers to a general base.

E9. Label-noise sensitivity (0 GPU-h, ≈ $300 annotation). Re-annotate 500 RefCOCO+ test items; recompute the ranking of 6 open models on original vs cleaned labels; report rank flips.

E10. Test-time scaling for grounding (≈ 10 GPU-h). Sample N = 1…16 boxes per query; aggregate by NMS-vote / mean / self-verify; measure gains on Ref-L4, ScreenSpot-Pro, RefSpatial-Bench. Establishes whether inference-time compute is a free baseline every RL paper should include.

E11. Multi-instance and rejection behaviour (≈ 10 GPU-h). Evaluate 8 open models on HumanRef, gRefCOCO no-target, OSWorld-G refusal, RefBench-PRO reject; quantify hallucinated boxes on absent targets. Grounding-hallucination link, cheap.

E12. Video pointing / tracking with Molmo 2 data (≈ 150 GPU-h). LoRA-tune Qwen3.5-4B on 50k Molmo2-VideoPoint samples; evaluate against Molmo2-Vid; tests whether Ai2's data alone transfers the capability.

## 7. What is out of reach at 1–8 GPUs for weeks

- Pretraining or continued pretraining of any VLM; re-creating Qwen3-VL-style grounding mixes (COCO + O365 + OpenImages + synthetic) or Rex-Omni's 22M-sample SFT, UGround 10M, OS-Atlas 13M, RefSpatial 20M, GroundingSuite 9.56M in full. Subsample to ≤500k.
- Full fine-tuning of ≥27B models (Qwen3.5-27B, Qwen3.6-27B, Muse Glimmer-30B, Gemma4-31B): weights + optimizer alone exceed 8×80 GB without ZeRO-3 + offload; LoRA only. RL on ≥32B: no.
- Proprietary training data and models: Gemini / GPT / Claude grounding data; Muse Spark; Qwen3.8-Max vision weights (text-only release); anything requiring >100k closed-model API calls for distillation (cost ≈ $1–5k, borderline but licence-restricted for some vendors).
- Human annotation beyond ≈10k boxes/points (≈ $0.5–2 each) without a grant; expert 3D or medical annotation at any scale.
- Real-robot or real-desktop closed-loop evaluation (Point-Act, OSWorld task success) unless the lab already runs that infrastructure; use OSWorld-G / ScreenSpot-Pro as proxies.
- Long-video STVG at high fps on 7B+ models (memory), and 3D grounding pipelines that need ScanNet reconstruction + 3D detectors (weeks of engineering, not GPU).
- New large benchmarks with human labels (>2k items) unless auto-generated + spot-checked.

## 8. Practical notes on candidate directions (after reading r1-frontier.md and r3-history.md)

Overlaps to resolve first (recommend `main` merges into IDEA-9x lines): D-R1b ≈ NF-1 (edit-to-verify / interventional verification), D-R1d ≈ NF-2 (attributed description with necessity + sufficiency), D-R1e ≈ NF-4 (selective / calibrated grounding). Notes below are per merged line.

Cost tiers: A = kill experiment ≤ 50 GPU-h and full study ≤ 100; B = full study 100–300 GPU-h; C = > 300 GPU-h, or needs data that does not exist.

### 8.1 Latent grounding gap (D-R1a) — tier A audit, tier B training
- The audit is inference-only. R1's kill experiment (IoU of the read-out vs GT on Qwen3-VL-8B failures over GroundingME + OpenRef) is ≈ 5 GPU-h; GroundingME (1,005 items, four dimensions) is on Hugging Face with vLLM-based evaluation code that already supports Qwen3-VL-8B-Thinking [VERIFIED: https://github.com/lirang04/GroundingME].
- Blocker 1 (architecture): Qwen3.5 / 3.6 / 3.8 use Gated-DeltaNet hybrid layers [VERIFIED: HF Qwen3.5-4B card]; linear-attention layers have no attention matrix to read, so attention read-outs work only on the minority softmax layers [LIKELY]. Centre the audit on all-softmax models: Qwen3-VL-4B/8B, Molmo2-8B (Qwen3-8B LLM), InternVL3.5-8B, Gemma4-E4B, with MolmoPoint as the non-text-decoder contrast. Gradient-based read-outs (entropy-gradient) still work on hybrids.
- Blocker 2 (tooling): attention weights need eager attention (flash-attn / SDPA return none) → HF forward, not vLLM; with ~4k visual tokens a 32-head layer's attention is ≈ 1 GB, so aggregate with hooks per layer instead of storing. Fine on one H100 for ≤ 8B [LIKELY].
- Blocker 3 (which tokens): with 0–1000 coordinate strings the referent's attention is spread over digit tokens; use a yes/no probe ("Is there a {expr}?") or the first coordinate token and fix the choice in the audit.
- Blocker 4 (RL stage): online frozen-teacher read-outs would need an HF forward per rollout (0.3–1 s each). Since the teacher is frozen, precompute its read-out mask per prompt once and store it; the reward becomes a lookup + IoU, the RL job needs no extra model copy, and EasyR1 7B GRPO fits 4×H100. The causal term (mask region → answer changes) doubles forward passes; batch them.
- Use Ref-L4's Acc@0.5 / 0.75 / 0.9 and mAcc for the high-IoU analysis [VERIFIED: 2406.16866 via R3].
- Budget: audit 20 GPU-h; test-time exploitation 10; RL 150 (4B) / 300 (8B).

### 8.2 Edit-to-verify / interventional verification (D-R1b + NF-1) — tier B with a clear engineering path
- Editing model (R3's question): two tiers. For removal at RL scale, mask-conditioned inpainting (LaMa-class, ≈ 50 ms per 1-Mpx image [LIKELY]) or Qwen-Image-Lightning (distilled, 4–8 steps, 12–25× faster than base, Apache-2.0) [VERIFIED: search]. For attribute swap and distractor insertion, Qwen-Image-Edit (monthly releases 2509, 2511, Qwen-Image-2.0 Feb 2026; Apache-2.0), FLUX.2 klein 4B (Apache-2.0; 9B is non-commercial), or Microsoft's 4B editor with a 0.59 s/image turbo mode on A100 (Jul 2026; name unverified) [VERIFIED: search snippets]. Budget 0.5–2 s per edit → 100k edits ≈ 30–60 GPU-h.
- Masks and instances: SAM 3 (Nov 19 2025; concept prompts return every instance of a noun phrase; SAM 3.1 Mar 2026) under a custom SAM licence that allows research and commercial use with restrictions, not Apache/MIT [VERIFIED: https://github.com/facebookresearch/sam3 ; search]. Check redistribution terms before releasing derived datasets.
- Blocker 1 (cost): editing per rollout for the policy's own proposals (8 rollouts × 3 edits × ~1 s per prompt per step) dominates. Fix: precompute an edit bank per image over SAM 3 instances (remove / swap each of 5–15 instances) and at reward time select the edit whose mask best matches the predicted box; fall back to the fast inpainter only for off-bank boxes. Rewards become lookups.
- Blocker 2 (artifact leakage): include control edits of matched size elsewhere and train a small edited-vs-control classifier as a gate; if a ResNet-18 exceeds ≈ 0.6 AUROC the reward is exploitable. This ≤ 2 GPU-h check should be the first kill experiment.
- Blocker 3 (judge): the crop judge needs a strong open VLM served by vLLM on its own GPU (Qwen3.5-9B or Gemma4-12B; ≈ 0.1–0.3 s per call).
- Blocker 4 (contamination): keep COCO out of the image pool; use SA-1B / OpenImages / Objects365.
- Evaluation targets exist: GroundingME, OpenRef, gRefCOCO no-target, Ref-Adv, HalluSegBench.
- Budget: kill experiment (2k edit pairs + artifact test + hallucinated-box rate on removals) 10 GPU-h; full study: edit bank 50 + GRPO 4B 100 + evaluation 20 ≈ 170 GPU-h.

### 8.3 Referring games / self-play (D-R1c) — tier B
- Run as alternating phases, not simultaneous self-play: speaker GRPO with a frozen listener served in vLLM inside the reward function (EasyR1 reward functions are plain Python and can call a server [LIKELY]); then listener SFT/RL on the produced expressions.
- Data: SA-1B has no labels; SAM 3 concept prompts supply same-category instance sets; pre-filter to images with ≥ 3 same-category instances.
- Blockers: language drift (KL to base + naturalness judge), listener overfitting to speaker idiosyncrasies (evaluate the listener on human benchmarks every round: GroundingME-Discriminative, Ref-Adv), and speaker quality needs a small human study (≈ $200 for 500 items).
- Budget: speaker 4B, 500 steps ≈ 50 GPU-h per round; two rounds + listener training ≈ 150 GPU-h. R1's kill comparison (listener trained on game expressions vs on equal-size Grounding-DINO pseudo-labels) is the right one.

### 8.4 Attributed description (D-R1d + NF-2) — tier B at 4B, tier C at 8B
- Long outputs (200+ words plus tags) make rollouts 600–1,000 tokens: 3–4× the REC cost. The verifier ensemble adds 10–30 judge calls per rollout → 3–10 s reward latency unless batched on a dedicated judge GPU. Estimate: 4B policy, 4 rollouts ≈ 250 GPU-h; 8B ≈ 500 GPU-h. Recommend 4B, precomputed sufficiency (crop judge) and necessity computed on a subsample of claims.
- Must run first: judge AUROC against DetailVerifyBench human span labels (R1's kill test) ≈ 5 GPU-h.
- Data: DenseWorld-1M for seed SFT [LIKELY], SAM 3 for masks, claim decomposition with a text LLM (Qwen3.5-9B).
- The P12 experiment (does better localization causally reduce hallucination?) is cheap and clean: same base, same data budget, attribution reward vs RLHF-V-style text correction; measure CHAIR / AMBER / Object HalBench ≈ 60 GPU-h.

### 8.5 Selective / calibrated grounding (D-R1e + NF-4) — tier A
- Cheapest on the list. Risk-coverage protocol across 8–10 models on REC / GUI / VTG is inference-only ≈ 20 GPU-h. Token-likelihood confidence is free (vLLM logprobs) and must be the first baseline (R3's point).
- NF-4 training: standard GRPO with IoU + Brier term at 4B ≈ 60 GPU-h. Calibration splits from GroundingME / ScreenSpot-Pro are hundreds of items → loose guarantees; use Ref-L4 (45k) as the calibration set.
- Blocker: novelty (SafeGround, RLCR, HKVLM). Strongest as the evaluation / deployment half of 8.1 or 8.2, as both authors already suggest.

### 8.6 Precision bounds factorial (D-R1f, handed to me) — tier B; I will draft it as IDEA-21 if `main` agrees
- Design: one 2B backbone (Qwen3-VL-2B or Gemma4-E2B), fixed 200k-sample grounding mix (Objects365 + GRIT subset, no COCO). Factors: coordinate format (absolute px / 0–1000 / 0–1 float / Hi-Token axis tokens / Rex-Omni quantised tokens) × decoder (autoregressive vs a small parallel box head as in LocateAnything / GUI-Actor) × patch merge (2×2 vs none) × position encoding (M-RoPE vs interleaved I-MRoPE). Full factorial is 40 runs; a fractional design of ≈ 12 runs at ≈ 15 GPU-h each ≈ 180 GPU-h. Evaluate on Ref-L4 Acc@0.5 / 0.75 / 0.9 stratified by object size, plus ScreenSpot-Pro.
- Blocker: changing patch merge or RoPE changes the token pitch the LLM sees → needs a short connector re-alignment (≈ 10 GPU-h at 2B on ~500k caption pairs) per such arm, otherwise results are confounded. Budget it explicitly. This is E2 from §6 extended.

### 8.7 Gaze as process supervision (NF-3) — tier C: the data does not exist at the needed scale
- Existing gaze-VQA sets: AiR (1,422 GQA questions, 20 participants), VQA-MHUG (3,990 VQAv2 stimuli, 49 participants, 2021), VQA-HAT (crowd blur-based, not eye-tracking), DriVQA (driving, 2025), Gaze-VLM (Oct 2025, interactive protocol, size unverified) [VERIFIED: search]. All small, and GQA / VQAv2 images are COCO / Visual Genome → contaminated.
- A new collection needs IRB, participants and weeks; webcam gaze error (~100 px) is coarser than the crops being supervised [LIKELY]. The diagnostic (human-fixation vs random-region process reward on ~1.4k items) is ≈ 30 GPU-h but underpowered. Deprioritise unless a partner lab has an eye tracker and a subject pool.

### 8.8 Prior-robust grounding under outcome rewards (NF-5) — tier A diagnostic, tier B training
- Diagnostic is cheap: Charades-CD / ActivityNet-CD splits exist (https://github.com/yytzsy/grounding_changing_distribution; ships I3D / C3D features only, MLLMs need raw video; Charades videos are downloadable, ActivityNet suffers link rot) [VERIFIED: search]; Ref-Adv; null-expression inputs. ≈ 10 GPU-h for five models.
- Training: the blind-policy baseline in the advantage requires null-expression rollouts → 2× rollout cost; ≈ 80–120 GPU-h at 4B.
- Conflict to flag: Charades-CD inherits Charades-STA labels that TimeLens found noisy enough to re-rank models [VERIFIED: 2512.14698]; part of the "moment prior" may be annotation artefact. Use TimeLens re-annotations where they overlap and report both.

### 8.9 Cross-cutting engineering notes
- Default bases for Phase 3: Qwen3-VL-4B/8B (Apache-2.0, all-softmax attention, EasyR1-native, 0–1000 coordinates); Gemma4-E4B as a second lineage (TRL-native); Molmo2-4B/8B for contamination control. Qwen3.5+ for inference baselines only until RL-framework support for the hybrid architecture is verified.
- RLVR forgetting (R3's open question): 2026 work reports RL forgets less than SFT (S-GRPO arXiv 2604.16557; arXiv 2605.09640), with the mechanism disputed (KL penalty vs on-policy data) and one counter-result that continual RL still forgets (arXiv 2607.04364) [VERIFIED: search snippets]. No grounding-specific measurement found → a 2 GPU-h add-on to any RL run: MMMU / MMBench / POPE before and after.
- Every RL plan should carry the free baselines: test-time sampling + NMS voting (E10), a max_pixels sweep (E4), and token-likelihood confidence.
- Use open judges (Qwen3.5-9B / Gemma4-12B via vLLM) so rewards are reproducible; closed-model judges break reproducibility and add cost.

### 8.10 Image editing inside an RL loop (lead question 1; serves IDEA-91 = D-R1b + NF-1)

Cost per edit on one H100 at ~1024² output, batch 1 (latencies are the best figures I could source; treat as ±2×):

| Editor | Size / licence | Steps | Latency per edit | Removal | Attribute swap | Distractor insertion | Notes |
|---|---|---|---|---|---|---|---|
| LaMa / big-LaMa (mask inpainting) | ~50M, Apache-2.0 [LIKELY] | 1 forward | ≈ 0.03–0.1 s [LIKELY] | good for objects ≤ ~15% of the image on textured backgrounds; smears on large objects | no | no | needs a mask (SAM 3); cheapest by far |
| SD-1.5 / SDXL inpainting | 0.9B / 2.6B, OpenRAIL-M [LIKELY] | 20–30 | ≈ 1–3 s [LIKELY] | ok | via prompt, unreliable | weak | classic failure: "remove the cat" inpaints another cat |
| FLUX.2 [klein] 4B | 4B, Apache-2.0 (9B is non-commercial) | 4 (distilled) | ≈ 0.57 s for 1024² generation; editing similar order [VERIFIED: search — inferencebench.io, comfy.org] | good | good | fair | instruction editing, no mask needed; multi-reference input |
| Mage-Flow-Edit-Turbo (Microsoft, Jul 22 2026) | 4B, MIT | 4 | ≈ 1.02 s per edit on A100 at 18–20 GB peak → ≈ 0.6–0.8 s on H100 [VERIFIED: search; H100 scaling SPECULATION] | good ("sufficient for simple removal") | fair | fair; weaker on compound instructions [VERIFIED snippet] | native-resolution; RL-aligned variant exists |
| Qwen-Image-Edit-2511 (+ Lightning LoRA) | 20B MMDiT, Apache-2.0 | 40 (base) / 4 (Lightning) | ≈ 15 s base, ≈ 60 s at 100 steps; Lightning ≈ 10× faster → ≈ 1.5–2 s [VERIFIED: search — oxen.ai, lightx2v HF; Lightning latency LIKELY] | best | best | best of the open set [LIKELY] | 40+ GB VRAM; use for the offline bank, not per rollout |
| FLUX.1 Kontext dev (12B) | non-commercial | 28 | ≈ 5–8 s [LIKELY] | good | good | fair | licence rules it out for released data |
| Copy-paste insertion (SAM 3 instance from another image + LaMa/Poisson blending, POBF-style) | free | — | ≈ 0.1 s [LIKELY] | — | — | controllable look-alike distractors | artifacts are obvious but *matched* controls are easy |

Recommended architecture: an offline **edit bank**, not per-rollout editing. For a 20k-image pool: SAM 3 instances (≈ 8 per image) → one LaMa removal each (160k × 0.05 s ≈ 2.5 GPU-h); two attribute swaps per image with Mage-Flow-Edit-Turbo or FLUX.2 klein (40k × 0.8 s ≈ 9 GPU-h); one distractor insertion per image (20k, copy-paste or 4B editor, ≈ 5 GPU-h); judge verification of every edit with Qwen3.5-9B on vLLM (≈ 0.15 s each, ≈ 10 GPU-h). Bank total ≈ 25–30 GPU-h, then rewards are lookups keyed by the SAM 3 instance whose mask best overlaps the predicted box. Per-rollout online editing would cost 128 prompts × 8 rollouts × 0.8 s ≈ 14 min per step on one GPU (≈ 2 min with two dedicated async edit GPUs), which is comparable to or larger than the rollout time itself; only worth it for the small fraction of predicted boxes that match no bank instance, and those can use LaMa.

Expected edit quality (to be measured in the pilot; all [SPECULATION]): judge-verified removal success ≈ 80–90% for objects under ~15% image area, falling steeply above ~30%; attribute swap success ≈ 60–80% with the 4B editors; look-alike distractor insertion ≈ 50%. Small referents (< 32 px) are the weak spot for every editor and are exactly the GroundingME "Limited" cases.

Artifact-leakage detection protocol (≤ 2 GPU-h):
1. **Always-edited design.** Every image in the pool, including positives where the referent stays, receives a control edit of matched mask size on a non-referent region with the same editor. "Was this image edited?" is then uninformative by construction; the only exploitable cue would be "was the edit *at the referent*".
2. **Classifier probe.** 5k referent-removed vs 5k control-edited images; ResNet-18 and ViT-S/16 (ImageNet init), 5 epochs, plus an off-the-shelf forgery localizer (TruFor / CAT-Net class [LIKELY names]) as a stronger adversary. Report AUROC with the referent location hidden (random crops) and shown. AUROC ≈ 0.5 hidden / higher shown is acceptable; AUROC > 0.6 hidden means the editor leaves a global signature and the editor or the control design must change.
3. **Behavioural probe on the policy.** Rejection rate on control-edited positives must match clean positives within the CI; a rise means the policy reads artifacts. Run before and after RL.
4. **Invariance operator** (NF-1's third relation): an edit elsewhere must not move the box; log the distribution of box shifts under control edits as a training-time monitor.
5. Frequency-domain sanity check: high-pass residual energy inside vs outside edit masks, once per editor.

### 8.11 Internal read-outs at scale (lead question 2; serves IDEA-11 = D-R1a)

Token counts that set the cost: Qwen3-VL uses 16-px patches with 2×2 merge → ~32 px per visual token [LIKELY], so 1024² ≈ 1k tokens and a 2.6-Mpx GUI screenshot ≈ 2.5k; InternVL3.5 uses 448² tiles at 256 tokens each, up to 12 tiles + thumbnail ≈ 3.3k [LIKELY]; Molmo 2 uses SigLIP2-so400m/14 at 384 with overlapping multi-crop ≈ 1–2k [LIKELY].

Memory: attention weights per layer = heads × seq² × 2 bytes. For a 32-head 8B model [LIKELY] at seq 1.5k that is ≈ 144 MB per layer, at 4k ≈ 1 GB per layer; never materialise all 36 layers (36 GB at 4k). Register per-layer hooks that keep only the rows of the probe / answer tokens (heads × answer_tokens × seq, i.e., kilobytes) and discard the rest. Eager attention (required; flash-attn and SDPA do not return weights) makes prefill ≈ 2–3× slower: an 8B prefill of 1.5k tokens ≈ 40–60 ms with flash vs ≈ 120–200 ms eager on H100 [SPECULATION]. Gradient read-outs (entropy-gradient to the visual embeddings) need one backward with respect to inputs only: ≈ 2× a forward and ≈ 10–15 GB of activations at batch 1 for 8B at 2k tokens [SPECULATION]. Intermediate-layer probes (InnerZoom-style) are free once the forward runs.

Three ways to use read-outs in GRPO on 8 GPUs (EasyR1, 8B policy, FSDP + vLLM colocate, 128 prompts × 8 rollouts per step, rollout time ≈ 60–120 s):
- **Option A, precomputed frozen-teacher read-outs (recommended).** The teacher never changes, so compute its read-out mask once per prompt for the whole pool (20k prompts × ≈ 0.3 s ≈ 1.7 GPU-h), store as 64×64 masks (8 KB each), and make the reward IoU(box, mask), a CPU lookup. Runtime and memory cost inside the RL job: zero.
- **Option B, online read-outs from the current policy** (needed only if the reward is agreement between the policy's *own* attention and its emitted box). Requires an HF eager copy of the policy synced after each step (16 GB at 8B): 1,024 forward passes × ≈ 0.2 s ≈ 200 s per step on one GPU, ≈ 25–30 s when sharded over 8 → +30–50% step time. Memory: 16 GB weights + ≈ 10 GB activations per GPU on top of FSDP shards and vLLM, so vLLM gpu_memory_utilization must drop to ≈ 0.4 and the policy needs gradient checkpointing; comfortable at 4B, tight at 8B on 80 GB [SPECULATION].
- **Option C, frozen teacher online.** Same cost as B without the weight sync; dominated by A.
Model-specific caveats: InternVL3.5 read-outs must be stitched across tiles with per-tile normalisation (the thumbnail tile absorbs most global attention); Molmo 2 crops overlap and need overlap-weighted un-tiling, and transformers-native support for Molmo 2 (vs trust_remote_code) must be checked [LIKELY]; Qwen3.5+ hybrids have no attention in the linear layers, so only gradient or intermediate-state read-outs apply there.
Budget: audit ≈ 20 GPU-h across 5 models × 4 tasks; RL with Option A adds ≈ 0; RL with Option B adds 30–50% to a 100–300 GPU-h run.

### 8.12 Gaze and human-attention data (lead question 3; serves IDEA-31 = NF-3)

| Dataset | Task / images | Size | Method | Licence / terms |
|---|---|---|---|---|
| VQA-HAT (2016) | VQA v1 on COCO | ≈ 58k train maps + 4.1k val maps (1,374 val questions × 3) [LIKELY] | mouse-driven de-blurring, not eye tracking | unspecified, research [LIKELY] |
| AiR-D (ECCV 2020 / 2022 ext.) | GQA on Visual Genome / COCO | 1,422 questions, 20 subjects [VERIFIED via R3 + search] | lab eye tracker | research |
| VQA-MHUG (2021) | VQAv2 on COCO | 3,990 stimuli, 49 participants, 3 recordings each [VERIFIED: arXiv 2109.13116] | high-speed lab tracker | paper CC BY-NC-ND; data terms unclear |
| COCO-Search18 (2021) | goal-directed search ("find the X") for 18 categories on COCO | 6,202 images (3,101 target-present / 3,101 target-absent), 10 subjects per category, ≈ 300k fixations; test fixations withheld [VERIFIED: https://sites.google.com/view/cocosearch/] | lab eye tracker | MIT for the data; COCO image terms non-commercial |
| COCO-FreeView | free viewing on the same images | companion set [VERIFIED exists] | lab tracker | as above |
| SALICON (2015) | free viewing, 10k COCO images | 10k train / 5k val [LIKELY] | mouse-contingent | research |
| DriVQA (2025) | driving VQA | size unverified | Tobii Pro X3-120 [VERIFIED snippet] | unknown |
| Gaze-VLM (Oct 2025) | interactive gaze-informed VQA | size unverified (OpenReview blocked the fetch) | unknown | unknown |
Two structural problems: every set is on COCO / Visual Genome images (contaminated for REC), and none is expression-conditioned referring except COCO-Search18 at category level ("find the clock"), which is the closest usable analogue to grounding and the one I would start from.

Realistic cost of 10k webcam-gaze items: WebGazer-class webcam tracking has ≈ 1.5–4° error, i.e. ≈ 60–150 px on a 1080p display at normal viewing distance, and drifts without frequent recalibration [LIKELY]. With 3 participants per item and ≈ 12 s per item including amortised calibration: 10k × 3 × 12 s ≈ 100 participant-hours; at $15/h plus ≈ 20% platform fee ≈ $1.8k, plus ≈ 30% rejects for failed calibration ≈ $2.5k; plus a custom web app (1–2 person-weeks) and IRB (typically 3–6 weeks). A lab tracker cuts item time to ≈ 4 s (≈ 33 subject-hours, ≈ $700) but needs the hardware. A click-based "mark the evidence" proxy (VQA-HAT / PixMo-Points style) costs ≈ $0.05 per click → ≈ $1.5k for 10k × 3 with ≈ 10-px precision, but changes the claim from "gaze" to "human evidence marks". Verdict: money is not the blocker ($2–4k); timeline (6–8 weeks before any training), image contamination (must collect on non-COCO images), and the precision mismatch between webcam gaze (≈ 100 px) and the crops being supervised are. Tier C for Phase 3 unless a partner lab with a tracker exists; the click proxy is the practical fallback.

### 8.13 The RefCOCO-free base model (for the Skeptic's REC demand)

**Answer: the Molmo family — Molmo 2 (4B / 8B; and Molmo2-O 7B, which is fully open end-to-end on Olmo) and Molmo 1.** Their public data cards list PixMo-Cap / PixMo-Points / PixMo-Count / CoSyn-Point / the nine Molmo2 datasets plus academic VQA sets (VQA v2, TextVQA, OK-VQA, A-OKVQA, ChartQA, DocVQA, InfographicVQA, AI2D, AndroidControl, ScienceQA, TabMWP, ST-VQA, TallyQA, DVQA, FigureQA, PlotQA) and contain **no RefCOCO / RefCOCO+ / RefCOCOg, Visual Genome, GRIT, Objects365 or any other box-grounding dataset**; pointing supervision is PixMo-Points (2.3M points on 223k web images) and, for Molmo 2, converted tracking datasets; the Molmo 2 report states all data was built without distilling from proprietary models [VERIFIED: https://arxiv.org/html/2409.17146v2 Table 7; https://arxiv.org/html/2601.10611].
Three caveats the Skeptic will raise, and how to pre-empt them:
1. Annotation-clean is not image-clean: VQA v2, OK-VQA, A-OKVQA and TallyQA use COCO images, so some RefCOCO test images (COCO train2014) were seen with VQA-style annotations. Report both levels and add an image-hash overlap count; use Ref-L4's Objects365 half, RefBench-PRO, GroundingME and HumanRef for image-clean evaluation.
2. Molmo emits points, not boxes. REC on Molmo must be evaluated as point-in-GT-box, or a box format must be taught by the same SFT applied to every base in the comparison (E1 does the latter, so the format is not a confound between bases).
3. Its LLM backbone (Qwen3-8B / Olmo) is text-pretrained only, so no visual leakage there; for the strictest claim use Molmo2-O (Olmo), whose entire stack has a public data card.
Second clean option (Verifier A2, V-012): the PaliGemma / PaliGemma 2 **pretraining-only ("pt") checkpoints** explicitly exclude all transfer datasets, RefCOCO included, and remove near-duplicate images; RefCOCO enters only in the separately released transfer fine-tunes. The pt checkpoints already emit `<locXXXX>` boxes learned on generated open-world detection data, so they are a RefCOCO-clean *box* baseline with a data card (Gemma licence, 3B/10B/28B at 224/448/896) [VERIFIED: Verifier A2, arXiv 2407.07726, 2412.03555]. No other open model with a public data card is cleaner: LLaVA-OneVision lists RefCOCO; Qwen3-VL lists RefCOCO/+/g and Objects365; InternVL2.5/3 list RefCOCO, GRIT and Objects365; Qwen2.5-VL, InternVL3.5, Qwen3.5, Gemma 3/4 and Muse Glimmer publish no usable grounding data card; SmolVLM's Cauldron-based mixture is COCO-heavy and its grounding sources are unlisted [LIKELY].

### 8.14 Standing methodology: contamination control for any REC claim (E1 folded in, per the lead)
Every idea file that claims a grounding improvement should carry these five lines, so the Skeptic's checklist item 2 is answered before it is asked:
1. **Base disclosure line.** State the base model's grounding-data status from Verifier A2: Qwen3-VL (RefCOCO/+/g, COCO, Objects365, OpenImages: Y), InternVL2.5/3 (RefCOCO, GRIT, Objects365: Y), Qwen2.5-VL / InternVL3.5 / Qwen3.5 / Gemma 4 (undisclosed), Molmo 1/2 and PaliGemma pt (documented N).
2. **Two-lineage rule.** Report the headline on one contaminated production base (Qwen3-VL) and one documented-clean base (Molmo2 or PaliGemma-pt), same recipe, same data; the delta between lineages is the memorization share and is reported, not hidden.
3. **Image-clean split.** For every benchmark, report the subset whose images are outside the base's disclosed mixes (Ref-L4 COCO half vs Objects365 half; GroundingME, RefBench-PRO, OpenRef, ScreenSpot-Pro as post-2024 or non-COCO sets), plus an image-hash overlap count against the training pool actually used.
4. **Paraphrase and novel-image probes** (cheap, ≈ 5 GPU-h): re-score on LLM-paraphrased expressions and on a 1k-item post-cutoff pool; a gap larger than the CI between original and paraphrased expressions flags expression memorization.
5. **RefCOCO in the appendix only**, at Acc@0.5 / 0.75 / 0.9 with the cleaned Ref-L4 labels, never as the headline (A4(a)).
The full memorization study (E1 across bases, overlap, paraphrase, novel images) is folded into IDEA-21 §3.3 and is not a standalone paper.

## 9. Claims I am least sure about (sent to Verifier)
1. RESOLVED by Verifier (V-005..V-007): Qwen3-VL reports only RefCOCO-avg (235B 92.1 / 91.9, 32B 91.9, 8B 89.1, 4B 89.0, 2B 85.6); there is no per-split table. Numbers now in §1.1 and §4.
2. Molmo 1 / Molmo 2 data cards contain no RefCOCO-family or box-grounding data — now checked against the paper HTML for both (§8.13), but the check relies on the absence of those names in the listed tables; the Verifier should confirm against the full Table 7 of 2409.17146 and the data appendix of 2601.10611, and check MolmoPoint's fine-tuning mix separately (its blog only names MolmoPoint-GUISyn).
2b. Editor latencies in §8.10 (Mage-Flow-Edit-Turbo 1.02 s on A100, FLUX.2 klein 0.57 s on H100, Qwen-Image-Edit-2511 ≈ 15 s base) come from vendor pages and third-party blogs; the H100 scaling and the Lightning 4-step latency are my extrapolations.
3. Whether Qwen3.5 / 3.6 / 3.8 use 0–1000 or absolute coordinates, and whether EasyR1 / ms-swift / TRL can train the hybrid Gated-DeltaNet architecture today.
4. RESAnything 74.6 gIoU: RESOLVED (val split, training-free pipeline; Verifier V-096). DEViL 43.1 m_vIoU on HC-STVG-v1: still unchecked.
5. The 48.8% RefCOCOg-val image overlap statistic (source paper unidentified).
6. InternVL3.5 weight licence (Apache-2.0 assumed) and coordinate convention.
7. RESOLVED (Verifier V-079): every ScreenSpot-Pro entry on llm-stats, Muse Glimmer-30B 75.4 included, is self-reported with 0 verified; only Gemini 3 Pro 72.7 and Qwen3-VL-235B 62.0 match primary reports. Do not cite llm-stats numbers as facts.
8. RESOLVED (Verifier V-097): TimeLens-8B is not Charades-STA SOTA; VITAL-7B 72.0 / 46.7 / 59.9 (arXiv 2508.04416) and the specialist SG-DETR 71.1 / 52.8 / 60.7 (arXiv 2410.01615) are higher, so a 74.33 R@0.5 for a fine-tuned Qwen2-VL-7B method is plausible. TAR-7B mIoU 61.1 still unchecked.
