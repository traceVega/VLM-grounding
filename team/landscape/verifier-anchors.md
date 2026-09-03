# Verifier anchors (T1.5) — facts the team may lean on

Author: Verifier. Date: 2026-09-01. Every entry states how it was checked:
- **P** = I read the primary source (arXiv HTML/PDF, official doc, official repo/JSON).
- **S** = secondary (search snippet, third-party leaderboard, another paper quoting it).
- **X** = could not access.
Verdicts use the charter vocabulary. Claim IDs (V-xxx) cross-reference `verifications/claims-log.md`.

Reading guide: numbers in tables are copied from the cited table cells; "avg" = RefCOCO/+/g average as defined by the source. Where two sources disagree I say so.

---

## A1. Best reported numbers per benchmark

### RefCOCO / RefCOCO+ / RefCOCOg (REC, Acc@0.5 on boxes)

| Model | RefCOCO val / testA / testB | RefCOCO+ val / testA / testB | RefCOCOg val / test | Source | How |
|---|---|---|---|---|---|
| InternVL2.5-78B (as listed by Qwen) | 93.7 / 95.6 / 92.5 | 90.4 / 94.7 / 86.9 | 92.7 / 92.2 | Qwen2.5-VL report Table 6, arXiv 2502.13923 | P |
| Qwen2.5-VL-72B | 92.7 / 94.6 / 89.7 | 88.9 / 92.2 / 83.7 | 89.9 / 90.3 | same | P |
| Qwen2.5-VL-7B | 90.0 / 92.5 / 85.4 | 84.2 / 89.1 / 76.9 | 87.2 / 87.2 | same | P |
| Grounding DINO (specialist, as listed by Qwen) | 90.6 / 93.2 / 88.2 | 88.2 / 89.0 / 75.9 | 86.1 / 87.0 | same | P |
| Gemini 1.5 Pro (as listed by Qwen) | 73.2 / 72.9 / 74.6 | 62.5 / 63.9 / 65.0 | 75.2 / 76.2 | same | P |
| InternVL3-78B | val 93.4 | val 90.1 | val 91.5 (overall avg 91.4) | InternVL3 report, arXiv 2504.10479 | P |
| InternVL3.5-241B-A28B | overall avg 92.4 | | | InternVL3.5 report Table 7, arXiv 2508.18265 | P |
| Qwen3-VL-235B-A22B | RefCOCO-avg 92.1 (Thinking) / 91.9 (Instruct) | | | Qwen3-VL report Table 2, arXiv 2511.21631 (PDF text) | P |
| Qwen3-VL-32B-Instruct / 8B-Instruct / 4B-Instruct | RefCOCO-avg 91.9 / 89.1 / 89.0 | | | Qwen3-VL report Tables 3–4 | P |
| Gemini 2.5 Pro (as listed by Qwen, "from its report") | RefCOCO-avg 74.6 | | | Qwen3-VL report Table 2 | P (number is second-hand inside that table) |
| GPT-5 (high), as measured by Qwen team | RefCOCO-avg 66.8 | | | Qwen3-VL report Table 2 | P |
| Ovis2.5-9B | 92.7 / 94.3 / 89.7 | 87.7 / 92.1 / 83.6 | 90.3 / 90.1 (avg 90.1) | Ovis2.5 report Table 7, arXiv 2508.11737 | P |
| "Qwen3.6 Plus" 93.5 avg | | | | llm-stats.com/benchmarks/refcoco-avg (9 entries, all self-reported, "0 verified", updated 2026-09-02) | S — do not cite as fact |

Saturation verdict: **VERIFIED-saturated on val/testA** (top open models 92–96, spread between the top five is ~1–2 points, which is inside the label-noise band: Ref-L4 paper reports labeling error rates of 14% / 24% / 5% for RefCOCO / RefCOCO+ / RefCOCOg [P, arXiv 2406.16866]). **Not fully saturated on RefCOCO+ testB** (best open ~84–87). Proprietary models score far lower when prompted for boxes (Gemini 2.5 Pro ~75 avg, GPT-5 ~67 avg per Qwen's measurement), which mostly reflects output-format/API mismatch, not perception. See claim (a) in A4.

### Ref-L4 (Acc@0.5, 45,341 annotations, 365 categories, avg 24.2 words)

| Model | Ref-L4 Acc@0.5 | Source | How |
|---|---|---|---|
| CogVLM-Grounding-17B | 81.70 (README: 81.699) | Ref-L4 paper Table 4, arXiv 2406.16866; github.com/JierunChen/Ref-L4 | P |
| SPHINX-v2-1k | 81.31 | Ref-L4 paper | P |
| Qwen2.5-VL-7B | 81.24 | RefBench-PRO Table 3, arXiv 2512.06276 (third-party measurement; split/metric not stated in caption) | P (of RefBench-PRO) |
| VLM-R1-3B | 82.56 | same | P |
| Qwen3-VL-8B | 81.70 | same | P |
| Ovis2.5-9B | 90.29 | same | P (single third-party measurement; Ovis2.5's own report does not report Ref-L4) |

Verdict: best number I could verify is **~82 (2024–2025 models)**, with one third-party outlier at 90.3 for Ovis2.5-9B (PARTIALLY: single source, protocol unspecified). Neither Qwen2.5-VL nor Qwen3-VL reports Ref-L4. **Not saturated.** No official leaderboard beyond the paper table.

### gRefCOCO (GRES; gIoU / cIoU / N-acc, val split)

| Method | val gIoU / cIoU / N-acc | Source | How |
|---|---|---|---|
| ReLA (2023 baseline) | 63.60 / 62.42 / 56.37 | quoted in AnchorSeg Table 3, arXiv 2604.18562 | P |
| GSVA-7B | 66.47 / 63.29 / 62.43 | same | P |
| AnchorSeg (LLaVA-1.5-7B + SAM ViT-H) | 74.76 / 68.68 / 76.00 (testA 75.75/73.43/71.78; testB 68.25/64.56/68.27) | AnchorSeg, arXiv 2604.18562 | P |
| STAMP-7B (Qwen2-7B, with SAM-based decoder) | val gIoU 73.6 / cIoU 77.6; "74.8" headline = cIoU averaged over splits | STAMP, arXiv 2512.00395 | P |
| Text4Seg InternLM2.5-7B | val 74.4 gIoU / 69.1 cIoU | quoted in STAMP Table 3 | P |

Verdict: **Not saturated**; gIoU ~74–75, no-target accuracy ~70–76 (a quarter of "no target" cases still fail). Metric conventions differ across papers (cIoU vs gIoU headline), so compare carefully.

### ReasonSeg (gIoU / cIoU)

| Method | val | test | Notes | Source | How |
|---|---|---|---|---|---|
| LISA-13B | 57.7 / 60.3 | 53.8 / 50.8 | fine-tuned | quoted in arXiv 2604.02040 Table 1 | P |
| Seg-Zero-7B | 60.9 / 57.4 (other paper: 62.6 / 62.0) | 57.7 / 54.4 | RL, Qwen2.5-VL-7B | 2604.02040; 2606.09303 (numbers differ between papers) | P |
| WISE-7B | 63.5 / 59.2 | 60.3 / 58.5 | | arXiv 2604.02040 | P |
| AnchorSeg-13B | 67.9 / 73.0 | 67.7 / 68.1 | fine-tuned LLaVA-1.5-13B | arXiv 2604.18562 | P |
| Rea2Seg (Qwen2.5-VL-3B, Top-3) | 68.4 / 70.0 | 66.6 / 65.5 | trained on 16K CoT data | arXiv 2606.09303 | P |
| RESAnything (training-free: Pixtral-12B or Claude 3.5 Sonnet + SAM ViT-H + CLIP) | 74.6 / 72.5 | not reported | zero-shot pipeline | arXiv 2505.02867 | P |

Verdict: **Not saturated** (trained SOTA ~68 gIoU val, ~67 test). The val split is small (about 200 samples [LIKELY: from LISA paper, not re-read here]) so 1–2 point differences are noise. Beware mixing training-free pipeline numbers (RESAnything 74.6 val) with trained-model numbers.

### ScreenSpot-v2 (GUI grounding, avg accuracy)

| Model | Score | Source | How |
|---|---|---|---|
| Jedi-7B | 91.7; UI-TARS-7B 91.6; UI-TARS-72B 90.3 | Jedi paper Table 2, arXiv 2505.13227 | P |
| InternVL3.5-241B-A28B | 92.9 | arXiv 2508.18265 | P |
| MAI-UI-32B | 96.5 | MAI-UI report, arXiv 2512.22047 | P |
| UI-Venus-72B | 95.28 (leaderboard top); duvo-eye-1 95.05; Holo2-30B-A3B 94.89 | gui-agent/grounding-leaderboard results/screenspot_v2.json | P (JSON read; entries are self-submitted, undated) |

Verdict: **VERIFIED-saturated** (>95, remaining errors are largely annotation-level). Use ScreenSpot-Pro / OSWorld-G / UI-Vision instead.

### ScreenSpot-Pro (high-res professional GUI grounding)

| Model | Score | Source | How |
|---|---|---|---|
| Qwen2.5-VL-72B | 43.6 | Qwen2.5-VL report Table 9 | P |
| Jedi-7B 39.5; UI-TARS-72B 38.1; Operator 36.6 | | Jedi paper Table 3 | P |
| Qwen3-VL-235B-A22B | 61.8 (Thinking) / 62.0 (Instruct); 32B-Instruct 57.9; 4B-Instruct 59.5; 8B-Instruct 54.6 | Qwen3-VL report Tables 2–4 | P |
| GUI-C2-3B / 7B | 46.4 / 50.8 (small-model SOTA claims) | arXiv 2605.30884 | P |
| MAI-UI-32B + zoom-in 73.5; Seed1.8 73.1; Gemini-3-Pro 72.7; MAI-UI-32B (no zoom) 67.9; UI-Venus-72B 61.9 | | MAI-UI report, arXiv 2512.22047 | P |
| Chain-of-Ground (Qwen3-VL-235B, iterative, training-free) 68.4 | | arXiv 2512.01979 | S (search snippet) |
| Community leaderboard top: Indeed-UI-32B-zoomin 82.73; Holo2-235B-A22B 81.47; KV-Ground + Qwen3.5-27B router 80.90; KV-Ground-GuiOwl1.5-8B-ZoomIn 80.52 | | results/screenspot_pro.json in gui-agent/grounding-leaderboard | P (JSON read; self-submitted, undated, many entries use multi-step zoom) |

Verdict: **Not saturated but climbing fast**: from 18.9 (Jan 2025) to ~62 (single-shot flagship, Nov 2025) to ~73 (tech reports with zoom, Dec 2025) to ~81–83 (community entries, 2026). Almost all top entries use zoom-in / multi-step refinement, so "single-pass grounding" and "agentic grounding" numbers are no longer comparable. Text vs icon split still matters (icons lag).

### OSWorld-G (564 samples; includes 54 refusal cases)

| Model | Score | Source | How |
|---|---|---|---|
| Jedi-3B 50.9; Jedi-7B 54.1; UI-TARS-7B 47.5; UI-TARS-72B 57.1; Qwen2.5-VL-32B 46.5; Gemini-2.5-Pro 45.2; Seed1.5-VL 62.9 | | Jedi paper Table 5, arXiv 2505.13227 | P |
| InternVL3.5-241B-A28B | 53.2 | arXiv 2508.18265 | P |
| Qwen3-VL-235B-A22B | 68.3 (Thinking) / 66.7 (Instruct); 32B-Instruct 65.1; 8B-Instruct 58.2 | Qwen3-VL report | P |
| MAI-UI-32B + zoom 70.9; UI-Venus-72B 70.4; MAI-UI-32B 67.6; GTA1-32B 65.2 | | MAI-UI report, arXiv 2512.22047 | P |

Verdict: **Not saturated** (~71; UI-Venus-2-27B reports 79.1 on the "OSWorld-G-R" variant [P, github.com/inclusionAI/UI-Venus], which is not the same split). Agent-level effect verified [P, Jedi paper]: with a GPT-4o planner, OSWorld success goes 5.0 → 24.0 (Jedi-3B) → 27.0 (Jedi-7B); with an o3 planner and Jedi-7B it reaches 51.0. Quote the planner when citing either number.

### Charades-STA (video temporal grounding; R@0.5 / R@0.7 / mIoU)

| Method | R@0.5 / R@0.7 / mIoU | Type | Source | How |
|---|---|---|---|---|
| SG-DETR (InternVideo2-1B features) | 70.2 / 49.5 / 59.1; with pretraining 71.1 / 52.8 / 60.7 | specialist, non-LLM | arXiv 2410.01615 | P |
| Mr.BLIP | 69.3 / 49.3 / 58.6 | | quoted in SG-DETR Table 4 | P |
| VITAL-7B (Qwen2.5-VL-7B, RL, trained on Charades-STA train) | 72.0 / 46.7 / 59.9 | MLLM | arXiv 2508.04416 | P |
| TimeLens-8B | 63.0 / 35.2 / 55.2 | MLLM, fine-tuned | arXiv 2512.14698 | P |
| Gemini-2.5-Pro | 61.1 / 34.0 / 52.8; GPT-5 42.0 / 22.0 / 40.5 | proprietary, prompted | TimeLens Table 1 | P |
| Qwen3-VL-235B-A22B-Instruct | mIoU 64.8 (Thinking 63.5); 8B-Thinking mIoU 59.9 | MLLM | Qwen3-VL report | P |
| AVI (training-free agent) | 69.0 / 37.6 / 60.0 | | arXiv 2511.14446 | P |

Verdict: **Not saturated at R@0.7** (best ~53, specialist). MLLMs now match specialists at R@0.5 but lag at R@0.7 (boundary precision). Caution: many MLLM papers report only mIoU; and Charades-STA is small/noisy, with known annotation issues [LIKELY: widely discussed, not re-verified].

### ScanRefer (3D REC; Acc@0.25 / Acc@0.5 overall)

| Method | Acc@0.25 / Acc@0.5 | Source | How |
|---|---|---|---|
| Chat-Scene 55.5 / 50.2; Video-3D LLM 58.1 / 51.7; Ross3D 61.1 / 54.4 | | quoted in SmartMage Table 1, arXiv 2608.05137 | P |
| SmartMage (Aug 2026) | 65.9 / 59.5 (Multi3DRefer F1@0.25/0.5: 65.4 / 60.7) | arXiv 2608.05137 | P |
| Zero-shot VLM pipelines (e.g., TAB 71.2 / 46.4; MCM-VG 62.0 / 53.6) | | arXiv 2604.00528, 2604.26261 | S (search snippets, not read) |

Verdict: **Not saturated** (best verified Acc@0.5 = 59.5). I did not verify classic supervised detectors (e.g., ConcreteNet/3D-VisTA) in this pass; treat any "specialist SOTA" numbers a teammate cites as [LIKELY] until checked.

### PointArena / Point-Bench and Where2Place (pointing)

| Model | Point-Bench avg (%) | Source | How |
|---|---|---|---|
| Human 89.1; MolmoPoint-8B 70.7; Molmo2-8B 68.7; Molmo-72B 63.8; Gemini-2.5-Pro 62.8; Qwen3-VL-235B 58.3 | | MolmoPoint Table 1, arXiv 2603.28069 | P |
| Poivre-7B (Qwen2.5-VL-7B + RL self-refinement) 67.5; Qwen2.5-VL-7B 56.3 | | arXiv 2509.23746 | P |
| PointArena original: Molmo-72B beat Gemini-2.5-Pro by 0.43 pt (p≈0.29, not significant); Point-Bench = 982 image–question pairs, 5 categories | | arXiv 2505.09990 | P |

Where2Place (free-space pointing, accuracy = point inside GT mask): Molmo-72B 63.8; Gemini Robotics-ER 45.0; Gemini 2.0 Flash 33.8; GPT-4o 20.6; Claude 3.5 Sonnet 16.2 [P, Gemini Robotics paper Table 3, arXiv 2503.20020, Mar 2025]. Gemini Robotics-ER 1.5 (Sep 2025) claims "most precise VLM for pointing" but the blog gives no per-benchmark numbers [P, developers.googleblog.com]. Verdict: **Not saturated** (~71 vs human 89); no maintained 2026 leaderboard for Where2Place found (UNVERIFIABLE current SOTA).

### Hallucination: POPE and AMBER

- POPE (adversarial, COCO) F1: InternVL3-8B 90.50; InternVL2.5-8B 89.88; Qwen2.5-VL-7B 87.08 [P, HOPE paper Table 2, arXiv 2508.06530]. HOPE reports "POPE is becoming increasingly outdated" and its harder probes drop precision by 9–23 points [P].
- RePOPE relabeling: among POPE "Yes" questions, 9.3% wrong labels and 13.8% ambiguous; among "No" questions 1.7% wrong and 4.3% ambiguous; "InternVL2.5-8B or -26B drop to the bottom of the ranking" after relabeling [P, arXiv 2504.15707].
- AMBER: AMBER Score = ½(1 − CHAIR + F1); original paper best GPT-4V 91.4, Qwen-VL 89.7 [P, arXiv 2311.07397]. No maintained leaderboard exists (GitHub README shows an image only) [P]. Current-model AMBER numbers vary by decoding protocol across papers → single SOTA number is UNVERIFIABLE.

Verdict: **POPE is saturated and label-noisy (VERIFIED)**; treat ±2 F1 on POPE as meaningless. AMBER usable but fragmented.

---

## A2. Training-data contamination (what the tech reports actually say)

Legend: Y = named in the training mixture; N = explicitly excluded; — = not mentioned (absence of evidence, not evidence of absence).

| Model (report) | RefCOCO/+/g | Visual Genome | Flickr30k | GRIT | Objects365 | ScreenSpot-type GUI data | How |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL (2502.13923) | — (only in eval tables; grounding data described as "publicly available datasets and proprietary data" + Grounding DINO/SAM synthesis) | — | — | — | — | Y (own screenshots on mobile/web/desktop with synthetic UI grounding labels; ScreenSpot itself only in eval) | P |
| Qwen3-VL (2511.21631) | **Y** ("aggregating widely used open-source datasets, including COCO, Objects365, OpenImages, and RefCOCO/+/g") + PixMo points + synthesis via Qwen2.5-VL & Grounding DINO | — | — | — | **Y** | — (GUI data not itemised) | P |
| InternVL2.5 (2412.05271; InternVL3 states it added no new grounding data on top of this) | **Y** (pre-train: Objects365, GRIT, RefCOCO, GPT4Gen-RD-BoxCoT, All-Seeing-V1/V2, V3Det, TolokaVQA; fine-tune: RefCOCO/+/g, All-Seeing-V2, V3Det, DsLMF, COCO-ReM, TolokaVQA) | — | — | **Y** | **Y** | Y (Screen2Words, WebSight, Widget-Caption, RICOSCA, SeeClick, ScreenQA, AMEX, AITW, Odyssey, UIBert, AndroidControl, Mind2Web, OmniACT, WaveUI) | P |
| InternVL3 (2504.10479) | inherits above; quote: "InternVL3's training data expansion does not include additional grounding-specific data" | | | | | | P |
| InternVL3.5 (2508.18265) | not disclosed (only evaluation on RefCOCO/+/g) | — | — | — | — | — | P |
| Molmo (2409.17146) | **N** (no RefCOCO or external grounding sets; pointing from PixMo-Points 2.3M; "all constructed without the use of VLMs") | — | — | — | — | Y (AndroidControl in academic mixture) | P |
| PaliGemma / PaliGemma 2 (2407.07726, 2412.03555) | **N in pretraining** ("We do not use any of our transfer datasets during pretraining, and furthermore remove all near-duplicates of their images"); RefCOCO used only in transfer fine-tuning; pretraining has detection/segmentation on "generated open-world data" and grounded captioning (LocCa-style) on WebLI; object-centric VQA on OpenImages | — | — | — | — | — | P |
| Florence-2 (2311.06242) | fine-tune only; FLD-5B images come from ImageNet-22k, Objects365, Open Images, Conceptual Captions, LAION; Flickr30k Entities used in fine-tuning; 1,000 location bins | — | Y (fine-tune) | — | **Y (images)** | — | P |
| Gemma 3 (2503.19786) | — (no vision datasets named; only "We decontaminate evaluation sets from our pre-training data mixture"; no box/point output described) | — | — | — | — | — | P |

Consequences for the team:
1. Any "zero-shot RefCOCO" claim for Qwen3-VL, InternVL2.5/3, Ovis, or most open models is REFUTED by their own reports (RefCOCO train is in the mix). Only Molmo and PaliGemma (pretraining) are documented as RefCOCO-clean; Qwen2.5-VL is undisclosed.
2. RefCOCO images are COCO train2014; COCO is in Qwen3-VL and (via Objects365/COCO-ReM) InternVL mixtures, so even held-out splits share images with training captions/detections.
3. Nobody documents ScreenSpot test contamination, but Qwen2.5-VL/Qwen3-VL/InternVL train on large synthetic GUI grounding corpora; ScreenSpot-v2 saturation is partly a data-scale effect.

---

## A3. Capability facts (output formats)

| Item | Fact | Source | How |
|---|---|---|---|
| Qwen2.5-VL coordinates | Absolute pixel coordinates of the (resized) input image: "Qwen2.5-VL uses coordinate values based on the actual dimensions of the input images during training to represent bounding boxes and points." | arXiv 2502.13923 | P |
| Qwen3-VL coordinates | "Different from Qwen2.5-VL, we adopt a normalized coordinate system scaled to the range [0, 1000] in this version." | arXiv 2511.21631 §3.2.4 | P |
| Gemini API boxes | `box_2d` = `[ymin, xmin, ymax, xmax]` "normalized to 0-1000"; docs currently exemplify `gemini-3.7-flash` | ai.google.dev/gemini-api/docs/image-understanding (read 2026-09-01) | P |
| Gemini API segmentation | JSON items with `box_2d`, `mask`, `label`; the current doc describes `mask` as a polygon of [x,y] normalized to 0-1000 (the 2025 doc described base64 PNG masks; the format appears to have changed; re-check before building on it) | same | P |
| Gemini pointing | Gemini Robotics-ER (`gemini-robotics-er-2-preview`, 1.6 deprecated): points as `[{"point": [y, x], "label": ...}]`, `[y, x]` normalized 0-1000; boxes `[y_min, x_min, y_max, x_max]`; also trajectories | ai.google.dev/gemini-api/docs/robotics-overview; developers.googleblog.com (ER 1.5 post, 2025-09-25) | P |
| Gemini 3 pointing in the general API | Not documented on the image-understanding page (PARTIALLY: pointing is documented only for Robotics-ER models) | same | P |
| Molmo pointing | `<point x="10.0" y="10.0" alt="...">text</point>` / `<points x1= y1= x2= y2= ...>`; coordinates scaled 0–100; points ordered top-down, left-right | arXiv 2409.17146 App. B.2 | P |
| MolmoPoint (2026) | replaces digit coordinates with 3 grounding tokens `<PATCH><SUBPATCH><LOCATION>` (3x3 grid inside a sub-patch); 3 tokens per coordinate instead of 8 | arXiv 2603.28069 | P |
| GPT-5 / o3 "thinking with images" | o3/o4-mini crop, zoom, rotate images inside chain-of-thought (OpenAI post 2025-04-16; openai.com returned HTTP 403 to me, so this is S via CNBC/the-decoder). OpenAI's vision guide (P) states "The model struggles with tasks requiring precise spatial localization, such as identifying chess positions" and offers no box/point output API; coordinates are only mentioned for computer-use ("map returned coordinates back to the original image"). Measured box grounding when prompted: RefCOCO-avg 66.8 (GPT-5 high, Qwen3-VL Table 2), RefBench-PRO 26.1 (arXiv 2512.06276), Charades-STA R@0.5 42.0 (TimeLens). | developers.openai.com/api/docs/guides/images-vision; 2511.21631; 2512.06276; 2512.14698 | P/S as marked |
| Claude computer use | "Coordinates are in screenshot pixels" with origin top-left; after a `zoom` action Claude "still expresses coordinates in the full screenshot's space"; if you downscale screenshots you must scale coordinates back up; recommended 1024x768 / 1280x720 (desktop) or 1280x800 / 1366x768 (web); current toolset `computer_toolset_20260801` (Claude 5 family, Opus 4.8); Opus 4.5–4.7 / Sonnet 4.6 use `computer_20251124` with a beta header | platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool | P |

---

## A4. Five commonly repeated claims

### (a) "RefCOCO is saturated" — verdict: **VERIFIED (with a caveat)**
For: top open models are at 92–96 on val/testA with ~1–2 point spread [P: Qwen2.5-VL, InternVL3, Qwen3-VL tables]; Ref-L4 paper documents 14% / 24% / 5% labeling error in RefCOCO / + / g and states "many LMMs surpassing 90% accuracy" [P, 2406.16866]; RefBench-PRO: "most models have reached performance saturation on RefCOCO" yet none exceed 71% on RefBench-PRO, and Ref-Adv shows Qwen2.5-VL-72B falling from 92.7 (RefCOCO) to 54.1 (Ref-Adv) [P, 2512.06276; 2602.23898].
Against: RefCOCO+ testB is still at 84–87 for the best open models [P]; proprietary models are far from saturated on it (GPT-5 ~67 avg) but that is an output-format problem; and the label-noise figures mean a "ceiling" is not well defined, so saturation is partly an artifact of noisy labels rather than solved perception.
Practical reading: RefCOCO val/testA cannot discriminate between frontier models; the field has moved to Ref-L4, Ref-Adv, RefBench-PRO, gRefCOCO, and GUI benchmarks.

### (b) "Grounding training reduces hallucination" — verdict: **REFUTED as a general claim; PARTIALLY for narrow settings**
For: Ferret abstract claims "a remarkable alleviation in object hallucination" [P, 2310.07704]; decoding-time visual-grounding methods (M3ID etc.) improve POPE [S].
Against: Geigle, Timofte, Glavaš (2024) test the claim directly and find "grounding objectives have little to no effect on object hallucination in open caption generation", attributing earlier positive results to evaluation on training-like QA formats rather than open generation [P, 2406.14492]. POPE itself is saturated and label-noisy (RePOPE) so POPE deltas are weak evidence [P].
Reading: no controlled evidence that adding box/point supervision lowers hallucination in free-form generation; the claim should be stated as a hypothesis to test, not a premise.

### (c) "RL with IoU rewards beats SFT for grounding" — verdict: **PARTIALLY (true for small data / OOD, unproven at scale)**
For: VLM-R1 (Qwen2.5-VL-3B, same data, 600 steps): RefCOCO val 90.55 (RL) vs 88.7 (SFT); RefCOCO+ 84.3 vs 82.25; LISA-Grounding OOD 63.16 vs 54.82 (SFT actually dropped below the 56.51 base); OVDEval 31.01 vs 26.50 [P, 2504.07615]. Visual-RFT reports similar small-data gains [S]. VITAL and other RL video grounders lead on Charades-STA R@0.5 [P].
Against / limits: (1) all these comparisons start from a base (Qwen2.5-VL) that was already trained with large-scale grounding SFT, so RL is polishing, not teaching; the flagship reports (Qwen3-VL, InternVL) still rely on SFT-scale grounding data. (2) H-GRPO finds "stronger models benefit substantially from SFT alone, but explicit grounding rewards remain useful for localization-sensitive tasks" and that vanilla GRPO gives "only marginal or unstable gains" for a 2.2B model [P, 2606.29915]. (3) VLM-R1 documents reward hacking in detection (predicting all categories) [P]. (4) ExpVG's design-space study (SFT only) shows format choices alone (normalized integer coords) move RefCOCO by several points [P, 2508.08066], so RL-vs-SFT deltas of 1–2 points in-domain are within format effects. (5) I found no controlled study comparing RL and SFT at equal, large data scale.
Reading: safe to claim "RL improves OOD robustness at small data"; unsafe to claim RL beats SFT in general.

### (d) "Attention maps inside VLMs localize the referent" — verdict: **PARTIALLY (true for a few heads / on average; unreliable per-sample and not a faithfulness guarantee)**
For: "MLLMs Know Where to Look" (ICLR 2025): MLLMs "consistently know where to look, even when they provide the wrong answer"; attention/gradient maps drive training-free crops that raise accuracy [P, 2502.17422]. "Your LVLM only needs a few attention heads": three localization heads in LLaVA-1.5-13B give RefCOCO 87.2 val / 90.0 testA / 83.3 testB training-free, close to Shikra-13B [P, 2503.06287]. Attention-driven GUI grounding without fine-tuning (AAAI) and GUI-AIMA exist [S].
Against: the average attention map is "uninformative for localization"; only selected heads work and selection needs 1,000 labelled pairs [P, 2503.06287]; the method fails for models that pool spatial tokens and for multi-object cases [P]; "When Looking Is Not Enough": "the model may still assign substantial attention mass to image tokens while internally drifting toward an incorrect answer" [P, 2605.11559]; attention sinks can be exploited to induce hallucination [S, 2501.15269]. Caveat on the "training-free" numbers: LLaVA-1.5's instruction mix contains RefCOCO/VG region data [LIKELY], so its heads were shaped by grounding supervision.
Reading: attention is a usable weak localizer (good for interventions, cropping, pseudo-labels), not a faithful explanation and not a substitute for explicit grounding output.

### (e) "CLIP cannot localize" — verdict: **REFUTED as stated; PARTIALLY true of vanilla inference**
For: vanilla CLIP dense inference is very poor: average 12.6 mIoU over VOC20/Context59/COCO-Stuff/Cityscapes/ADE (ADE 2.1, COCO-Stuff 4.4) [P, ClearCLIP, 2407.12442]; the paper attributes noise to the residual connection and image-level contrastive training.
Against: MaskCLIP (ECCV 2022 oral) extracts dense labels from frozen CLIP "in the absence of annotations and fine-tuning" using the value embeddings of the last attention block [P, 2112.01071]; SCLIP / ClearCLIP-style training-free changes to the last layer raise the average to 37.5 mIoU (VOC20 80.9) with zero training [P, 2407.12442]; PaliGemma/Gemini/Qwen build detection on CLIP-like (SigLIP) encoders.
Reading: CLIP's patch features carry localization information; the last layer's global pooling and residual stream hide it. The accurate claim is "CLIP's default output is not localized" rather than "CLIP cannot localize".

---

## A5. Things I could not verify (UNVERIFIABLE / X)
- OpenAI's own "Thinking with images" post and o3 system card (HTTP 403 from openai.com); relied on the developer docs (P) and press (S).
- Current Where2Place and AMBER leaderboards (none maintained).
- Any official Ref-L4 number for Qwen3-VL / Qwen2.5-VL (their reports omit it).
- Classic supervised specialist SOTA on ScanRefer (not read this pass).
- Whether Gemini 3 (non-robotics) exposes point output in the general API (docs do not say).
