# Survey D — what frontier and open labs actually did about absent-target grounding (2026-09-16)

Author: `SurveyD`. Sources: primary tech reports / model cards / docs, PDFs extracted with pypdf where the HTML summariser truncated (Qwen2.5-VL, Qwen3-VL, Seed1.5-VL, Seed2.0, InternVL3.5, SAM 3, Jedi, Ferret, PaliGemma 2, GroundingME, RefBench-PRO, OpenRef, PR-Bench, Molmo2). Labels: `[V: id]` = read the primary text; `[V-html: id]` = read via WebFetch summary of the HTML (exact wording quoted where given); `[L]` = likely; `[S]` = speculation; `[P]` = project measurement (charter).

## 0. Method and one correction to the brief

- Every report below was grepped (PDF text) or asked (HTML) for: non-exist*, negative, reject*, refus*, absent, "no object", hallucin*, null, None, and for the grounding data section and grounding benchmark tables.
- **Correction**: the brief says both Qwen reports mention non-existent-object data. Only **Qwen2.5-VL** does (one sentence, §2.2.1 data). The **Qwen3-VL** report (42 pp., 149,701 chars extracted) has no grounding hit for any of those keywords; §3.2.4 "Grounding and Counting" describes only positive data synthesis. `[V: 2502.13923, 2511.21631]`
- Benchmark prompts and scoring rules were re-read from the benchmark PDFs, because the "how is none expressed / does the scorer accept it" question is decided there, not in the model reports.

## 1. Model table

Columns: negatives in grounding training mix | how built | "none" output format the model was trained to emit | rejection benchmarks the **lab itself** reports | thinking vs non-thinking | what the report says about over-refusal / trade-off. Third-party rejection numbers for each model are in §2.

| Model | Report | Negatives in mix | How built | "None" format | Rejection benchmarks reported by lab | Thinking vs non | Over-refusal / trade-off statement |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL | 2502.13923 `[V]` | **yes (one sentence)** | "to improve the model's effectiveness in extreme object detection scenarios, we synthesized non-existent object categories within the queries and constructed image data containing multiple instances for each object." Category-level, detection-style; target output not described | not stated (boxes are absolute-pixel JSON `bbox_2d`; the null convention is undocumented) | none (RefCOCO/+/g, ODinW, PointGrounding, CountBench only) | n/a | none |
| Qwen3-VL | 2511.21631 `[V]` | **not stated** (no keyword hit) | §3.2.4: COCO, Objects365, OpenImages, RefCOCO/+/g + 3-stage synthesis (candidates from Qwen2.5-VL → Grounding DINO + Qwen2.5-VL localise → filter low-confidence). Points: PixMo + synthetic. Coords normalised [0,1000] | not stated | none (Table 16: RefCOCO/+/g, ODinW-13, CountBench; 3D Omni3D) | report gives no grounding numbers by mode; GroundingME (third party) does, see §2 | none |
| Qwen3.5 (native VL, Feb 2026) | HF card Qwen3.5-2B `[V-html]` | not stated | card only | not stated (JSON `bbox_2d` cookbook format `[L]`) | none (RefCOCO-avg 84.3, CountBench 86.8, ODinW13 40.5, ScreenSpot-Pro 54.5 for 2B) | n/a | none |
| Qwen3.6-27B | HF card `[V-html]` | not stated | card only ("Causal Language Model with Vision Encoder"; RefCOCO-avg 92.5) | not stated | none | n/a | none |
| InternVL3 | 2504.10479 `[V-html]` | no | "InternVL3's training data expansion does not include additional grounding-specific data" | not stated | none (RefCOCO/+/g) | n/a | none |
| InternVL3.5 | 2508.18265 `[V]` | **not stated**; the only "negative samples" phrase is about RL ("the core advantage of RL lies in its ability to introduce negative samples") | — | not stated | none (Table 7 RefCOCO/+/g; 241B-A28B 92.4 overall) | grounding evaluated without test-time scaling; no per-mode grounding numbers | none |
| Gemini 3 / 3.1 / 3.8 | 3.1 Pro model card 2026-02-19 `[V-html]`; API image-understanding docs `[V-html]` | not stated | — | JSON list of `{"box_2d":[ymin,xmin,ymax,xmax] (0-1000), "label"}`; absent case undocumented (empty list `[S]`); docs: "For better results, disable thinking by setting the thinking level to 'minimal'" (segmentation) | none in the card; card has no grounding eval at all | docs recommend thinking off for masks | none |
| GPT-5.x | system cards blocked (openai.com 403); dev docs images guide `[V-html: developers.openai.com]` | not stated `[L]` | — | none documented; docs: "The model struggles with tasks requiring precise spatial localization, such as identifying chess positions." | none `[L: search of 5/5.2/5.4/5.5 card coverage shows no grounding section]` | n/a | none |
| Claude (Opus 4.x-5, Sonnet 4.6, Fable/Mythos 5.x) | vision + coordinates + computer-use docs `[V-html: platform.claude.com]` | not stated | — | absolute-pixel `[x1,y1,x2,y2]` or points, prose or structured-output JSON; "Claude does not work well when you ask for normalized coordinates"; no absent-object convention; computer-use doc only specifies `is_error` for failed actions | none; docs: "Spatial reasoning: Claude's coordinate and localization outputs are approximate" | n/a | none |
| Molmo (2024) | 2409.17146 `[V-html]` | **yes** | "We also collected 'not present' data so models can learn to handle cases where an item is not in the image." (count not given) | **text**: eval counts P=R=1 "if the model responds correctly (e.g. outputs 'This isn't in the image.')" | pointing eval includes no-target cases, no separate number | n/a | none |
| Molmo2 (2026) | 2601.10611 `[V]` | inherits PixMo-Points/PixMo-Count/CoSyn-Point in pre-training and SFT; no "not present" passage of its own `[L: inherits]` | — | as Molmo `[L]` | none for absence; counting via pointing (Table 9a) | n/a | none |
| MolmoPoint (2026) | 2603.28069 `[V-html]` | structural: **no-more-points class** with a fixed key embedding the `<PATCH>` token can attend to; "if the model chooses to generate a `<PATCH>` token, it is forced to select a point, even if none of the scores in s_p are high" | — | zero `<PATCH>` tokens (the end token can come first) | none for absent targets | n/a | "Removing the no-more-points token hurts performance and more than doubles the amount of overcounting" (negatives help positives) |
| SAM 3 | 2511.16719 `[V]` | **yes, hard** | data engine: "a Llama-based pipeline that also proposes hard negative NPs adversarial to SAM 3"; "Hard negatives are phrases that are not present in the image but that (a previous generation of) SAM 3 predicts masks for"; SA-Co benchmark "over 3M media-phrase pairs with hard negative labels"; SA-Co/EXT "enriched with hard negatives using our ontology pipeline" | **presence token**: "solely responsible for predicting whether the target concept ... is present ... The final score for each proposal query is the product of its own score and the presence score" → empty mask set | SA-Co IL_MCC (image-level presence MCC), cgF1 | n/a | Table 9b: hard negs/img 0→5→15→30: cgF1 28.3→39.4→41.8→43.0, IL_MCC 0.44→0.62→0.67→0.68, **pmF1 62.4→62.9→62.4→62.8 (flat)**. Table 9a presence head: cgF1 50.7→52.2, IL_MCC 0.77→0.82, **pmF1 65.4→63.4 (−2.0)**. Human: cgF1 72.8 / IL_MCC 0.94 vs SAM 3 54.0 / 0.82 |
| Gemma 4 | 2607.02770 `[V-html]`; HF launch blog `[V-html]` | not stated (report has no grounding section or benchmark) | — | native JSON `[{"box_2d":[...0-1000...],"label":...}]`; absent case undocumented | none | n/a | none |
| Seed1.5-VL | 2505.07062 `[V]` | **not stated** | §3.1.3: Objects365/OpenImages/RefCOCO filtered by previous VLM (48M), Grounding DINO auto-annotation on web captions (200M), PixMo-Points + Molmo/CountGD pipeline (170M), counting 8M; coords [0,999] | RL prompt asks for `<bbox>...</bbox>`, IoU reward; none not described | none (RefCOCO-avg 91.6, LVIS-MG 73.8, CountBench 93.7, FSC-147) | thinking/non-thinking reported per benchmark for positives only | none |
| Seed1.8 / Seed2.0 cards | 2603.20633 `[V-html]`, 2607.00248 `[V]` | not stated | — | not stated | none (ScreenSpot-Pro 64.3 / 73.1 with crop-box tool; CountBench 96.3; PointBench) | n/a | none |
| Kimi-VL | 2504.07491 `[V-html]` | not stated | — | not stated | none (ScreenSpot-V2 92.8, -Pro 34.5) | n/a | none |
| GLM-4.1V-Thinking / 4.5V | 2507.01006 `[V-html]` | **no** (filter keeps only samples "containing at least two valid bounding boxes") | LAION-115M + GLIPv2 pseudo-boxes; 140M GUI REC/REG pairs from CommonCrawl screenshots | `<|begin_of_box|>…<|end_of_box|>`, "only one boxed span is acceptable"; none not described | none (RefCOCO-avg 91.3, TreeBench 50.1, Ref-L4 89.5) | GroundingME third-party only | RL grounding reward "#boxes with IoU > τ divided by total boxes" — undefined for a no-box target |
| GLM-5V-Turbo | 2604.26752 `[V-html]` | not stated | — | not stated | none (RL +4.8 RefCOCO-avg over SFT) | n/a | none |
| MiMo-VL-7B | 2506.03569 `[V-html]` | not stated | element/instruction grounding, absolute coords | not stated | reports **"OSWorld-G (no_refusal) 56.1"** — refusal items excluded from the headline | GroundingME third-party | none |
| UI-TARS | 2501.12326 `[V-html]` | no | unified public GUI grounding sets | click coordinate | none (ScreenSpot-Pro 38.1) | n/a | none |
| UI-TARS-2 | 2509.02544 `[V-html]` | no | — | — | no grounding benchmark at all | n/a | none |
| Jedi | 2505.13227 `[V]` | **yes, easy** | "we construct a refusal part in our dataset by mismatching existing instructions with unrelated screenshots, yield over 2.6 million examples"; data table row "JEDI Refusal ... 165,235 / 2,666,124 / 2,666,124 / Random:5%" (last column read as per-epoch sampling `[L]`) | refusal response (exact string not given in the paper) | OSWorld-G refusal subset (54 items): JEDI-3B 7.4, JEDI-7B 7.4 | n/a | "although we included refusal data during training to encourage the model to reject instructions referring to elements not present on the screen, the model rarely produces refusal responses"; "Refusal modeling in GUI grounding remains a significant challenge, as models show limited improvement due to the inherent limitations in pretraining and the hallucination phenomenon in VLMs" |
| OS-Atlas | 2410.23218 `[V-html]` | no | — | click | none | n/a | none (scores 7.4 on OSWorld-G refusal untrained = Jedi trained) |
| Ferret (2023) | 2310.07704 `[V]` | **yes, 95K, incl. hard** | §4.3 "Spatial Negative Mining": (i) image-conditioned: Object365 "randomly select the object class from the vocabulary that is not shown in the given image"; (ii) semantics-conditioned: Flickr30k, GPT-4 picks "entities that are most analogous to the original class, attribute, or quantity, e.g., 'man' vs. 'woman', 'blue' vs. 'yellow', 'two' vs. 'three'"; "equilibrium between positive and negative samples" | **text with counter-box**: "Is there a cat in the image? No, but there is a dog [box0] in the image." | POPE only (Ferret-13B adversarial acc 82.36) — yes/no, not REC | n/a | none on grounding positives |
| Florence-2 | 2311.06242 `[V-html]` | not stated | FLD-5B filtering removes low-confidence boxes | not stated | none | n/a | none |
| PaliGemma / PaliGemma 2 | 2407.07726 `[V-html]`, 2412.03555 `[V]` | detection transfer only: pix2seq "noise boxes ... a dedicated `<noise>` token in place of the class name ... it provides a mechanism for the model to represent the confidence that a prediction represents a real object, in form of the probability assigned to the `<noise>` token. During inference, the `<noise>` and `<EOS>` tokens are excluded from sampling." | — | REC: `<locDDDD>` tokens; absent case undocumented (empty suffix `[L]`) | none | n/a | none |
| Rex-Omni | 2510.12798 `[V-html]` | structural | "If a particular phrase refers to an object that is not present in the image, the corresponding COORDS field is replaced with None." | **`None` in the COORDS field** | HumanRef "We use the first five subsets" — **rejection subset excluded** | n/a | GRPO removes duplicate boxes (SFT-only dup-removal +1.23 F1 vs +0.08 after GRPO) |
| Rex-Thinker | 2506.04034 `[V-html]` | yes (HumanRef-CoT incl. rejection subset) | GPT-4o CoT kept when final answer matches GT | no box selected from box hints | HumanRef Rejection: Plain 53.5 → CoT 67.3 → GRPO 68.2; Qwen2.5-VL-7B 7.1; RexSeek-7B 54.1 | CoT vs plain: +13.8 rejection | none on positives |

## 2. Rejection numbers per model across the benchmarks (third-party + lab)

Scoring rules (what counts as "none"), all `[V]` from the benchmark PDFs:
- **GroundingME** (2512.17495): unified prompt for every entry, greedy, thinking off unless in Table 6: "...Provide at most one bounding box. If a matching object is found, provide its bounding box as a JSON in the format {"bbox_2d": [x1, y1, x2, y2]}. If no matching object is found, output {"bbox_2d": null}." Rejection negatives: human descriptions of a present object with one clause false. Official scorer counts unparseable output as correct `[P: charter]`.
- **RefBench-PRO** (2512.06276): "a prediction is correct if the model outputs no bounding box or explicitly states that the target is absent." Negatives by a "Minimal, Plausible Mutation" prompt (one attribute / identity / location change of a real object).
- **PR-Bench** (2607.24407, Motto paper): same mutation prompt text; metric N-Acc (no-box output `[L: gRefCOCO convention]`).
- **OpenRef** (2605.25706): negatives by perturbing colour / orientation / category noun / proper noun; N3R = mean over negatives of Π_k(1−c_k), c_k = mean coordinate-token probability of each hallucinated box, so zero boxes = 1.0 and a model that emits no boxes on anything scores high (Mistral-3: F1 5.9, N3R 70.9).
- **OSWorld-G** (Jedi): 54 infeasible instructions; "the model is expected to refrain from taking action".
- **HumanRef** Rejection Score, **gRefCOCO** N-Acc: no-box output on category-absent / other-image expressions (easy).

| Model | GroundingME Rej (non-think / think) | RefBench-PRO Reject (grounding / yes-no classification) | PR-Bench N-Acc | OpenRef N3R | HumanRef Rej | gRefCOCO N-Acc | OSWorld-G refusal |
|---|---|---|---|---|---|---|---|
| Qwen2-VL-7B | – | 28.5 / 51.7 | – | 90.9 (F1 34.1) | – | – | – |
| Qwen2.5-VL-3B/7B/32B/72B | 7B 0.5, 32B 0.0, 72B 3.0 / – | 7B 3.1 / 55.1; 72B 23.6 | – | 7B 14.1, 32B 37.9 | 7B 7.1 | – | 0.0 / 0.0 / 0.0 |
| Qwen3-VL-2B/4B/8B/32B/A3B/A22B | **all 0.0** / 8B 4.5, 32B 9.5, A3B 5.5, A22B 5.5 | 8B 15.8 / 64.2 | 8B 16.2; 2B – | 2B 38.1, 4B 37.9, 8B 84.6 | 8B 47.9 | 2B 25.1, 8B 55.0 | – |
| Qwen3.5-9B / 27B | – | – | 9B 11.9; 27B – | – | 10.8 / 13.4 | – | – |
| InternVL3-8B / 78B | – | 21.3 / 46.8; 78B 24.8 (Acc_p 20.1 / 21.8) | – | – | – | – | – |
| InternVL3.5-8B / 38B / A28B | 8B 1.5, A28B 0.0 / – | 8B – / 49.2 | 38B 12.7 | 8B 51.0 | 8B 33.1 | – | – |
| GLM-4.1V-T / 4.5V / 4.6V | 4.5V 0.5 / 4.0 | 4.1V-Base – | – | 4.6V 78.7 | 4.1V-T 34.5 | 4.1V-T 29.0 | – |
| MiMo-VL-7B-RL | 0.0 / 5.0 | 0.1 | – | 20.7 | – | – | lab reports no_refusal subset only |
| Seed1.5-VL / Seed-1.6-Vision | 1.6: 1.0 / 1.5 | – | – | – | – | – | 1.5-VL 18.5 |
| Kimi-VL | – | – | – | – | – | – | (planner only) |
| Gemini-2.5-Pro / Flash | 7.0 / 0.0 (Pro total 20.7) | Pro – (Acc_p 9.6) | – | – | – | – | Pro **38.9** |
| Gemini 3.x | **no number on any rejection benchmark** | | | | | | |
| GPT-4o / GPT-5 / Operator | – | – (GPT-5 Acc_p 26.1) | – | – | – | – | Operator 0.0 |
| Claude-Sonnet-4.5 + PyVision | 9.5 (total 12.4; App 10 / Cmp 7.8 / Txt 14 / Sta 6) | – | – | – | – | – | – |
| Llama-4-Maverick / Scout; Llama-Nemotron-8B | 6.0 / 2.5; 5.5 (totals 13.0 / 8.9 / 10.4) | – | – | – | – | – | – |
| Gemma-3-27B; Mistral-3.2-24B; Phi-4-MM; MiniCPM-V-4.5; Keye-VL-1.5; LLaVA-OV-1.5 | 0.0 each | – | – | Mistral-3 70.9, Keye 70.3, MiniCPM 73.6, LLaVA-OV-1.5 75.1 | – | – | – |
| UI-TARS-7B / 72B; UGround; Aguvis; OS-Atlas-7B | – | – | – | – | – | – | 0.0 / 0.0; 0.0; 0.0; 7.4 |
| Jedi-3B / 7B | – | – | – | – | – | – | 7.4 / 7.4 |
| Rex-Omni-3B | – | – | – | – | excluded by authors | 33.6 | – |
| Rex-Thinker-7B | – | – (Acc_p 63.6) | – | – | 68.2 | – | – |
| VLM-FO1-3B; VLM-R1-3B; Migician; DeepEyes; ROD-MLLM | – | VLM-R1 –, Migician – | FO1 11.4 | – | 46.4; 27.1; 26.0; 22.8; – | 52.5; 30.6; 14.4; 38.4; 57.8 | – |
| Motto-2B / 4B / Qwen2.5-VL-3B (PR-Bench authors, 2026) | – | – | **46.9 / 47.5 / 47.4** | – | 53.4 | 58.0 | – |
| Ref-R1 (RefBench-PRO authors, Qwen2.5-VL-7B) | – | 3.1 → **58.2** | – | – | – | – | – |
| Grounding DINO L; GLEE | – | 0.1; 7.1 | – | GDINO 47.1 | – | – | – |
| 2026 open models with no rejection number anywhere | LocateAnything-3B (2026), Youtu-VL-4B (2026), Qwen3.5-27B on PR-Bench, GLM-5V-Turbo, Seed1.8/2.0, Gemma 4, Molmo2, MolmoPoint | | | | | | |

Leaderboard state `[V-html: groundingme.github.io]`: 28 open-source rows = the paper's 21 open models + 7 thinking variants, 4 commercial rows; **no 2026 model has been added**. All entries use the unified prompt above.

## 3. Pattern: did any lab move hard-negative rejection across generations?

**No.** Every generation-over-generation movement is on easy negatives, and the one lab whose report mentions negatives (Qwen2.5-VL) is the worst rejector in its own family.

Qwen line (7-9B tier), easy → hard:

| Generation | HumanRef Rej (easy) | OpenRef N3R (easy-medium) | gRefCOCO N-Acc (easy) | RefBench-PRO Reject (hard) | PR-Bench N-Acc (hard) | GroundingME Rej (hard, non-think) |
|---|---|---|---|---|---|---|
| Qwen2-VL-7B (2024) | – | 90.9 (but F1 34.1: rejects by not boxing) | – | 28.5 | – | – |
| Qwen2.5-VL-7B (2025-02, the one with "non-existent object categories" in the mix) | 7.1 | 14.1 | – | 3.1 | – | 0.5 |
| Qwen3-VL-8B (2025-10) | 47.9 | 84.6 | 55.0 | 15.8 | 16.2 | 0.0 |
| Qwen3.5-9B / 27B (2026-02) | 10.8 / 13.4 | – | – | – | 11.9 / – | – |

- 2.5 → 3: easy metrics jump (+40.8 HumanRef, +70.5 N3R); hard metrics stay near floor (+12.7 RefBench-PRO, GroundingME 0.5 → 0.0). Qwen3-VL's report adds no negatives, so the easy-metric jump most likely comes from the general instruction mix (Molmo-style PixMo pointing data, which contains "not present" answers, entered Qwen3-VL's point data) `[L]`.
- 3 → 3.5: the easy metric **regresses** (47.9 → 10.8/13.4 HumanRef; PR-Bench 16.2 → 11.9) while positives improve (Ref-L4 88.5 → 89.0/90.2, PR-Bench mAcc 59.4 → 63.7/65.3). Rejection is not in the objective of the native-multimodal generation `[V: 2607.24407 Table 1, 11]`.
- Scale within a generation: GroundingME 2B→235B all 0.0; OpenRef N3R 2B 38.1 → 8B 84.6 (easy negatives do scale) `[V]`.

Other lines: InternVL3-8B → 3.5-8B: RefBench-PRO grounding-format 21.3 → not reported (yes/no 46.8 → 49.2, chance ≈ 50); GroundingME 1.5; InternVL3's 21.3 sits on Acc_p 20.1, i.e. it is format failure counted as rejection `[L]`. GLM-4.5V GroundingME 0.5 / 4.0 think; GLM-4.6V OpenRef 78.7. Seed1.5-VL OSWorld-G 18.5 → Seed-1.6-Vision GroundingME 1.0 / 1.5 think. MiMo-VL 0.0 / 5.0 think. Thinking buys 4.0-9.5 GroundingME rejection at a Discriminative cost in every model (8B 61.3 → 52.5, 32B 75.0 → 65.7, A22B 69.6 → 65.2, Seed 59.8 → 59.3) `[V: Tables 3, 6]`.

Gemini-2.5-Pro is the only frontier model that refuses on both GUI (OSWorld-G 38.9; next Seed1.5-VL 18.5; every GUI-specialised model and Operator 0.0; "in all models except Gemini-2.5-Pro ... refusal predictions are consistently absent" `[V: 2505.13227]`) and photos (GroundingME 7.0, best non-thinking), but with weak positives (GroundingME total 20.7, RefBench-PRO Acc_p 9.6) — so its rejection cannot be separated from non-compliance. No Gemini 3.x, GPT-5.x, Claude 4.6+/5 or Seed 2.0 number exists on any rejection benchmark (the charter's gap stands).

The only entities that moved hard-negative rejection are benchmark authors fine-tuning a Qwen: GroundingME 2:1 SFT (0 → 27.9), Ref-R1 (3.1 → 58.2), Motto (Qwen3-VL-2B → 46.9). None is a lab release.

## 4. Output format: what "none" each model was trained to say, and what scorers accept

| Family | Trained "none" form | Documented? | Accepted by GroundingME (`{"bbox_2d": null}` literal or parse failure) | Accepted by RefBench-PRO (no box or explicit absence text) | Accepted by OpenRef N3R (zero boxes) |
|---|---|---|---|---|---|
| Qwen2.5/3-VL, 3.5, 3.6 | undocumented; native positive format is JSON with key `bbox_2d` — the **only** family whose positive format matches the GroundingME schema | no | matches schema, still 0.0 → the model never emits `null` `[P: p(null) AUROC 0.298]` | yes | yes |
| Gemini / Gemma 4 | JSON list `box_2d`; empty list `[S]` | no | an empty list is unparseable under the schema → counted correct (Gemini-2.5-Pro 7.0, Flash 0.0 — Flash therefore produced boxes on all 201) | yes | yes |
| GLM-4.x | `<|begin_of_box|>…<|end_of_box|>`, one span required | none form not documented | 0.5 | — | 78.7 (GLM-4.6V) |
| Seed | `<bbox></bbox>` (RL prompt) | no | 1.0 | — | — |
| Molmo / Molmo2 | text "This isn't in the image." | yes (Molmo) | text → parse failure → counted correct (not evaluated) | yes (explicit absence) | yes |
| MolmoPoint | end token before any `<PATCH>` | yes | n/a | n/a | n/a |
| Rex-Omni | `COORDS: None` per phrase | yes | not `bbox_2d` schema → parse failure | yes | yes |
| Ferret | "No, but there is a dog [box0] in the image." — **a box is still emitted** | yes | would be scored as a wrong box, not a rejection | ambiguous (states absence *and* boxes) | scored as a hallucinated box |
| SAM 3 | presence score → empty mask set; threshold on `p(NP present)` | yes | n/a | n/a | n/a |
| PaliGemma 2 (detection transfer only) | `<noise>` class-token likelihood = objectness; no explicit none | yes | n/a | n/a | n/a |
| Jedi / GUI models | refusal text (string not given) | partially | n/a | n/a | n/a |
| Claude | prose or structured-output JSON; no none convention; PyVision run output whatever → 9.5 | no | parse-failure credit likely `[L]` | — | — |

Consequences `[V]`:
1. Every non-zero non-thinking GroundingME rejection belongs to a weak grounder (Llama-4-Maverick 6.0 at total 13.0; Nemotron-8B 5.5 at 10.4; Gemini-2.5-Pro 7.0 at 20.7; Claude+PyVision 9.5 at 12.4; Qwen2.5-VL-72B 3.0; InternVL3.5-8B 1.5 at total 3.3). Every strong grounder (Qwen3-VL all sizes, Seed-1.6, GLM-4.5V, MiMo) is 0.0-1.0. This is the parse-failure-credit signature; none of these numbers is interpretable without the parse-failure rate (charter do-not #9 stands).
2. The prompt is the same for every leaderboard entry, so training-format mismatch is not what separates models; the family with zero mismatch (Qwen, `bbox_2d`) is the one that never rejects.
3. **RefBench-PRO Table 5** is the one published format experiment: the same models score 3.1-28.5 when told "output a box if the object exists" and 46.8-64.2 when asked "yes/no, is it present?" (Qwen3-VL-8B 15.8 → 64.2; InternVL3-8B 21.3 → 46.8; Qwen2.5-VL-7B 3.1 → 55.1). Authors: "performance remains close to random chance (approximately 50%) under the binary classification setting", i.e. the yes/no route mostly buys the prior, and only Qwen3-VL-8B (64.2) clears it. Reframing the decision as a classification changes the number by 40-50 points on the same images — the format is part of the measured deficiency, not just noise.
4. OpenRef's N3R rewards silence: Mistral-3 (F1 5.9) gets 70.9; Qwen2-VL (F1 34.1) gets 90.9; the same metric ranks Qwen3-VL-8B (F1 63.7) at 84.6. Cross-benchmark rejection rankings therefore disagree by construction (charter's −0.304 correlation).
5. Labs that do define a none form put it **outside the coordinate stream** (Molmo text, Rex-Omni `None`, SAM 3 presence head, MolmoPoint end-class, PaliGemma 2 `<noise>` likelihood, Ferret yes/no + counter-box). No lab trains the "same JSON key, value null" form that GroundingME scores.

## 5. What the labs' own ablations say about negatives hurting positives

- **Frontier VLM reports: zero ablations.** None of Qwen2.5-VL, Qwen3-VL, Qwen3.5/3.6 cards, InternVL3/3.5, Seed1.5-VL/1.8/2.0, Kimi-VL, GLM-4.1V/4.5V/5V-Turbo, MiMo-VL, UI-TARS/-2, Gemma 4, Gemini 3.1 card, Claude docs reports a negatives ablation, an over-refusal metric, or a rejection-vs-positive trade-off `[V for the PDFs, V-html for the rest]`.
- **SAM 3** (detector with decoupled presence head): negatives are free for localisation and large for recognition — pmF1 62.4 → 62.8 while IL_MCC 0.44 → 0.68 going 0 → 30 hard negatives/image; the presence head itself costs pmF1 65.4 → 63.4 for cgF1 +1.5 / IL_MCC +0.05 `[V: Table 9]`. Negatives are *adversarial to the previous model generation*, i.e. mined hard negatives, and AI verifiers check them "with close to no human-annotator involvement".
- **Jedi** (GUI): 2.67M easy negatives (instruction ↔ unrelated screenshot), sampled ~5%/epoch, → refusal 7.4 = untrained OS-Atlas-7B; no positive cost reported (JEDI-7B is the best grounder in its table at 54.1 overall) `[V]`. Easy negatives at scale neither help nor hurt.
- **GroundingME** authors (Qwen3-VL-8B, 30k RefCOCOg + description-modified negatives, 3 epochs): 1:8 → 2:1 gives Rej 3.5 / 8.0 / 11.4 / 16.4 / 27.9 with Dis 57.4 / 49.5 / 54.4 / 46.6 / 40.2 (base 61.3), Lim 24.3 / 19.3 / 23.0 / 26.3 / 17.0 (base 36.0), total 27.0 / 25.0 / 28.7 / 28.5 / 26.0 (base 31.0); "GroundingME w/o Rejection ... from 38.8% to a max of 33.0%"; in-domain RefCOCOg 88.2 → 83.1-90.4 `[V: Table 7, §5.3]`. Even the 1:8 mix loses 4 points of Dis before rejection moves. Their statement: "the rejection capability gained from simple data mixture does not generalize for free".
- **Motto** (PR-Bench authors): L_cls with top-K hard-negative selection, K=20 optimal; "an excessively large K leads to performance drops as easy negatives dominate the training process" — an explicit easy-vs-hard negative ablation in a grounding MLLM, with positives (Ref-L4 92.5, PR-Bench mAcc 71.7) and N-Acc 46.9 both above the Qwen3-VL-2B base `[V: D.2]`.
- **Ref-R1** (RefBench-PRO authors, Qwen2.5-VL-7B, CoT SFT + DyIoU-GRPO): Reject 3.1 → 58.2 `[V: Table 6/8 via summary]`; positive side not extracted here.
- **Rex-Thinker**: CoT +13.8 HumanRef rejection (53.5 → 67.3), GRPO +0.9 more; positives also rise `[V-html]`.
- **MolmoPoint**: removing the no-more-points class "more than doubles the amount of overcounting" — the structural negative improves positives `[V-html]`.
- **OpenRef MCC** (training-free): F1 +2.3 / +8.0 / +7.2 / −0.4 / +2.7 and N3R +53.8 / **−5.7** / +18.2 / +21.7 / +8.8 for Qwen3-VL-2B / GLM-4.6V / InternVL3.5 / LLaVA-OV-1.5 / Keye-VL-1.5 — the strongest rejector loses rejection `[V: Tables 4-5]`.
- **Ferret**: balanced pos/neg 95K; POPE adversarial 82.4; no REC-positive cost reported.

Net: the only evidence that negatives do not hurt positives comes from architectures that route presence through a separate head or token (SAM 3, MolmoPoint, Motto's L_cls); every result that puts "none" into the same autoregressive coordinate stream by SFT pays 4-21 points on positives (GroundingME) or does nothing (Jedi).

## 6. Open disputes and what I could not verify

- GPT-5.x system cards: openai.com blocks WebFetch; third-party coverage (search results, Wikipedia entries for 5.1/5.2/5.4/5.5) shows no grounding section; treat "GPT-5.x: no grounding negatives documented" as `[L]`.
- Gemini: only the 3.1 Pro card was read (no grounding content); 3 Pro / 3.8 cards assumed identical `[L]`. The API docs page now uses `gemini-3.8-flash`.
- Qwen3.5 launch blog is JS-rendered (fetch returned only "Qwen"); Qwen3.5 facts come from the HF 2B card and PR-Bench's third-party rows.
- Seed2.0 HTML 404; PDF grep shows only CountBench / FSC-147 / PointBench / spatial suites, no negatives.
- Molmo2's inheritance of PixMo "not present" answers is inferred from its data tables, not stated.
- Jedi's exact refusal output string and the semantics of the "Random:5%" column are not in the text.
- PR-Bench's N-Acc acceptance rule (no-box vs explicit text) is not printed; assumed gRefCOCO convention.
- Whether Gemini-2.5-Pro's, Llama-4's and Claude+PyVision's GroundingME rejections are real or parse failures cannot be settled from the paper; needs the raw outputs (lmms-eval / VLMEvalKit reproductions exist per SATURATION-AUDIT).
- RefBench-PRO thinking-mode table (Table 4) covers Relation/Commonsense only; no thinking number for Reject exists anywhere except GroundingME Table 6.

## Sources

Primary PDFs (in scratchpad): 2502.13923, 2511.21631, 2505.07062, 2508.18265, 2511.16719, 2505.13227, 2310.07704, 2412.03555, 2512.17495, 2512.06276, 2605.25706, 2607.24407, 2601.10611, 2607.00248. HTML/summaries: 2504.10479, 2504.07491, 2507.01006, 2604.26752, 2506.03569, 2510.12798, 2506.04034, 2409.17146, 2603.28069, 2607.02770, 2501.12326, 2509.02544, 2410.23218, 2311.06242, 2407.07726, 2603.20633. Docs: ai.google.dev/gemini-api/docs/image-understanding; deepmind.google/models/model-cards/gemini-3-1-pro; developers.openai.com/api/docs/guides/images-vision; platform.claude.com/docs/en/build-with-claude/vision, /vision-coordinates, /agents-and-tools/tool-use/computer-use-tool; huggingface.co/Qwen/Qwen3.5-2B, /Qwen/Qwen3.6-27B, /blog/gemma4; groundingme.github.io.
