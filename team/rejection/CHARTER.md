# Rejection deep-dive — team charter (2026-09-16)

Lead: `main`. Workspace: `D:\Dev\ArcNova\auto-research\VLM-grounding\team\rejection\`.
Today is 2026-09-16. Your training data may end before mid-2026: **use WebSearch / WebFetch for anything from 2025-2026** and treat your memory of that period as `[LIKELY]` at best.

## The question (from the user)

The project has established that the model's largest untouched headroom on GroundingME is the **Rejection** dimension: the referring expression describes an object that is **not in the image** (or is in the image but with one clause false), and the correct answer is to abstain / output no box. Three deliverables:

1. **Survey**: what are the influential recent works (2024-2026, with pre-2024 ancestors where they matter) on rejection / abstention / no-target handling in grounding and its neighbouring fields?
2. **Deficiency analysis**: what exactly is wrong with current models on this capability — with evidence, not adjectives?
3. **Brainstorm**: how could the capability be improved — ranked by information cost (which cheap experiment settles the most), with dataset / model / GPU RAM / CPU RAM / GPU-hours for each experiment, split into local (RTX 5090 32 GB VRAM, 32 GB host RAM, 23 GB usable inside WSL) vs cloud (RunPod).

## What is already known (do not re-derive; cite these files)

Read these first: `notes/CAPABILITY-MECHANISMS-2026-09-06.md` (§2, §4 rule 3, §5, §7 step 2), `notes/SATURATION-AUDIT-2026-09-06.md` (GroundingME section, the "levers" table, the do-not list), `notes/BENCHMARK-LANDSCAPE.md`, `notes/ABSTAIN-SIGNAL.md` + `notes/CORRECTION-2026-09-05.md` + `notes/GME-PRECHECK.md` (our own measurements), `notes/RL-DESIGN-CANDIDATE-VERIFICATION.md` (the current training design; brainstormers must go beyond or critique it, not restate it).

Facts established there (P = measured in this project):

- GroundingME (2512.17495) Rejection: 201 items, real unedited SA-1B photos, human 39-word descriptions of an absent object, four sub-axes (appearance / component / text / state). Scoring is **format scoring**: official `evaluate.py` accepts only the literal `{"bbox_2d": null}` and **counts unparseable output as correct rejection**. 20 of 25 models score exactly 0.0 in non-thinking mode; best 9.5; thinking mode shows some rejection. Scale does not help (Qwen3-VL 2B to 235B all 0.0 non-thinking).
- Qwen3-VL-8B-Instruct here: 0/201 (P). p(null) at the `{"bbox_2d":` decision token has **AUROC 0.298** for absent-vs-present across the benchmark: it tracks difficulty, not absence. 41/804 positives abstained (5.1%), 40 of them in Limited (tiny objects).
- On our own inpainting removals with **class-label** expressions ("the Spoon"), the same model shows p(null) separation of 16 orders of magnitude, AUROC 0.964, 50% abstention, 0/316 false abstentions on control edits. On GroundingME images, **short checkable-clause expressions (≤20 words)** give 43% abstention / AUROC 0.972 while **long graded-appearance paragraphs (>35 words)** give 0% / 0.685. The variable is the expression, not the scene.
- Easy negatives are solved, hard ones are not: Qwen3-VL-4B gRefCOCO no-target N-acc 77.2 (negatives = category absent) vs GroundingME Rejection 0.0 (object present, one clause false). OpenRef N3R 84.6 for Qwen3-VL-8B. GroundingME Rejection and OpenRef N3R correlate **−0.304** across models: the two rejection metrics disagree.
- Training levers and their costs: GroundingME's own 2:1 negative SFT: Rejection 0 → 27.9 but Discriminative 61.3 → 40.2, Limited 36.0 → 17.0, RefCOCOg 88.2 → 83.1. RC-GRPO (2608.04698, forced None rollouts + over-refusal penalty + negative-advantage scaling, 2k samples, LoRA): gRefCOCO N-acc +65 on a 7B that never abstained, only +8.1 on Qwen3-VL-4B that already did; MMBench unchanged. Jedi (2505.13227) generated 2.67M synthetic refusal samples and scored 7.4 on OSWorld-G refusal, same as untrained OS-Atlas-7B. Training-free consistency check MCC (OpenRef, 2605.25706): N3R 38.1 → 91.9 on a 2B model, hurts the strongest model.
- Other rejection numbers: PR-Bench (2607.24407) reject subtask no model above 47.5; RefBench-PRO (2512.06276) Qwen3-VL-8B reject 15.8; HumanRef (2503.08507) rejection tops at 68; OSWorld-G 54 refusal items.
- Mechanism claims from the wider audit: grounding errors are committed in prefill; self-verification confidence correlates with correctness at r≈0.22 (2606.13156); calibrated abstention with AUROC > 0.9 is an open target.

## Roster and files

Phase 1 (parallel):
- `SurveyA` — REC / grounding-native rejection line → `survey-A-rec-rejection.md`
- `SurveyB` — general VLM/LLM abstention, unanswerable-question detection, calibration, abstention-aware RL → `survey-B-abstention-general.md`
- `SurveyC` — adjacent domains where absence is handled: detection/segmentation presence & objectness heads (SAM 3 presence token, DETR no-object, OVD negatives, GRES no-target), GUI refusal, mechanistic work on refusal/abstention directions and "knows-it-doesn't-know" probes → `survey-C-adjacent-domains.md`
- `SurveyD` — what frontier labs actually did: tech reports / model cards of Qwen3-VL, Qwen3.5/3.6, InternVL3.5, Gemini 3.x, GPT-5.x, Claude, Molmo2, SAM 3, Gemma 4, Seed-VL, Kimi-VL, GLM-V, UI-TARS, Jedi/OS-Atlas — do they train negatives for grounding, how, and what do they report → `survey-D-frontier-practice.md`
- `Diagnostician` — evidence-based deficiency taxonomy from our own notes and tables (CPU-only analysis) → `diagnosis-local.md`

Phase 2: `Verifier` → `verifier.md`; `Skeptic` → `skeptic-survey.md`; three brainstormers → `ideas-1-objective.md`, `ideas-2-inference-architecture.md`, `ideas-3-data-evaluation.md`.
Phase 3: `Skeptic` → `skeptic-ideas.md`. Lead synthesis → `notes/REJECTION-DEEP-DIVE-2026-09-16.md`.

## Rules

- **Evidence labels on every non-trivial claim**: `[VERIFIED: arXiv id or URL]` (you read it), `[LIKELY: reason]`, `[SPECULATION]`, `[P: file]` for project measurements. A number without a source is `[LIKELY]` at best.
- Every paper entry: title, arXiv id, venue/date, **what it measures rejection with** (metric name, what counts as a negative, how negatives were built, whether parse failures count), headline numbers with the base model, and the cost on positives if reported. Distinguish easy negatives (category absent / expression from another image) from hard negatives (object present, one attribute false).
- Web tooling: load with `ToolSearch` query `select:WebSearch,WebFetch` first. arXiv HTML (`arxiv.org/html/<id>`) or abs pages are the cheapest primary source; `export.arxiv.org/api/query?search_query=...` lists recent papers. HTML fetch truncates large tables — if a table matters, say so and mark the number `[LIKELY]` rather than guessing. openai.com blocks WebFetch. Each agent has a finite web-search quota (about 200 calls): prefer fetching known ids over repeated searches.
- Write your file with the Write tool (heredocs with apostrophes fail in this shell). Append, do not rewrite other people's files. Keep the file dense: tables over prose, no filler.
- Do **not** run GPU jobs. CPU analysis of existing tables is fine (python on Windows works for `tables/*.jsonl`; the GroundingME parquet with expressions and `subtask_l2` lives inside WSL: `wsl.exe -d Ubuntu -e bash -lc 'cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding && ~/vlmg-env/bin/python -c "..."'`, data at `~/vlmg-data/prepared/groundingme/`).
- Final report to the lead: at most 25 lines — file path, the 5 to 8 findings that change decisions, open disputes, what you could not verify.

## Corrections after Phase 2 (Verifier V-R35/V-R36, Skeptic S1-S3)

- "gRefCOCO no-target = category absent" is wrong: rule 1 (present category, false attribute) is the default, rule 2 (other-image expression) the fallback; the ratio is unpublished. The 77.2 / 85.3 N-acc there is not a clean easy-negative number.
- "Short checkable-clause expressions (<=20 words) give 43% / 0.972" should read "Discriminative/Text removals give 43% / 0.972"; the 28 items are exactly the Text items, and length is untested below 23 words for non-Text.
- GroundingME Rejection expressions have median 54 words (quantiles 38/46/54/64/74), not 39.
- The 201 Rejection expressions are text-separable from positives at NB AUROC 0.920 (0.814 vs Discriminative); every AUROC on GME needs a gray-image control and a text-floor row.
- Synthesis: `notes/REJECTION-DEEP-DIVE-2026-09-16.md`.
