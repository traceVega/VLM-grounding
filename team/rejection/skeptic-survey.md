# Skeptic — Phase 2 review of the five Phase 1 files

Author: `Skeptic`, 2026-09-16. Inputs: `diagnosis-local.md`, `survey-A..D`, `notes/CORRECTION-2026-09-05.md`, `notes/CAPABILITY-MECHANISMS-2026-09-06.md` §2/§4/§5/§7, `notes/RL-DESIGN-CANDIDATE-VERIFICATION.md` §0–§3, §5–§7, `notes/ABSTAIN-SIGNAL.md`, `notes/GME-PRECHECK.md`, `notes/SATURATION-AUDIT` (GME section), `notes/BENCHMARK-LANDSCAPE.md`. Labels as in the charter; `[P: skeptic]` = my own CPU re-analysis this session (script: scratchpad `skeptic_checks.py`; parquet + `dimensions.csv` inside WSL, joined to `tables/gme_remove.jsonl` / `gme_original.jsonl`). Web: 2 searches, 6 fetches. No GPU.

## 0. Three local numbers this review adds

| # | measurement | value | why it matters |
|---|---|---|---|
| S1 | Discriminative removals with ≤20 words, by `subtask_l2` | **28/28 Text**, 0 Appearance / Component / State | Confirms the Diagnostician: the CORRECTION "≤20 words" cell *is* the Text cell (43% declines, +12.65 orders) |
| S2 | Word-count range of the 87 non-Text Discriminative removals | min 23, P10 34, Q1 40, median 48, max 86 | The "no length effect within non-Text" (ρ=0.056, p=0.61) is measured on a range that **never contains the short regime**. It cannot rule length out; it only says 34–65 words behave alike |
| S3 | **Text-only separability of GroundingME Rejection** (5-fold CV, naive Bayes on uni+bigrams of the expression, no image) | AUROC Rejection vs all 804 positives **0.920**; vs Discriminative only **0.814**; vs Limited only **1.000**. Single features: word count 0.751, hedge words 0.685, comma count 0.730, quoted string 0.650 | The 201 negatives are topically skewed (top Rejection tokens: *young, shirt, sleeved, jacket, modern, grille*; top Discriminative tokens: *with_number, obscured, base, rough, handle*). Any probe, any SFT/RL run, any "AUROC absent vs present" on GME must beat this floor or be run blind. The Diagnostician's Probe 8 threshold (0.75) sits **below** it |

Also read off the join: Rejection/Text has 49/50 quoted strings and min 12 words (the only sub-axis with items ≤20 words, n=3); Appearance has hedge words in 72%; Discriminative/Text positives are 14 words median vs 46–50 for the other sub-axes. The Diagnostician's tables reproduce.

---

## 1. Dispute rulings

### 1(a) Sub-axis vs expression length vs clause count

**Ruling.** The Diagnostician wins the narrow point: CORRECTION §1 and RL-DESIGN §1.2/§1.4 attribute the 43% / AUROC 0.972 cell to "≤20 words", and that cell is 100% Text (S1). But the replacement claim — "the sub-axis is the variable, length is a minor modulator" — over-reaches, for three reasons:

1. Range restriction (S2): no non-Text removal is shorter than 23 words, so the ρ=0.06 null is uninformative about short non-Text expressions.
2. Inside Text, length does move the operating point: the 9 long Text removals shift +8.0 orders median (the same text-bearing object is gone) but decline **0%**, versus 43% for the 28 short ones. Same sub-axis, same edit, different decision. That is a dilution effect, not a sub-axis effect.
3. Rejection is strictly harder than any removal: a removal makes *every* object-specific clause false at once (the object is gone), Rejection makes exactly one clause false among ~5 true ones. Even the +8-order long-Text removal shift does not appear on Rejection/Text (median −17.1). So neither "Text" nor "short" is sufficient; both matter through one quantity.

**Both can be true under one variable:** the fraction of clauses that the best *remaining* same-category candidate satisfies (the Diagnostician's own D2, partial-match acceptance). Text strings are binary, OCR-checkable tokens: a text clause is satisfied 0 or 1; appearance clauses are graded and mostly satisfied by any same-category candidate. Short-Text removal = 0/1 satisfied → null; long removal ≈ (k−2)/k → box; Rejection = (k−1)/k → box. FINER's "clause count is the dial" (Survey A) is consistent with this, with a caveat below (§3, claim 11): paired accuracy decays mechanically as q^(k+1) even at constant per-clause skill, so FINER alone does not show that *detecting the false clause* gets harder with count.

**What would overturn it (the experiment to run first, before Probe 6).** A 2×2 *text-only* edit set on the 50 Discriminative/Text positives (a text string is present; same-category distractors exist by construction). No image edits. Variants, expression otherwise byte-identical: S-T = head noun + text clause only, string swapped to a plausible absent string; L-T = the full paragraph with only the string swapped; S-A = head noun + one appearance/state clause, flipped; L-A = full paragraph with one appearance clause flipped; plus S-true / L-true controls. Read p(null) at the decision token. ≈300 prompts, 0.15 GPU-h, 8B-Instruct, local.

| outcome | S-T | L-T | S-A | L-A | reading |
|---|---|---|---|---|---|
| Diagnostician (sub-axis) | high | high | low | low | text identity is verified when present; dilution irrelevant |
| dilution / partial match | high | low | high | low | the count of unsatisfied-vs-satisfied clauses is the variable; sub-axis only sets checkability |
| neither | low | low | low | low | text identity is not verified even in isolation; the 28-item removal signal was "text-bearing object gone" (category-like absence), and D2's mechanism stands in its strongest form |

**Implications for how negatives are built** (regardless of outcome): (i) span the dilution axis deliberately — one false clause among k true, k ∈ {0, 1, 2, 4}; GME lives at k≈4–5, and a set of only single-clause negatives teaches the regime the model already half-handles; (ii) do not let Text dominate the mix — the model already registers text-bearing-object absence, so Text negatives are the easiest hard negatives; (iii) the false clause must be verifiable by the base model in isolation at the eval resolution (D15/D16 below) or the label is unlearnable; (iv) RL-DESIGN's "prefer discrete checkable clauses" survives, but its justification is label verifiability, not "short expressions carry the signal"; long paragraphs with one false clause must be in the training distribution; (v) because the benchmark itself is textually separable at 0.92 (S3), the blind control is not optional for this data design.

### 1(b) RefBench-PRO Table 5: negative type vs output format

Numbers agree across B and D (Qwen3-VL-8B 15.8 grounding / 64.2 yes-no; Qwen2.5-VL-7B 3.1 / 55.1; InternVL3-8B 46.8 / 17.4 `[VERIFIED: 2512.06276 PDF, both surveys]`). Fetched the HTML: the binary setting asks the model to "respond with 'yes' or 'no' to indicate whether the target object is present in the image"; the "close to random chance (approximately 50%)" sentence is the authors'; **no positive-side yes/no accuracy is printed next to the reject numbers** `[VERIFIED: 2512.06276 HTML]`.

**Ruling.** Survey D's "the decision route moves the number by 40–50 points" is a scale artifact: grounding-reject is a no-box rate with a floor at 0 for a model that never emits a refusal (Qwen2.5-VL-7B's 3.1 is that floor, per Survey A §5 artifact 1); yes/no accuracy has a floor at the model's prior "no" rate (~50). Two metrics with different floors 50 points apart are not evidence of a 50-point format effect. Survey B's "the contrast holds across negative types, not formats" is closer, but its "chance" reading is only valid if the 64.2 is measured on a balanced set — and without the positive-side "no" rate, 64.2 could be AUROC 0.6 or 0.9. So Table 5 cannot settle the dispute; it is uninterpretable as a capability claim. Two further points: (i) for **our** model the emission-floor argument is false — Qwen3-VL-8B emitted literal `null` 41 times on GME positives `[P: diagnosis §4]`, so its 0/201 is a decision, and "format" can only mean that the JSON-null channel is trained on visibility rather than satisfaction (D5), which is Survey D's point in a defensible form; (ii) RefBench-PRO negatives are L2 LLM mutations; GME is L4; transfer of the yes/no result is unknown.

**What overturns it.** The Diagnostician's Probe 1 (existence yes/no on 201 Rejection + 204 Discriminative positives, p("no") AUROC, 0.3 GPU-h) *with the text-only floor of 0.81 (S3, vs Discriminative) as the bar and a gray-image run as control*: AUROC > 0.85 sighted and a drop ≥ 0.2 blind → Survey D is right in substance (a presence channel exists; decoupling the decision from the coordinate stream — MEGA-GUI's +30 `[VERIFIED: 2511.13087 via C]` — is the lever). AUROC ≤ 0.65 → Survey B is right (verification deficiency; format is only a floor effect on other models).

### 1(c) D1 "not an operating-point problem" vs "read-out, not representation"

**Ruling: same word, different layers, both can be true, and the published norm says they usually are.** D1 is about the LM-head logit at one token: on that scalar an operating point cannot work on GME (AUROC 0.298; length-matched upper bound 0.565 `[P: diagnosis §1]`). TRAPSBench (0.91), HALP (0.89), VA neurons and UniProbe are supervised probes on *mid-layer* residual / FFN / attention states, on easy or textual negatives, and UniProbe reports the signal "degrades toward output layers" while VA neurons stay high with p("No") low `[VERIFIED via Survey C §3.2]`. So D1's label is correct for p(null) and says nothing about mid-layer states; the Diagnostician's sentence "the decision-token *representation* carries no absence signal" should read "decision-token *logit*". Survey C's converse — "the 0.298 is a readout failure rather than proof of absence of the signal" — is a hypothesis, not a finding: every cited probe is on category-absent or textual negatives, and 2410.02707 shows probes do not transfer across task types.

**The separating experiment** is Probe 8 with three amendments the Diagnostician's spec lacks: (1) train on removals (class-label + the 28 short-Text GME removals) and test on GME Rejection vs *Discriminative* positives — not only 5-fold CV inside GME; (2) a blind control: the same probe on hidden states from a gray-image forward pass, plus the NB text floor (0.81); (3) report every layer, decision token and last image token. 0.5 GPU-h + CPU.

| outcome | meaning | what the brainstorm may use |
|---|---|---|
| ≈ chance at every layer, sighted and blind | representation deficiency: "unsatisfied" is never computed | objective/data ideas only; steering and probe-gating are dead |
| > 0.85 mid-layer, falls toward the last layers; blind ≤ 0.6 | read-out deficiency at the LM head | probe-gated null, gated steering (CR-VLM), a small head, or small-data RL that only has to connect signal to token |
| > 0.85 sighted **and** > 0.8 blind | the probe reads expression statistics (S3), not the image | redesign negatives before anything else; every "AUROC absent vs present" so far is suspect |
| high only in-distribution (GME CV), ≈ chance when trained on removals | dataset-specific signal (GME style or removal artifact) | ambiguous; add the 2×2 text-edit set of §1(a) as a third training source |

Probe 1 (yes/no token) is a third read-out. If Probe 1 > 0.85 while the decision-token probe is at chance at every layer, the computation is *format-conditioned* — it runs only when the question is asked as presence — which is Survey D's position in its strongest form and points at decoupled decision architectures.

---

## 2. Deficiency taxonomy audit

| # | label as given | supported? | cheaper falsifier / sharpening | survey contradiction / my note |
|---|---|---|---|---|
| D1 | Strong: not an operating-point deficiency on GME | Yes for the p(null) scalar; the wording generalises to "representation", which is untested (§1c) | Probe 1 (0.3 GPU-h) with the 0.81 text floor and a blind run; then Probe 8 amended | None direct; TRAPSBench/HALP are on easy negatives. The "operating point" language should be retired for GME: it invites threshold ideas that D5 says will cost Limited first |
| D2 | Strong pattern / suggestive mechanism (partial-match acceptance) | Pattern yes; mechanism contested | The §1(a) 2×2 text-edit set (0.15 GPU-h) before Probe 2 (0.8) and Probe 6 (1.0) | FINER `[VERIFIED via A]`: models verify a single fine-grained false clause at ~80% in MCQ form, so "clauses never enter the decision" is too strong; 2605.09090 `[VERIFIED: abs]` reports the same "approximation behaviour" in TransVG/SwimVG, so D2 is not MLLM-specific. **Confound not in the taxonomy**: at the P21 pixel cap the 50 Rejection/Text strings may be unreadable (D15); then "text-identity not verified" is visibility, not verification |
| D3 | Strong that p(null) is not a correctness score; "untested whether any internal candidate-level score exists" | First half yes; second half is contradicted | Compute MTLA (attention of coordinate tokens onto the emitted box, layers 8–21) on our 1,005 outputs; 0.5 GPU-h with an attention dump; bar to beat: wrong-object vs right 0.574 | Propose-and-Attend `[VERIFIED via C: 2607.05978]` already gets AUROC 0.89 for box correctness on Qwen3-VL-8B COCO — a candidate-level score *exists* in the model; absence is what is untested |
| D4 | Strong: format-scoring artifact | Yes | Probe 3, but extended to the 204 Discriminative positives: literal-null rate there is the over-refusal number nobody has | Add: thinking rows and the 2:1 SFT row share one signature — Rejection up, Discriminative down (8B 61.3→52.5; 61.3→40.2). That is a prior shift, which raises the rate at constant AUROC. Parse-failure is one artifact; prior shift is a second, independent one |
| D5 | Strong: over-abstention is visibility | Yes (ρ −0.694 within Limited; confident nulls) | The planned pixel-budget sweep read as abstention rate | Consequence the taxonomy understates: the null channel is *trained* (median p(null) 0.9994 on the 41) — most plausibly on tiny-object or PixMo-style "not present" pointing data `[L: Survey D §3]`. So the model has a null token whose meaning is "cannot see". Every threshold/steering method inherits that meaning first |
| D6 | Strong: metric inconsistency | Yes | n/a | Add S3: each benchmark's negatives also carry their own *text* signature; part of the −0.304 is surface-statistics disagreement, not just metric definition |
| D7 | Strong for the score, untested for the representation | Yes | Probe 7 | Thinking 32B 9.5 vs 8B 4.5 is inside CI ±4 on 201; not evidence of scale in thinking mode either |
| D8 | Strong: pass@k ≈ 0 for `null` | Yes, conditional on the harness prompt | Probe 5 (prompt variants) *before* concluding forced rollouts are mandatory: a "check each clause first" prompt changes the prefix distribution; Probe 4 must use free generation with a strict parser and a prose-refusal detector | Nothing in the surveys contradicts; RC-GRPO's forced-None (lowest-logprob rollout replaced) is the published instance. Note that free-generation prose refusals ("no such object") are invisible to the decision-token harness and would be scored *correct* by the official parser — Probe 4 must count them separately |

**Deficiencies the taxonomy missed**

| # | deficiency | evidence | cheapest probe |
|---|---|---|---|
| D9 | **Over-refusal on GME positives has never been measured by any published method.** Thinking rows, the 2:1 SFT rows and best-of-16 report dimension accuracies, not false-null rates; a Discriminative drop of 8.8 could be all false nulls or all wrong boxes | `[VERIFIED: 2512.17495 Tables 3, 6, 7 via A/D]` — no false-null column anywhere | Any run: literal-null count on the 804 by dimension, alongside every Rejection number |
| D10 | **Head-noun-present is assumed, not measured, on the 201.** If a fraction of Rejection images contain no instance of the head noun, those items are category-absent negatives the model still boxes — which would contradict the "category-presence detector" model of null (D2/D5) | Diagnostician §9 concedes it is benchmark construction, `[LIKELY]` | Cross-family judge (Molmo2 pointing on the head noun or Gemma4 yes/no) on 201 images, ~0.1 GPU-h; 50 human checks in the existing review UI |
| D11 | **Published thinking numbers are doubly suspect**: parse artifact (D4) and prior shift (Rejection up with Discriminative down in every model); plus Thinking checkpoint vs Instruct+CoT are confounded (`CAPABILITY §7`) | `[VERIFIED: Table 6 via D]`; CI ±4 | Probe 3 with strict parser + false-null on 204 Discriminative; compute-matched Instruct N-sample oracle |
| D12 | **No human ceiling on the 201.** The only human number is a 100-item binary probe (51 rejection) at 91% → ~9% of negatives may be wrong or unverifiable at resolution; a target of "50" should be read against ~90, and the ±6.2 CI compounds it | `[VERIFIED: 2512.17495 via A]` | A per-item human pass on the 201 (~$100, calendar time not GPU); gives D10 for free |
| D13 | **Test-set narrowness**: 98.5% of negatives > 20 words, 99/201 person or vehicle, clothing vocabulary, one image source | `[P: diagnosis §2]`, S3 | Replicate any Rejection gain on PR-Bench Reject (1,000), RefBench-PRO Reject (1,000) and VenusBench-GD (900 edited GUI negatives) before calling it capability |
| D14 | **Text-shortcut separability of the benchmark**: NB 0.92 vs all positives, 0.81 vs Discriminative, 1.00 vs Limited | `[P: skeptic]` S3 | Mandatory blind (gray-image) control for every method and probe; text-only baseline reported next to every AUROC |
| D15 | **Resolution confound on Rejection/Text**: whether the quoted string is legible at the P21 cap is unknown | `[P: harness cap; untested]` | 50 OCR prompts ("what is written on the …?") at the cap and at 2× the cap, 0.05 GPU-h |
| D16 | D2 is two deficiencies: **D2a** the false clause is perceptible but ignored; **D2b** it is imperceptible at this resolution/scale. Different fixes (objective vs resolution/data) | Probe 2 exists in the plan but the taxonomy has no entry that separates them | Probe 2 per sub-axis, reported as per-clause yes/no accuracy on the *false* clause vs on true clauses of the same type |

---

## 3. Survey audit — the ten claims the brainstorm is most likely to over-rely on

| # | claim (where) | specific weakness |
|---|---|---|
| 1 | RC-GRPO "+65 N-acc, the only recipe with an over-refusal control" (A §4/§6, C, D §5, SATURATION §5.2) | Single v1 preprint, no venue; +65 is a format-floor effect on a 7B that never refused, **+8.1** on Qwen3-VL-4B; P-acc **−4.0** on the 7B; OOD only *down* the ladder (L2→L1); never on L4; "MMBench/POPE unchanged" is the least sensitive suite (`CAPABILITY §4`) |
| 2 | CFCamo "pair accuracy 80–91%, +3.7 S_α on Qwen3-VL-4B" (B §0.8, §3, §5) | **Abstract only**; camouflaged-object detection with the target inpainted out = category-absent removal (L3), not clause-false; no over-refusal number on hard target-present images; no replication |
| 3 | "The information is there; the output policy does not read it" — TRAPSBench 0.91 / HALP 0.89 / VA neurons (B §0.2, §2, C §3) | Probes trained and evaluated in-distribution; negatives easy or textual; **no text-only baseline in any of them**; TRAPSBench itself says visual gaps are 4× harder; none on grounding or on attribute-false referents; 2410.02707 says probes do not transfer across task types |
| 4 | Molmo-7B-D 68.6 zero-shot HumanRef rejection — "absence data in pretraining transfers" (C §1.2, D §1) | Rejection Score = any no-box output, and Molmo answers in text, so any non-pointing reply counts; DF1 72.6 (weak positives); person-only, LLM-rewritten L2 negatives; PixMo "not present" fraction unpublished |
| 5 | SAM 3: "30 hard negatives/image → IL_MCC 0.44→0.68, pmF1 flat; data ≫ architecture" (C §1.3, D §5) | Detector with a BCE presence head and per-candidate scores — no autoregressive coordinate stream; negatives are class-level confusers, not clause-level; in-distribution SA-Co; pmF1 is mask quality *given* presence and says nothing about false-absence on positives (IL_MCC bundles both directions) |
| 6 | MCC training-free "N3R 38→92 on a 2B" (A §4, D §5, SATURATION §5.1) | N3R is a soft product that rewards silence (Mistral-3 F1 5.9 → 70.9); hurts the strongest model (GLM-4.6V −5.7); single benchmark; per-type negatives unpublished; 2–3 passes |
| 7 | GroundingME thinking "4.5–9.5 real rejection on strong Qwen3-VL" (A §2, D §3) | Parse-failure-as-correct; CI ±4; coupled Discriminative drop (prior-shift signature, D11); Thinking checkpoint ≠ CoT prompt |
| 8 | Abstain-R1 "+8.4 accuracy", KoNA "0 cost", TruthRL "ternary beats binary" (B §3, §5) | Text or VQA, 2–3B bases with low starting accuracy; unanswerables built by the same construction at train and test; the reward-*shape* claim is confounded with negative *fraction* and with the false-abstain penalty; no hard visual negative anywhere in the row |
| 9 | VenusBench-GD "base models keep 64–78 refusal; positive-only specialists 0" (C §2.1–2.4) | Rows read from a truncated table (`[LIKELY]` by C itself); strict `[-1,-1]` format; no false-refusal rate on positives printed beside it; GUI |
| 10 | RefBench-PRO Table 5 "yes/no ≈ chance" (B) and "format = 40–50 points" (D) | No positive-side yes/no rate; two metrics with different floors; L2 negatives; InternVL3-8B reversed (46.8 / 17.4) shows the column flips sign by format in either direction — a format artifact, not a capability reading |

Also watch: **FINER "clause count is the dial"** (A) — paired accuracy decays as q^(k+1) mechanically; negative-only accuracy by level is not reported, so the claim that *detecting* the false clause degrades with count is unproven. **Jedi "2.67M samples → 7.4, so corpus size does not matter"** (D, RL-DESIGN §1.5, §8) — easy negatives, `wait` label, ~5%/epoch sampling; it says nothing about hard negatives at scale. **GroundingME's own SFT "0 → 27.9 OOD"** (everywhere) — the negative construction ("modified description") is unpublished; if it is L4-like, 27.9 is in-distribution.

---

## 4. Traps for the brainstorm — each with the one control that catches it

1. **A threshold or prior shift on the visibility scalar.** Control: report the AUROC of the decision score (absent vs Discriminative-present, and absent vs Limited-present) next to the rate; a threshold moves the rate at constant AUROC, capability moves the AUROC.
2. **Parse failures scored as rejection.** Control: strict scorer (OSWorld-G rule: parse failure = wrong) plus the literal-`null` rate; both printed.
3. **Over-refusal hidden by a positive-only or short-expression regression suite.** Control: false-null rate on the 804 by dimension *and* on long positives (Discriminative ≥36 words); a length-triggered refuser passes RefCOCOg untouched.
4. **Easy negatives dominating the mixture.** Control: difficulty gate (base model boxes ≥90% of training negatives) and a per-negative head-noun-present flag; report gains stratified by that flag.
5. **Text-shortcut negatives.** Control: gray-image run of the trained model on the eval negatives — blind rejection rate must be ≤ ⅓ of sighted; and an NB text-only AUROC on the training negatives < 0.65.
6. **Edited-image artifacts.** Control: C-condition (edit a non-referent) false-null = 0, and a never-edited test set (GME is unedited); N0 vs N0′ gap reported.
7. **In-distribution negatives.** Control: hold out by construction — train on clause flips, test on human-edited (GME), PR-Bench Reject and VenusBench-GD; budget the 3–10× decay of `CAPABILITY §4`.
8. **A separate head that is a category-presence detector.** Control: evaluate the head on (i) category-absent, (ii) category-present / clause-false with the head noun present, (iii) positives; it must separate (ii) from (iii), not only (i).
9. **Thinking-mode format effects.** Control: strict parser on thinking outputs; compute-matched Instruct N-sample oracle; Discriminative false-null rate.
10. **N3R soft pooling.** Control: hard N-acc (zero boxes) as the primary; N3R only as secondary.
11. **Selecting α / thresholds / prompts on the 201.** Control: freeze a 100/101 dev/test split of GME Rejection, or select on PR-Bench; write the split down before the first run.
12. **Judge-lineage leakage** (policy family verifies its own negatives). Control: cross-family listener and verifier (RL-DESIGN §9.0b); report their disagreement rate.
13. **Probe leakage through expression statistics.** Control: blind-forward-pass probe; the sighted probe must beat the text floor (0.81 / 0.92) by > 0.05 *and* drop ≥ 0.2 blind.
14. **A prompt that names the trick ("check each clause") counted as capability.** Control: same prompt on positives (false-null), and the protocol swap — the harness prompt byte-identical at eval.
15. **A rejection gain that is a prior shift coupled to a Discriminative drop.** Control: Discriminative ≥ −2.0 *and* trap-1 AUROC rise; the 2:1 SFT and every thinking row fail this.
16. **Image contamination** (SA-1B / HR-Bench in any training set). Control: pHash + DINOv2 exclusion of the 1,005 images and their near-duplicates.
17. **Rejection pooled into a weighted total with positives.** Control: never (`CAPABILITY §5`); report the two factors separately.

---

## 5. What the five surveys missed

| item | id / source | label | why it bears on hard-negative rejection |
|---|---|---|---|
| SugarCrepe (and ARO before it) | 2306.14610 (2210.01936) | `[VERIFIED: abs]` for the blind-model claim; numbers `[LIKELY]` | Atomic replace / swap / add hard negatives at caption level — exactly N0 — and the finding that on earlier benchmarks "blind models with no access to the image outperform state-of-the-art vision-language models": LLM-written negatives leak through fluency. Also the encoder-level attribute-binding hypothesis: the representation may be missing before the LLM (probe the patch tokens, not only the decision token) |
| CounterCurate | 2402.13254 | `[VERIFIED: abs]` | Counterfactual *image* generation (GLIGEN; GPT-4V + DALLE-3) for attribute / counting / position, fine-tuning CLIP and LLaVA; the direct precedent for RL-DESIGN's arm C; no positive cost reported — which is the number to demand |
| NaturalBench | 2410.14669, NeurIPS 2024 | `[VERIFIED: abs]` | Paired design: each question with two images that yield different answers, so blind solutions score at chance; frontier VLMs "lag 50–70% behind human performance". This is the evaluation template a rejection set needs: same expression on a present/absent image pair, same image with a true/false expression pair, scored as a group |
| Counterfactual perturbations in grounding | 2605.09090 (XAI4CV @ CVPR 2026) | `[VERIFIED: abs]` | TransVG / SwimVG show "approximation behaviour, producing a plausible bounding box that satisfies only part of the expression" on captions describing absent objects; embedding anisotropy does not explain it. D2 predates MLLMs and is architecture-independent |
| CORAL (medical VLM) | 2607.03647 | `[VERIFIED: abs]` | Blank / pixel-shuffled / image-absent / retrieved-hard-negative surrogates → Visual Reliance Score and Visual Hallucination Rate; a contrastive grounding objective that penalises image-invariant answers; Qwen2.5-VL-7B LoRA, VHR −8.0 pp. A ready-made blind-control metric (trap 5) and an objective term against text shortcuts |
| Contrastive decoding with a distorted image (VCD; M3ID, ICD family) | 2311.16922 | `[LIKELY: memory]` | Training-free logit contrast p(y \| x) − p(y \| x_distorted); the grounding analogue is p(null \| image) vs p(null \| candidate region masked) — a decode-time version of BCEA's ℓ(x) − ℓ(∅) that Survey B did cite |
| Pre-MLLM REC trained with negatives in the loss | Mao et al. 2016 MMI 1511.02283; Yu et al. 2017 speaker-listener-reinforcer 1612.09542 | `[LIKELY: memory]` | The field trained comprehension as *ranking against same-image distractor regions and distractor expressions* — the per-candidate verification the coordinate-emitting line dropped. The pre-2019 ancestor of ROD-MLLM / Rex-Thinker that Survey A's timeline (2022→) omits |
| Selective classification | Chow 1970; SelectiveNet 1901.09192 | `[LIKELY]` | Coverage-constrained abstain head and risk–coverage curves; rejection should be reported as a risk–coverage curve on GME, not a single rate (Survey B cites the Chow threshold only via 2609.17686) |
| Visual entailment | SNLI-VE 1901.06706 | `[LIKELY]` | Per-clause entail / contradict / neutral is the verification primitive; clause-level entailment data and heads exist and were never connected to REC rejection |
| OOD detection scores | MSP vs Mahalanobis / energy | `[LIKELY]` | p(null) is an output-space (MSP-like) score; feature-space scores usually beat it on the same model — the generic reason to expect Probe 8 to differ from D1 |
| Not found (2 searches, 6 fetches) | — | — | No 2026 paper trains on human-edited long-description negatives; no grounding-rejection paper reports a text-only baseline; RC-GRPO and OpenRef were the only REC hits for "abstain + clause verification" |

---

## 6. What I could not verify, and limits of my own numbers

- S3 is a 5-fold CV naive-Bayes bound on 1,005 items from one image source; it is a *leakage ceiling*, not evidence that any model uses it. Fold seed 0; a different seed will move it by a few hundredths. Tokens were read off log-ratios, not human-checked.
- RefBench-PRO's positive-side yes/no accuracy: not in the text I could fetch. FINER's negative-only accuracy by level: not extracted by Survey A and not fetched here.
- Legibility of the Rejection/Text strings at the P21 cap (D15) and head-noun presence on the 201 (D10): untested; both are < 0.2 GPU-h.
- VCD, MMI, speaker-listener, SelectiveNet, SNLI-VE, OOD scores are from memory and marked `[LIKELY]`; ids should be checked before citation.
- I did not re-read RC-GRPO, SAM 3 or GroundingME primary text; the audit of those claims relies on the surveys' verified extractions and disputes their *interpretation*, not their numbers.
