# Diagnosis (local evidence): what is wrong with Qwen3-VL-8B-Instruct on GroundingME Rejection

Author: `Diagnostician`, 2026-09-16. CPU-only re-analysis of this project's own tables; no GPU job was run.
Every number below is `[P: this run]` unless labelled otherwise. Scripts and raw output: scratchpad `diag.py`, `diag2.py`, `diag_out.txt`, `diag2_out.txt`.
Inputs: `tables/gme_original.jsonl` (1,005 items, ORIGINAL condition, decision-token p(null) at `{"bbox_2d":`), `tables/gme_remove.jsonl` (333 REMOVE edits on GroundingME positives), `tables/pilot_abstain.jsonl` (332 class-label removals × 3 conditions, OpenImages) + `pilot_abstain_not_clean.jsonl` (127), the GroundingME items parquet + `dimensions.csv` sidecar (`subtask_l2`, head noun).
CIs: bootstrap 95% (2,000 resamples) for AUROC; Wilson 95% for proportions. AUROC is P(score_absent > score_present), so 0.5 = no information and < 0.5 = signal in the wrong direction.

One structural fact first: our harness records the decision token, so `output_type` is only `box` or `none` and a parse-failure category cannot exist in these runs. **Our 0/201 is a true-decision zero**, not the official scorer's parse artifact.

---

## 1. Reproduction (task 1)

| quantity | value | note |
|---|---|---|
| Rejection abstentions | **0 / 201**, Wilson [0, 1.9%] | matches `GME-PRECHECK.md` |
| AUROC p(null), absent (201) vs present (804) | **0.298 [0.265, 0.335]** | matches `CORRECTION.md` §3 |
| … vs Discriminative positives only (204) | 0.468 [0.407, 0.523] | same image source and expression style: **no signal** |
| … vs Limited positives only (300) | 0.169 [0.137, 0.206] | driven by visibility, see §4 |
| … vs Spatial positives only (300) | 0.311 [0.266, 0.357] | |
| … vs positives answered correctly (333) | 0.353 [0.306, 0.398] | |
| … vs Discriminative positives with ≥36 words (134) | 0.565 [0.501, 0.629] | the honest **upper bound** of the absence signal on GME: length-matched, ≈ +0.3 orders |
| AUROC p(null), wrong (IoU<0.5, n=430) vs right (333), positives that boxed | **0.573 [0.533, 0.614]** | p(null) is barely a correctness signal |
| … Discriminative / Limited / Spatial | 0.536 [0.45, 0.62] / 0.648 [0.58, 0.71] / **0.494 [0.42, 0.57]** | the Limited value is size, not correctness |
| … wrong-object (IoU<0.05, n=338) vs right | 0.574 [0.530, 0.617] | the 42% wrong-object failures are as confident as hits |

p(null), log10 median [Q1, Q3], by dimension (abstentions included / box outputs only):

| dimension | n | all | box only |
|---|---:|---|---|
| Discriminative | 204 | −17.0 [−19.0, −14.5] | −17.0 |
| Limited | 300 | −12.5 [−15.7, −5.2] | −13.2 [−16.3, −8.9] |
| Spatial | 300 | −15.1 [−17.4, −13.0] | −15.1 |
| **Rejection** | 201 | **−17.5 [−18.8, −15.8]** | −17.5 |

The Rejection median (3.1e−18) is the **lowest** of the four dimensions. A Rejection item's p(null) sits at the 29th percentile of the whole benchmark (Q1 16, Q3 48). The single highest Rejection p(null) is 3.35e−4; **65/804 positives are above it**; **611/804 positives are above the Rejection median**.

Abstentions by dimension × size_bin (positives): Limited/tiny 5/8, Limited/small 25/80, Limited/medium 8/129, Limited/large 2/58, Discriminative/medium 1/28; all others 0. By `subtask_l2`: **Limited/Small 40/150 (26.7%)**, Limited/Occlusion 0/150, Discriminative/Text 1/50. Over positives by size_bin: tiny 62.5%, small 21.7%, medium 4.1%, large 0.8%, xl 0.0%.

---

## 2. Inside the 201 Rejection items (task 2)

| sub-axis (`subtask_l2`) | n | log10 p(null) median [Q1,Q3] | pct-rank in 1,005 | words (median) | pred box area (median frac) |
|---|---:|---|---:|---:|---:|
| Appearance | 50 | −17.6 [−19.0, −14.8] | 28.6 | 53 | 0.032 |
| Component | 51 | −17.3 [−18.3, −15.9] | 31.2 | 58 | 0.114 |
| State | 50 | −17.9 [−18.9, −16.3] | 25.0 | 53 | 0.041 |
| Text | 50 | −17.1 [−18.8, −15.7] | 33.3 | 53 | 0.039 |

- Kruskal-Wallis across sub-axes: H=1.96, **p=0.58**. AUROC Text > Appearance **0.496 [0.382, 0.612]**; Component > Appearance 0.506; State > Appearance 0.453; Text > State 0.565 [0.445, 0.677]. **Text-clause negatives carry no more p(null) than appearance ones.**
- Matched by sub-axis against Discriminative *positives*: Appearance 0.495 [0.39, 0.61], Component 0.460 [0.34, 0.58], State 0.556 [0.44, 0.67], Text 0.354 [0.25, 0.47]. Absent and present are indistinguishable at the decision token in every sub-axis.
- Length: Spearman(log p(null), words) within Rejection **ρ=−0.094, p=0.19**; sentences ρ=−0.12 (p=0.10); clause proxy (sentence ends + commas + and/with/which/that) ρ=−0.09 (p=0.22). No trend. (Within positives the same correlation is ρ=−0.23, p=9e−11, because short expressions are the Limited ones.)
- 81/201 expressions contain a quoted string (Text 49/50, Appearance 14/50, State 11/50, Component 7/51). Quoted vs not: AUROC 0.455 [0.372, 0.539].
- **The benchmark's negatives do not sample the regime where our removal signal lives.** Rejection expressions: median 54 words [46, 64]; only **3/201 (1.5%) are ≤20 words** (Discriminative 20.1%, Limited 81.0%, Spatial 2.7%). Those three are all Text — *license plate "552-H2"*, *bolt head "UME"*, *cargo plane "N967EE"* — and sit at log10 p(null) −16.9 / −17.9 / −15.5 with plate-, bolt- and plane-sized boxes (area 0.001 / 0.004 / 0.015). Anecdotal (n=3) but pointed: the checkable clause is present and short, and the model boxes the object of that category anyway.
- Head nouns: car 27, person 19, man 10, sign 9, boat 9, signage 7, toy 7, woman 6, building 5, tourist 5, plane 5. **99/201 are person-like or vehicles**; 78 distinct nouns vs 107 in Discriminative. Hedge words (*possibly / likely / appears*) in 59% of Rejection vs 50% Discriminative vs 13% Limited expressions.
- Top-8 Rejection items by p(null) (−3.5 to −11.5): a basketball player (State), a camouflage helicopter (Text), an arm, a boy, a Japanese sign, a tourist, a child, a garment — nothing systematic; no sub-axis, length or quote pattern.

Reading: within the 201, **nothing in the tables predicts p(null)**. The CORRECTION note's expression-style hypothesis is not testable on the benchmark's own negatives because 98.5% of them are in the long-paragraph regime; the 3 short ones and the 50 Text ones argue against it (see §6 for why).

---

## 3. Where the 201 boxes land (task 3)

| population | n | box area / image, median [Q1, Q3] | mean | >10% | >25% | >50% |
|---|---:|---|---:|---:|---:|---:|
| Rejection, predicted | 201 | **0.05 [0.02, 0.14]** | 0.107 | 33.8% | 11.9% | 2.5% |
| positives, predicted | 763 | 0.02 [0.00, 0.05] | 0.040 | 10.7% | 2.0% | 0.1% |
| Discriminative, predicted | 203 | 0.03 [0.02, 0.08] | | | | |
| positives, ground truth | 804 | 0.01 [0.00, 0.03] | 0.024 | 5.0% | 0.1% | 0.0% |

- AUROC(area) Rejection-pred > positives-pred **0.732 [0.694, 0.767]**; > Discriminative-pred 0.621 [0.567, 0.673]; > positives-GT 0.807 [0.774, 0.837].
- Only 2.5% of Rejection boxes cover more than half the image; 12.4% touch a border (positives 17.6%). Aspect ratio w/h 0.83 [0.49, 1.29] vs 0.84 [0.45, 1.34]. Box centres are *closer* to the image centre (offset 0.27 vs 0.34 of image dims).
- **p(box) ≥ 0.9996 on all 201** (100% > 0.999; positives 96.9%). Within Rejection, box area is uncorrelated with p(null) (ρ=+0.02, p=0.74); within positives it is (ρ=−0.33, p=4e−21: smaller box, higher p(null)).
- Component-axis boxes are largest (0.114): a "component" negative describes a whole person/vehicle and the box takes the whole thing.

Verdict: **object-sized, not whole-scene.** The model does not give up with a big box; it selects a plausible object of the head-noun category (car, person, boat, sign) with full confidence. The 2× larger area than positives is what car/person/boat-sized referents look like, not a scene box. This is "hallucinates a plausible referent", the same operation as the 338 wrong-object positives (`CAPABILITY-MECHANISMS.md` §1: correct size, wrong object).

---

## 4. The 41 abstained positives (task 4)

- Composition: 40 Limited/**Small** (tiny 5, small 25, medium 8, large 2) + 1 Discriminative/Text (medium). Limited/Occlusion: 0/150.
- **They are confident, not marginal**: p(null) median 0.9994, min 0.679, Q1 0.982; 68% > 0.99; only 10% in (0.5, 0.9). Log-odds margin log10(p(null)/p(box)) median 3.26, min 0.33.
- GT area: median **0.005% of the image**; non-abstained Limited/Small median 0.017%. AUROC(GT area) non-abstained-Limited > abstained **0.877 [0.822, 0.923]**.

Abstention and p(null) by GT-area bucket (804 positives):

| GT area / image | n | abstain | log10 p(null) median | Acc@0.5 of the boxes that were emitted |
|---|---:|---:|---:|---:|
| < 0.05% | 134 | **38 (28.4%)** | −4.5 | 36% |
| 0.05–0.1% | 32 | 1 (3.1%) | −12.4 | 42% |
| 0.1–0.2% | 51 | 0 | −15.1 | 31% |
| 0.2–0.5% | 83 | 1 (1.2%) | −15.1 | 37% |
| 0.5–2% | 229 | 1 (0.4%) | −15.8 | 47% |
| > 2% | 275 | 0 | −16.2 | 48% |

- Spearman(log p(null), GT area) over 804 positives **ρ=−0.476** (p=1e−46); within Limited **ρ=−0.694**; box outputs only −0.394.
- Separability from the true-absent population: AUROC p(null) abstained-positives > Rejection = **1.000** — in the *wrong* direction. The false abstentions occupy the top of the p(null) scale and the true absences the bottom; there is no threshold, and nothing else in the tables (dimension, size, words, box) that a deployed model would have at inference time, separates them in the right direction. The only separating variable is GT area, which is not available at inference.
- Near-abstain positives (0.01 < p(null) < 0.5, still boxed): n=14, Acc@0.5 21%, all small/medium.
- Cost of the over-abstention in points: the 41 items, answered at the 36% hit rate of boxed positives with GT area < 0.05%, would recover ≈14.9 hits = **≤1.9 points** on the 804. Small; the diagnostic value is what it reveals: **the model's `null` means "I cannot see it", not "it is not there."**

---

## 5. The class-label pilot, recomputed, and the threshold transfer (task 5)

`tables/pilot_abstain.jsonl`: 332 rows, class-label expressions of 2–4 words ("the Spoon"), OpenImages, LaMa removals with matched controls; ORIGINAL-correct (IoU ≥ 0.5) = **316**.

| quantity | value |
|---|---|
| AUROC REMOVE vs CONTROL_OBJ (316 vs 316) | **0.964 [0.950, 0.976]** (20 tied values; identical vs ORIGINAL: 0.964) |
| declines on REMOVE | 159/316 = **50.3% [44.8, 55.8]** |
| false abstentions on CONTROL_OBJ | **0/316** (rule-of-three upper 0.9%) |
| log10 p(null) medians ORIGINAL / CONTROL / REMOVE | −18.0 / −17.9 / **−0.2** |
| REMOVE items that still boxed (157) | median −6.3, i.e. 11.6 orders above control |

Threshold table (threshold = CONTROL_OBJ quantile at 1 − budget), **with the same threshold applied to GroundingME**:

| false-abstention budget | threshold | correct abstention on REMOVE (OpenImages) | GME Rejection caught | GME positives falsely abstained |
|---|---:|---:|---:|---:|
| today (0.5) | 5.0e−1 | 51.6% | 0/201 | 41/804 (5.1%) |
| 0% | 3.2e−1 | 52.5% | 0/201 | 41/804 |
| 1% | 5.4e−5 | 69.6% | 1/201 (0.5%) | 72/804 (9.0%) |
| 5% | 6.3e−10 | **83.2% [75.6, 88.3]** | **2/201 (1.0%)** | **126/804 (15.7%)** |
| 10% | 5.3e−13 | 90.8% | 10/201 (5.0%) | 229/804 (28.5%) |

Artifact leg (`pilot_abstain_not_clean.jsonl`, 108 ORIGINAL-correct, referent survived the inpaint): REMOVE(not_clean) vs CONTROL(clean, 316) = **0.583 [0.513, 0.652]** (the figure in `ABSTAIN-SIGNAL.md`); vs CONTROL(not_clean, 108) = 0.673 [0.602, 0.743] (the not_clean pool's own controls sit lower: AUROC 0.377 vs clean controls); REMOVE(clean, gone) vs REMOVE(not_clean, present) = **0.911 [0.877, 0.941]**; declines on not_clean 4/108. The decomposition in the note stands (artifact ≈1.4 orders, absence ≈16).

**What this experiment can say**: on a category-absent negative with a 2–4-word expression, the decision token carries a 16-order, AUROC-0.96 absence signal; the greedy policy reads it half the time; no false abstention on 316 matched controls; ~5% of the signal is inpainting artifact.
**What it cannot say**: anything about GroundingME-style negatives. Every one of its variables differs from GME Rejection at once — expression (2–4 words vs 54), negative type (category gone vs category present with one false clause), image source (OpenImages vs SA-1B), and provenance (edited vs never-there). The threshold row above is the transfer test: a threshold calibrated to 5% false abstention on OpenImages controls falsely abstains **15.7%** of GME positives (3× the budget) and catches **1%** of GME negatives. The OpenImages operating-point conclusion ("the information is present, the policy is the problem") does **not** carry to GME.

---

## 6. GroundingME removals reproduced, and the confound in the CORRECTION table (extra)

`tables/gme_remove.jsonl` (333 REMOVE edits on GroundingME positives, one condition; ORIGINAL p(null) joined from `gme_original.jsonl`; removal cleanliness of the 255 unreviewed items still unchecked, as `CORRECTION.md` warns). The CORRECTION tables reproduce exactly (all-333 by words: 14.1 / 27.5 / 0.0 / 0.0% declines, +2.3 / +3.3 / +1.1 / +0.6 orders, AUROC 0.708 / 0.706 / 0.633 / 0.597; Discriminative ≤20 words: 42.9%, +12.6 orders, AUROC 0.972 [0.925, 1.000]).

Now split by `subtask_l2`, which the CORRECTION note did not have:

| cell | n | words (median) | declines [Wilson] | log10 p(null) ORIGINAL → REMOVE (medians) | AUROC REMOVE vs ORIGINAL |
|---|---:|---:|---|---|---|
| Discriminative / **Text**, ≤20 words | 28 | 12 | **42.9% [26.5, 60.9]** | −16.0 → **−1.0** | 0.972 [0.923, 1.000] |
| Discriminative / **Text**, >20 words | 9 | 46 | 0% [0, 29.9] | −19.7 → **−10.3** (per-item shifts 14.0, 8.9, 13.7, 0.5, 7.7, 8.0, 10.5, 0.6, 4.6) | 0.901 [0.716, 1.000] |
| Discriminative / Appearance | 30 | 48 | 6.7% | +2.4 orders | 0.686 |
| Discriminative / Component | 36 | 49 | 0% | +1.2 | 0.630 |
| Discriminative / State | 21 | 46 | 0% | +0.5 | 0.656 |
| Discriminative / non-Text, ≤35 words | 14 | 32 | 14.3% [4.0, 39.9] | −15.1 → −14.4 | — |
| Limited / **Small** | 43 | 17 | **48.8% [34.6, 63.2]** | −10.5 → **−0.4** | 0.852 [0.769, 0.929] |
| Limited / Occlusion | 80 | 13 | **0% [0, 4.6]** | −15.2 → −13.8 | 0.632 |
| Spatial | 86 | 58 | 0% [0, 4.3] | −14.7 → −14.1 | 0.550 |
| class-label pilot (OpenImages) | 316 | 2–4 | 50.3% | −18.0 → −0.2 | 0.964 |

- **The 28 "Discriminative ≤20 words" items are the 28 Discriminative/Text items with ≤20 words — the two cells are identical.** Text expressions are short by construction (median 13 words vs 46–49 for the other three sub-axes). "Short checkable clause" and "text-string clause" are perfectly confounded in the removal data.
- Deconfounded: within Discriminative/non-Text (n=87), Spearman(words, shift) **ρ=+0.056, p=0.61** — no length effect at all. Within Discriminative/Text (n=37) ρ=−0.33 (p=0.045) — length modulates the text signal, but the >20-word Text items still shift a median +9 orders. Length is a minor modulator; **the sub-axis is the variable**.
- The same pattern inside Limited: Small and Occlusion have the same short expressions (17 vs 13 words) and opposite results (48.8% vs 0% declines). Small referents (GT median 0.025% of image) are already near the model's visibility floor (ORIGINAL −10.5); removing them leaves nothing of that kind. Occluded referents (0.77%) sit inside a scene that still holds the occluder and same-category neighbours.
- 94.3% of REMOVE boxes moved off the hole (IoU < 0.5 with the ORIGINAL box): after a removal the model finds a substitute, it does not re-box the gap.

The unifying reading (`[P: this run]` for the pattern, `[SPECULATION]` for the mechanism): **p(null) rises when no candidate matching the head noun plus its most salient discrete token (a text string; a tiny isolated blob) remains anywhere in the image. It does not rise when a same-category candidate remains and the mismatch lives in a clause.** Text strings act as part of the "gestalt" the model matches on (Qwen3-VL is OCR-strong), which is why removing a text-bearing object registers 12 orders while a text-bearing object with the *wrong* string (Rejection/Text, 49/50 with a quoted string) registers nothing: text-present is matched, text-identity is not verified. GroundingME Rejection is entirely in the second regime by construction (head-noun category present, one clause false; 99/201 person or vehicle), which is why its p(null) is the lowest of the four dimensions. This replaces `GME-PRECHECK.md`'s three-way ambiguity with a ranking: partial-match acceptance (explanation 3) is supported; expression *length* (explanation 1) is not; removal-vs-absence (explanation 2) is untested here and becomes probe 6 below.

---

## 7. Taxonomy of deficiencies

| # | deficiency | evidence (file, number) | strength | what would falsify it | cheapest sharpening probe |
|---|---|---|---|---|---|
| D1 | **Decoding operating point** — "the signal is there, greedy can't act on it" | For: pilot AUROC 0.964, 83% correct abstention at 5% budget (§5). **Against, on GME**: AUROC 0.298 [0.265, 0.335]; same-sub-axis absent-vs-present 0.35–0.56 (§2); the OpenImages 5%-threshold catches 2/201 and abstains 126/804 on GME (§5); catching half of Rejection costs 611/804 positives (§1) | **Strong** that the *decision-token* representation carries no absence signal on GME; i.e. on hard negatives this is **not** an operating-point deficiency | A linear probe on hidden states (decision token, or last image token) reaching AUROC > 0.75 absent-vs-present on GME with CV would show a readout deficiency instead | Probe 8 (hidden-state probe, 0.5 GPU-h). Cheaper still, probe 1 (existence VQA): if p("no") has AUROC > 0.8, the knowledge exists in a channel the box format cannot reach |
| D2 | **Partial-match acceptance / no clause verification** — a same-category candidate is accepted when one clause is false | Rejection sub-axis KW p=0.58, Text > Appearance 0.496 (§2); 3 short Text negatives at 1e−16 with plate/bolt/plane-sized boxes (§2); removals: signal lives in Text and Small cells, absent in Appearance/Component/State/Occlusion/Spatial at any length; no length effect within non-Text (ρ=0.06, n=87) (§6); head nouns car/person/boat present by construction (§2); boxes are object-sized, p(box) ≥ 0.9996 (§3) | **Strong** for the pattern (what p(null) responds to); **suggestive** for the mechanism (clauses never enter the decision) — same-category presence on Rejection images is `[LIKELY: benchmark construction]`, not measured | Text-swap edits on the 50 Discriminative/Text positives (object stays, string changes) producing a ≥5-order p(null) rise would show the model does verify a clause once the string is wrong; per-clause yes/no on the 201 answering "no" to the false clause at > 80% would show the check is available in isolation and the deficiency is integration | Probe 2 (per-clause verification) and probe 6 (text-swap N0′, RL-DESIGN's own design at n=50) |
| D3 | **No per-candidate score in the output channel** — the one scalar exposed (null vs box) is not a correctness score, and *which-object* uncertainty is not exposed at all | wrong-vs-right AUROC 0.573 [0.533, 0.614], Spatial 0.494 (§1); wrong-object (IoU<0.05) vs right 0.574; p(box) > 0.999 on 96.9% of positives and 100% of Rejection (§3); `ABSTAIN-SIGNAL.md`: box-on-the-hole failures at −7.2 vs hits at −8.1 | **Strong** that p(null) is not a correctness signal; **untested** whether any internal candidate-level score exists | First-coordinate-token entropy or top-2 margin with AUROC > 0.7 for wrong-vs-right would show a usable score exists in the channel | Re-run the 1,005 items storing the full coordinate-token distribution (0.5 GPU-h); zero design cost |
| D4 | **Format-scoring artifact** — official `evaluate.py` counts unparseable output as correct rejection | `SATURATION-AUDIT.md` §2.2 `[P: source read]`; our harness has no parse-failure category, 0/201 is a true decision (§0); every published non-zero Rejection number (thinking 5.5–9.5, best-of-16 15.9) is uninterpretable without a parse-failure rate `[LIKELY]` | **Strong** that the metric is contaminated; **untested** how much of the published thinking gain it explains | A thinking-mode run on the 201 whose literal-`null` rate ≈ its official Rejection score (parse failures ≈ 0) | Probe 3 (thinking run with parse-failure rate) |
| D5 | **Over-abstention is visibility, not absence** | 40/41 Limited/Small, GT median 0.005% of image; 28.4% abstention below 0.05% area vs ≤1.2% above 0.1%; ρ(log p(null), GT area) = −0.476, −0.694 within Limited; abstentions are confident (median 0.9994); AUROC absent vs Limited-present 0.169 (§1, §4); Small removals start at −10.5 (§6) | **Strong**. Cost ≤1.9 points; diagnostic value: `null` = "cannot see" | A pixel-budget sweep that leaves the Limited/Small abstention rate at 26.7% would mean it is not resolution-driven | The pixel-budget sweep already in `CAPABILITY-MECHANISMS.md` §7 step 4 (<1 GPU-h), read as abstention rate |
| D6 | **Metric inconsistency across "rejection" benchmarks** | One model, one decision token: class-label removals AUROC 0.964 / 50% declines; GME 0.298 / 0% (§1, §5). Cross-model: GME Rejection vs OpenRef N3R ρ = −0.304 `[P: CAPABILITY §5]`; gRefCOCO N-acc 77.2 vs GME 0.0 for the 4B `[VERIFIED per CHARTER]` | **Strong** | n/a (definitional); a single metric that ranks models consistently across category-absent / swapped-noun / one-clause-false negatives would dissolve it | None needed: report negative type with every number; never pool them |
| D7 | **Scale invariance of the zero** | Qwen3-VL 2B→235B all 0.0 non-thinking `[VERIFIED: 2512.17495 via CHARTER]`; this project has one model, so **nothing local** on whether p(null) *AUROC* moves with scale while the score stays 0 | **Strong** for the score; **untested** for the representation | Decision-token AUROC rising monotonically with scale (e.g. 0.3 → 0.6 at 32B) would mean scale improves the representation and only the operating point is stuck | Probe 7 (2B/4B locally, 32B on cloud) |
| D8 | **RL premise: pass@k ≈ 0 for `null`** — no on-policy positive rollout exists | Σ p(null) over the 201 = 3.4e−4; expected number of Rejection items with ≥1 `null` in k=16 rollouts at T=1: **0.005** (k=256: 0.08); at T=2: 0.29; T=4: 1.9 (and 87/804 positives falsely null); only at T=8 do 28 items sample `null`, at the price of 255/804 positives (§8 arithmetic) | **Strong** (arithmetic on measured decision-token probabilities; assumes the format prefix is near-deterministic, which p(box)+p(null) ≈ 1 supports) | An actual k=16 T=1 run producing ≥3 `null` outputs on the 201 (prefix variation mattering) | Probe 4 (50 items × 16 samples, 0.2 GPU-h) — mostly a check on the prefix assumption; the arithmetic already answers the design question: **forced/off-policy `null` rollouts or a separate channel are mandatory, GRPO on its own has nothing to reinforce** |

Two derived points for the brainstormers:

- **What `null` currently encodes** is "no candidate of the head-noun gestalt is visible", with visibility as the dominant driver (D5) and text strings and tiny isolated blobs as the tokens that count as the gestalt (D2). It is a category-presence detector with a visibility term, not a description-satisfaction detector. Any method that only moves the threshold on this scalar (D1) trades positives for nothing on GME.
- **The two GME failure modes are one operation.** The 338 wrong-object positives (right size, wrong object, full confidence) and the 201 confident boxes on Rejection are the same selection step ignoring clauses; `RL-DESIGN.md` §0 asserts this, and §3 + §6 here are the local evidence for it.

---

## 8. Probes to run later (not now), costed

Timing basis: 1.4 s/item for an 8B decision-token pass at the P21 resolution cap `[P: GME-PRECHECK]`; free generation ~10× that; thinking ~20–40× `[LIKELY]`. 8B bf16 ≈ 17 GB weights → ~20–22 GB GPU with image tokens; 4B ≈ 9–10 GB; host RAM ≤ 20 GB during load (WSL ceiling 23 GB `[P: memory note]`, one model resident at a time).

| # | probe | what it decides | dataset | model | GPU RAM | host RAM | GPU-h | where |
|---|---|---|---|---|---|---|---|---|
| 1 | **Existence yes/no VQA** on the 201 Rejection + 204 Discriminative positives: "Is there <expression> in this image? Answer yes or no", read p("no") at the answer token | D1 vs D2: whether the absence knowledge exists in a channel the box format cannot reach (AUROC > 0.8) or does not exist at all (≈0.5) | GME 405 items | Qwen3-VL-8B-Instruct | ~20 GB | ~18 GB | 0.3 | local |
| 2 | **Per-clause verification**: decompose each of the 201 expressions into atomic clauses (RL-DESIGN §9.4 decomposition prompt, text-only LLM, CPU/API), ask "Does any object in the image satisfy: <clause>?" per clause; also on 100 Discriminative positives as controls | D2: whether the false clause is detectable in isolation (the check exists, integration fails) or not (the check does not exist); gives a per-sub-axis map of which clause types the model can verify | GME 201 + 100 controls, ~6 clauses each ≈ 1,800 prompts | 8B-Instruct | ~20 GB | ~18 GB | 0.8 | local |
| 3 | **Thinking-mode run** on the 201 + 204 Discriminative with the official prompt; report literal-`null` rate, parse-failure rate and official score separately | D4: how much of the published thinking "rejection" is parse failure; also whether reasoning ever surfaces the false clause | GME 405 items | Qwen3-VL-8B-Thinking | ~20 GB | ~18 GB | 3–6 | local (sequential, not co-resident with Instruct) |
| 4 | **pass@k of `null`** at T=1, k=16, free generation, on 50 Rejection items (+ 50 Limited/Small positives to measure false-null sampling) | D8: confirms the 0.005 arithmetic and the prefix-determinism assumption | GME 100 items × 16 | 8B-Instruct | ~22 GB | ~18 GB | 0.2 | local |
| 5 | **Null-option prompt variants**: (a) the harness prompt, (b) + explicit "if no object satisfies every part of the description output `{"bbox_2d": null}`", (c) + "first check each clause"; declines on 201, false declines on 804, p(null) AUROC per variant | Whether any part of the zero is instruction-sensitivity (a cheap win) or none is (D1/D2 stand); also the cost of a permissive prompt on positives | GME 1,005 × 2 extra variants | 8B-Instruct | ~20 GB | ~18 GB | 0.8 | local |
| 6 | **Text-swap N0′ edits** on the 50 Discriminative/Text positives (render a different string on the same plate/sign; LaMa + text overlay, or plain overlay first) + C control (swap text on a non-referent), expression byte-identical | D2 directly: p(null) rise on object-present-clause-false with pixel-identical context; the missing negative type in `pilot_abstain`; also settles GME-PRECHECK explanation 2 (edit vs never-there) if the rise matches the Rejection/Text zero | 50 + 50 edits, 150 prompts | 8B-Instruct (+ LaMa) | ~22 GB | ~18 GB | 1.0 incl. editing | local |
| 7 | **Scale sweep** of the decision-token p(null) on all 1,005 | D7: does AUROC absent-vs-present move with scale (representation) while declines stay 0 (operating point) | GME 1,005 | Qwen3-VL-2B, 4B locally; 32B (bf16 ~65 GB) on cloud | 4B: ~10 GB; 32B: 80 GB | ~18 GB / 60 GB | 0.5 local + ~1 cloud (1× H100, ≈ $3) | local + cloud |
| 8 | **Hidden-state linear probe** (residual stream at the decision token and at the last image token, all layers), 5-fold CV absent-vs-present, stratified by dimension; store 1,005 × L × 4,096 floats (≈ 0.6 GB) | D1: readout vs representation at the source; if AUROC > 0.75 at some layer, a probe head is a candidate rejection channel; if ≈ 0.5 everywhere, rejection needs new supervision, not a readout | GME 1,005 | 8B-Instruct | ~22 GB | ~20 GB | 0.5 | local |

Suggested order by information per GPU-hour: 1 → 4 → 8 → 6 → 2 → 5 → 7 → 3. Probes 1, 4 and 8 together (≈1 GPU-h) settle D1 vs D2 before any training design is fixed.

---

## 9. Limits of this analysis, and disputes with existing notes

- One model, one revision (`0c351dd0`), one decoding pass. Everything about p(null) is the decision token after a fixed prefix; free-generation behaviour is inferred (D8) not measured.
- "Same-category candidate present on Rejection images" is the benchmark's construction, not something I measured; probe 1/2 would measure it.
- The 255 unreviewed GroundingME removals are still unchecked for cleanliness (`CORRECTION.md` §1); an unclean removal lowers p(null), so it cannot manufacture the Text-cell rise, but the 42.9% should carry that caveat.
- Text >20 words is n=9; Discriminative non-Text ≤35 words is n=14. The deconfounding rests on the n=87 null result (ρ=0.06) plus these small cells.
- **Dispute with `CORRECTION.md` §1 / `RL-DESIGN.md` §1.2**: the variable is not expression *length* ("≤20 words → 43%"); it is the sub-axis (a unique text string, or a tiny isolated referent). RL-DESIGN's instruction "prefer discrete checkable clauses, avoid graded paragraphs" survives, but its justification should be restated, and its Text-clause training items will inherit the same blind spot the benchmark exposes: text-present is matched, text-identity is not verified (§6).
- **Dispute with `ABSTAIN-SIGNAL.md`'s title claim** ("the model always knows; greedy cannot act on it"): true on OpenImages class labels, false on GME (D1). The 0.583 there is REMOVE(not_clean) vs the *clean* controls; within the not_clean pool it is 0.673.
- Not attempted: anything needing images (where exactly the Rejection boxes land, whether a head-noun instance is present), the Chinese-expression table (`gme_expr_zh.jsonl`), and the invalid head-noun probe (`gme_headnoun.jsonl`, per CORRECTION §2).
