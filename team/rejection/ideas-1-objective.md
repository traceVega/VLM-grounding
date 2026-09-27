# Ideas 1 — training-objective lens: how to train hard-negative rejection without losing positives

Author: `Ideas1`, 2026-09-16. Inputs: the five Phase-1 files, `notes/RL-DESIGN-CANDIDATE-VERIFICATION.md`, `notes/CAPABILITY-MECHANISMS-2026-09-06.md` §4 and §7. No GPU used. Two web checks this session (CFCamo, ICPO).

Labels. `[P: file]` project measurement. `[VERIFIED: id]` read by me this session. `[S-A/B/C/D: id]` number relayed from the named Phase-1 survey, which read the primary; I did not re-read it. `[E: CAP-MECH §n]` claim carried in the mechanisms note. `[LIKELY]`, `[SPECULATION]` as in the charter.

## 0. What every idea must respect, in one table

| constraint | number | consequence for objective design |
|---|---|---|
| D8: no on-policy `null` | Σ p(null) over 201 = 3.4e-4; expected `null` items at k=16, T=1: 0.005 `[P: diagnosis-local §7]` | plain GRPO has nothing to reinforce; the null must come from a forced string (RC-GRPO), a steered policy, a curriculum rung where pass@k > 0, or a different channel |
| D1: decision-token scalar carries no absence signal on GME | AUROC 0.298; same-sub-axis absent-vs-present 0.35–0.56 `[P: diagnosis-local §1-2]`; mid-layer probes reach 0.89–0.91 elsewhere `[S-B: 2603.05465, 2608.13167]`, unverified for REC | any threshold on p(null) is dead; a probe/steer route lives only if a mid-layer direction exists on GME (untested, 0.5 GPU-h) |
| D5: `null` = "cannot see" | 40/41 false abstentions are Limited/Small, GT area median 0.005% `[P: diagnosis-local §4]` | any objective that rewards null when the model *fails to localise* (TIAR/KARL p̂ logic applied naively) reinforces D5 and taxes Limited |
| D2: partial-match acceptance | removal signal lives only in Text (42.9%) and Small (48.8%) cells; App 6.7 / Cmp 0 / Sta 0 / Occlusion 0 / Spatial 0; no length effect within non-Text (ρ=0.056, n=87) `[P: diagnosis-local §6]` | the thing to teach is clause verification integrated into the box decision; category-absent negatives teach the wrong regime |
| null-in-stream SFT costs positives, monotone in fraction | 10% → −0.05, 20+20% → −6/−8, 67% → −21 `[S-B §3]`; GME 2:1: Dis 61.3→40.2, Lim 36.0→17.0, RefCOCOg 88.2→83.1 `[S-A: 2512.17495]` | either keep the negative fraction ≤ 30% and the update magnitude small (rule 3), or keep null out of the coordinate stream |
| separate presence channel is positive-safe for the *data*, not free for the head | SAM 3: hard negs 0→30/img: IL_MCC 0.44→0.68, pmF1 flat 62.4→62.8; presence head itself pmF1 65.4→63.4 (−2.0); loss masking on negatives beats DETR-style negative supervision 54.0 vs 52.2 cgF1 `[S-C: 2511.16719 Tables 9-10]` | the transplant must mask the box loss on negatives; whether the −2.0 recurs in an MLLM is the thing to measure |
| fixed positive null reward collapses; neutral + shaped advantage does not | AWA-RL 0.05 → ~100% refusal; TIAR coupled −0.76 acc; Abstain-R1 false-abstain −1.0 gives +8.4 acc `[S-B]` | zero fixed null bonus; null earns credit only conditionally (verified reason, or a flip against a twin) |
| binary-reward RL on positives-only erases residual abstention | Hallucination Tax 0.30→0.08; VenusBench-GD: every positive-only GUI specialist at 0 `[S-B: 2505.13988; S-C: 2512.16501]` | every RL arm needs negatives in the mix and an over-abstention canary; the base Instruct is the right start |
| RC-GRPO is one knob, never run on GME | α 0.2→1.0: N-acc 19.6→74.5 vs P-acc 73.6→60.9 `[S-A: 2608.04698]` | "RC-GRPO on GME" is the baseline every idea below must beat, not an idea |
| two open disputes | (a) length vs sub-axis: CORRECTION says ≤20 words → 43%; Diagnostician says the variable is Text/Small gestalt, ρ(words)=0.06 within non-Text. (b) format vs negative type: RefBench-PRO same items 15.8 (box) vs 64.2 (yes/no) `[S-B: 2512.06276]` says ~40 points are channel; the yes/no being near chance on hard negatives says the rest is verification | each idea states which resolution it assumes; where possible its precondition test *is* evidence on the dispute |

Baseline to beat: RC-GRPO's recipe run on GME (RL-DESIGN S3), predicted by rule 1 `[E: CAP-MECH §4]` to give Rejection well under the 13-point CI floor because the base's addressable error on L4 is not "never emits the string" but "cannot verify" — the 4B that already refuses got +8.1 on gRefCOCO, and GME is two rungs harder `[S-A §3]`.

Shared cost items (labelled estimates, from `[P: diagnosis-local §8]` rates: 8B decision pass 1.4 s/item, 4B ≈ 1 s; free generation ~3-10×; 4B ≈ 9-10 GB, 8B ≈ 20-22 GB; host ≤ 20 GB of the 23 GB WSL ceiling; 4B LoRA GRPO with co-located vLLM 28-31 GB `[RL-DESIGN §4]`):

| item | GPU-h | where |
|---|---|---|
| guardrail eval per checkpoint (GME 1,005 free-gen + RefCOCOg val 4,896 + DocVQA 1k + AI2D 1k), HF harness | 2-3 on 4B, 5-6 on 8B | local |
| the four preconditions bundle (probes 1, 2, 4, 8 of diagnosis-local §8) | ≈ 2.0 total | local |
| RL-DESIGN S0 A-arm (200 clausal items, flips, certification) | ≈ 1 | local, infra mostly built `[RL-DESIGN §9.5]` |
| RL-DESIGN S1 (3-5k certified items) | 15-30 | local |

## 1. Summary table

| # | name | what it changes | mechanism | one-sentence delta vs RL-DESIGN and RC-GRPO | first kill (GPU-h) | dispute assumption |
|---|---|---|---|---|---|---|
| 1 | PRESENCE-TURN | channel | D1, D5, D8 | absence gets its own token, its own loss and its own adapter; the coordinate policy is never updated on a negative and never sees `null` | 0.3 precondition; 3-4 headline | (b) partly channel; (a) agnostic, length-matched control built in |
| 2 | STEER-ROLLOUT GRPO | source of signal + bootstrapping | D8, D1, rule 3 | null rollouts are sampled from a probe-steered copy of the policy instead of injected as a constant string, so the advantage is item-conditional and on-format | 0.5 precondition (the probe); 18-27 headline | (b) readout: a mid-layer direction exists; (a) sub-axis: train on Text/Small cells, test all four |
| 3 | FLIP-PAIR GRPO with rung curriculum | objective + curriculum | D8, D5, collapse | null is never rewarded on its own — only a decision flip against the twin the policy just boxed earns credit; on-policy nulls come from starting at the rung where pass@k > 0 | 0.5 precondition; 5-8 stage A | (a) sub-axis: rung transfer is the test; (b) negative type |
| 4 | SELF-VERIFY DISTILLATION | source of signal | D2, D8, D1 | no external label source: the model's own per-clause yes/no channel labels the grounding channel, teaching the integration step D2 says is missing, and every self-labelled negative is adversarial to the current model by construction | 0.8 precondition (= D2 test); 10-13 headline | (b) channel; (a) sub-axis (predicts per-clause AUROC order) |
| 5 | SFT AT THE FORGETTING FLOOR | optimizer / curriculum | rule 3, fraction rule | nobody has checked whether the null-in-stream SFT cost that motivates all the RL machinery survives lr 1e-6 full-parameter at 10-20% negatives | 8-10 cloud + 5 local, headline directly | (b) negative type; (a) untested unless long flips added |
| 6 | NAME-THE-CLAUSE reward | objective | D2, collapse, D8 partly | null earns reward only when the rollout names the clause that fails, verified against the edit pipeline's flip record — an abstain switch cannot earn it | 1.5 precondition; 25-35 headline | (a) sub-axis (per-axis accuracy predicted); (b) negative type |

Cross-cutting control on every training arm: the lr 1e-6 full-parameter arm (cloud 1×80 GB) whenever a LoRA arm passes; rule 3 says it wins both axes and it costs 2-4 GPU-h per arm `[E: CAP-MECH §4]`.

## 2. The ideas

### Idea 1 — PRESENCE-TURN: a separately trained presence decision gates a template `null`; the coordinate stream never sees a negative

**Mechanism targeted.** D1 (build a channel that carries absence, since the decision token does not), D5 (the presence turn is trained only on negatives where the head-noun category *is* visible, so it cannot learn "cannot see"), D8 (both presence classes are on-policy: Qwen3-VL-8B emits "no" on hard negatives often enough to score 64.2 yes/no on RefBench-PRO `[S-B: 2512.06276]`, so plain cross-entropy works and no rollout forcing is needed).

**Method.** Two-turn protocol. Turn 1 (presence): `<image>` + description + "Does an object matching every part of this description exist? Answer yes or no." Loss = token CE on the yes/no token only — this is SAM 3's BCE presence loss `[S-C: 2511.16719 §A.1]`. Turn 2 (box): the official GroundingME prompt, byte-identical, run only when turn 1 = yes. **The presence turn trains a LoRA adapter that is active only during turn 1; turn 2 runs the base weights with no adapter.** So the coordinate policy is literally the base — SAM 3 Table 10 rule (a), per-query loss masked on negatives, taken to its limit `[S-C]`. Inference: no → emit `{"bbox_2d": null}` by template (parse failure impossible on the gated path); yes → turn 2's box. Data: 1:1 positives (RefCOCOg + clausal P1) and hard negatives N0 (text flip) + N0' (attribute edit) + C controls from RL-DESIGN §1.3, **adversarially filtered against the current model twice**: keep a negative only if the base *boxes* it (RL-DESIGN gate 1, ≥90%) and the base presence turn says *yes* (SAM 3 §D.4: a negative the model already rejects teaches nothing `[S-C]`). Re-mine after each epoch (SAM 3 data-engine loop). Variant arm (count-turn): turn 1 asks "how many objects satisfy every part: 0 / 1 / 2 / 3+"; 0 gates null, ≥2 gates a multi-box turn. This is HieA2G's count head (+12 N-acc in a specialist `[S-A: 2501.01416]`) and it runs RL-DESIGN's k>1 acceptance test without a single multi-target training item.

**Closest prior work.** (1) SAM 3 presence token `[S-C: 2511.16719]`: global BCE, per-query loss gated off on negatives; hard negatives 0→30/img move IL_MCC 0.44→0.68 with pmF1 flat; the head itself costs pmF1 −2.0. Delta: transplanted into an MLLM as a yes/no turn with its own adapter, negatives are one-clause-false rather than confusable-class, and the −2.0 is designed away by leaving turn 2 on base weights. (2) GSVA `[REJ]` / PostAlign `<REJ>` `[S-C: 2312.10103; 2506.17901]`: a dedicated null token with BCE, in the *same* stream as `[SEG]`, never tested on attribute-false negatives. Delta: separate turn and adapter; hard negatives; positive cost reported. (3) RefBench-PRO Table 5 `[S-B: 2512.06276]` and MEGA-GUI's separate refuser stage (38.9 → 68.5 on OSWorld-G refusal by decoupling deliberation from the click, prompting only) `[S-C: 2511.13087]`: the untrained and the prompt-only versions of this channel. Delta: the channel is trained on mined negatives.

**Kill experiment.** *Precondition (instrument-only, 0.3 GPU-h; = diagnosis-local probe 1):* zero-shot presence turn on 201 Rejection + 204 Discriminative positives, read p("no"). AUROC < 0.60 means the channel has no head start; the idea then survives only as a pure data bet (SAM 3 says data ≫ head), so record and continue with the prediction lowered. *Headline (tests the claim "decoupling removes the positive cost"):* train the presence adapter on 3k items (1.5k P1, 1.5k mined N; 1 epoch, LoRA r=32) on 4B; matched control arm = the same 3k items as null-in-coordinate-stream SFT (the GroundingME recipe at 1:1). Pre-registered stop rule: pass iff GME Rejection (gated) ≥ 13.0 (CI floor on n=201) AND false-"no" on the 804 positives ≤ 5% overall and ≤ 8% on Limited/Small AND Discriminative/Limited/Spatial drop ≤ 2.0 AND RefCOCOg val drop ≤ 1.0. Because turn 2 is base weights, any non-rejection drop can only come from false-"no" gating, so the second and third conditions are the same number — that is what makes this kill clean. Kill if Rejection < 13 or false-no > 8% on Limited/Small. *Cheaper decisive variant:* 1k items, 1 GPU-h, presence adapter only; the control arm can wait. *Protocol swap:* evaluate the trained system under the official single-turn prompt as well — Rejection is expected to stay ≈0 there by design; report it, and scope the claim to the rejection channel (rule 2 `[E: CAP-MECH §4]`: this is not a target-selection gain).

**Cost table (labelled estimates).**

| step | dataset | model | GPU RAM | host RAM | GPU-h | where |
|---|---|---|---|---|---|---|
| precondition | GME 405 items | 4B / 8B Instruct | 10 / 22 GB | 18 GB | 0.3 | local |
| data | 3k P1/N0/N0'/C via RL-DESIGN S0-S1 A-arm (+B-arm for N0', needs the attribute editor, ask first) | Qwen3.5-9B, Molmo2-8B, Gemma4-12B sequential | ≤ 24 GB | 18-20 GB | 4-8 (+6 with edits) | local |
| presence adapter | 3k × 1 ep | 4B LoRA (plain SFT, no vLLM) | ~16 GB | 18 GB | 1-2 | local |
| control arm | same 3k, null in stream | 4B LoRA | ~16 GB | 18 GB | 1-2 | local |
| eval | guardrail suite | 4B | 10 GB | 18 GB | 2-3 per ckpt | local |
| 8B replication | same | 8B LoRA | ~30 GB → cloud | — | 3-4 | cloud 1×80 GB |

**Predicted outcome.** Rejection 15-30 (the yes/no channel starts near 64% on L2 negatives; budget the 3× rung decay `[E: CAP-MECH §4 rule 1]`); false-"no" on positives 3-6%, concentrated on Limited/Small (the IDK-token precedent: recall cost lands on the long tail `[S-B: 2412.06676]`); Discriminative drop within the false-no rate. Kills: Rejection < 13 (channel training does not reach L4 from L2/L3 negatives), or false-no > 8% on Limited/Small (the gate inherits D5 after all), or the control arm shows the *same* positive cost as the gated arm (then decoupling bought nothing and the cost is in the gate, not the stream).

**Guardrails.** Over-abstention canary = false-"no" on 804 positives every 100 steps, stratified by dimension and by length (Discriminative ≥36 words, n=134, reported separately — this is the length-vs-sub-axis control). Non-rejection dims ≤ 2.0; RefCOCOg ≤ 1.0; parse-failure rate reported for turn 2 (0 for the gated path by construction); Rejection by sub-axis (App/Cmp/Txt/Sta) and by negative source (N0 vs N0' held-out); protocol swap as above.

**Traps and controls.** (i) Format-scoring artifact: the gate could mask a broken turn 2 — report turn-2 parse failures and the un-gated Rejection. (ii) Text shortcut on N0 (flipped sentence reads oddly → "no"): N0' held-out (byte-identical text) and the N0−N0' gap; grey-image control — the presence turn on a grey image must be ≈ chance, else it is a text prior. (iii) "Long description → no" prior (POPE-style yes-bias in reverse): the length-matched positive false-no rate above. (iv) Edit artifact: C controls (non-referent edited) must stay "yes"; baseline 0/316 false abstention on controls `[P: ABSTAIN-SIGNAL]`. (v) Two-call deployment: scope the claim.

**Disputes assumed.** (b): the RefBench-PRO 40-point same-item gap is real channel headroom; if the precondition AUROC is ≈0.5 on GME, the L4 zero is all verification and Idea 1 becomes a data bet. (a): agnostic; the length-matched control is the evidence either way.

### Idea 2 — STEER-ROLLOUT GRPO: replace the forced `None` with null rollouts sampled from a probe-steered copy of the policy

**Mechanism targeted.** D8 (sample from a nearby policy where null has mass, conditional on an internal absence direction firing), D1 (the decision-token scalar is dead, but if a mid-layer direction exists — TRAPSBench 0.91, VA neurons, HALP 0.89 `[S-B/C]` — the steered rollouts inherit its item-conditional information; a forced constant string carries none), rule 3 (steered rollouts are full generations in the model's own format distribution; RL-DESIGN's forced `{"boxes": []}` is exactly the "forced rollout breaks on-policy format preservation" case `[E: CAP-MECH §4 rule 3]`).

**Method.** (1) Direction: difference-in-means at layer ℓ (mid, 10-20 of 36) at the decision-token position, from REMOVE vs CONTROL_OBJ on the class-label pilot (316 vs 316, where p(null) separation is 0.964 so the direction is well-defined) `[P: pilot_abstain.jsonl]`; later from N0' pairs. (2) Steering: add α·d at ℓ with a CR-VLM sigmoid gate so positives are protected (over-refusal 0-12% in their setting `[S-C: 2602.07013]`). (3) GRPO: per item, G=8 = 6 on-policy (vLLM) + 2 from the steered HF trainer copy (hooks are impossible in vLLM; the trainer's HF model with the LoRA already exists, so generate the 2 samples there before the vLLM rollouts). Compute log π_θ(y) and log π_steer(y) for the steered samples (one hooked forward) and use ICPO's regularised importance weight f(x)=x/(x+λ), λ=0.01 `[LIKELY: 2510.26519 via summary]`, or PPO clipping. **Steered samples are drawn on positive items too** — those are the over-refusal samples, priced at −λ_refuse, so the policy learns exactly where the direction misfires. (4) Reward: RL-DESIGN's set-F1 with over-refusal penalty; zero fixed null bonus beyond r=1 on G=∅; α negative-advantage scaling kept. (5) Log per step the fraction of nulls on negatives that came from on-policy vs steered samples; the on-policy share must rise, or the policy is not absorbing the behaviour and steering is acting as SFT-in-disguise.

**Closest prior work.** (1) RC-GRPO forced-rejection rollout `[S-A: 2608.04698]`: constrained decoding of the refusal when no rollout refuses. Delta: replace the constant with sampled generations from a steered policy; over-refusal samples on positives come free. (2) ICPO `[LIKELY: 2510.26519]`: same model, off-policy trajectories via in-context expert demonstrations, 7 on-policy + 1 steered per group, regularised importance weights, annealed expert bonus; +4.1 on Qwen3-1.7B math. Delta: activation-level steering from an absence probe instead of in-context demonstrations; target is a binary decision, not reasoning. EEPO's unlearning-forced exploration `[LIKELY: 2510.05837]` is the same motive with a third mechanism. (3) TRAPSBench steering + CR-VLM gate `[S-C: 2608.13167; 2602.07013]`: a single-layer void direction causally induces abstention (75% at layer 20, α=10) with a gate bounding over-refusal. Delta: the direction is a rollout generator inside RL, not an inference-time hook; the weights end up carrying the behaviour and the hook is removed at eval.

**Kill experiment.** *Precondition 1 (instrument-only, 0.5 GPU-h; = diagnosis-local probe 8 and Survey C §3.3's "experiment nobody has run"):* dump residual-stream states at the decision token (and the last image token) for all 1,005 GME items and the 632 pilot items, all layers; train the direction/probe on the pilot, test on GME Rejection vs positives, 5-fold, stratified by dimension. **Stop: best-layer AUROC < 0.70 → no direction to steer with → the idea is dead before any RL.** This is also the cheapest test of whether D1 is a representation failure or a readout failure and is worth running for the diagnosis alone. *Precondition 2 (0.3 GPU-h):* (α, ℓ) sweep with the gate on 100 Rejection + 100 positives: need ≥ 25% null on Rejection at ≤ 10% on positives; no operating point → dead. *Headline:* 4B LoRA GRPO, 2k items (1k P1, 1k N0/N0'), 300 steps, three arms with identical data and seeds: steered / forced-None / no injection. Stop rule: steered arm GME Rejection ≥ forced-None arm + 5 AND positive false-null ≤ forced-None arm AND guardrails; otherwise steering buys nothing over the constant and forced-None stays the default. *Cheaper decisive variant:* precondition 1 alone carries most of the information.

**Cost table.**

| step | dataset | model | GPU RAM | host RAM | GPU-h | where |
|---|---|---|---|---|---|---|
| precondition 1 | 1,637 items, states at 2 positions × 36 layers (≈0.6 GB fp16, stream to disk) | 4B (or 8B) HF eager | 12 / 22 GB | 20 GB | 0.5 | local |
| precondition 2 | 200 items × ~12 (α,ℓ) settings | 4B HF | 12 GB | 18 GB | 0.3 | local |
| GRPO 3 arms | 2k items, 300 steps, G=8, LoRA r=32 | 4B + co-located vLLM | 28-31 GB | 20 GB | 3 × 6-9 = 18-27 (+40% wall time for HF-generated steered samples) | local, or cloud 3×80 GB in parallel |
| eval | guardrail suite × 3 | 4B | 10 GB | 18 GB | 6-9 | local |

**Predicted outcome.** Precondition 1: I put ~55% on AUROC < 0.70 — TRAPSBench's own finding that visual voids are 4× less detectable than textual ones and 2410.02707's non-transfer of probes across task types `[S-C]` both predict that a direction found on class-label removals will not fire on attribute-false referents, which is precisely the Text/Small-vs-rest split in `[P: diagnosis-local §6]`. If it passes (≥ 0.75): Rejection 10-20 with the on-policy null share > 50% by step 300; forced-None arm 5-12. Kill: AUROC < 0.70; or steered arm ≤ forced-None + 5.

**Guardrails.** All six; plus steered-sample over-refusal on positives logged per step; grey-image control on the direction (a direction that fires on grey images is a text prior); clipped-fraction of steered samples logged (if most are clipped away, the arm is not learning from them).

**Traps and controls.** (i) The direction trained on removals encodes edited-ness or tiny-blob visibility (D5), not absence: report probe AUROC on CONTROL_OBJ vs ORIGINAL (must be ≈ 0.5) and on Limited/Small positives vs other positives (if it separates small positives, it is a visibility direction). (ii) Steered rollouts may be malformed → parse failure = reward 0, logged. (iii) The L3-trained direction may not fire on L4 — precondition 1 catches it. (iv) Without importance weights this is SFT in disguise (rule 3).

**Disputes assumed.** (b): readout — the representation exists mid-layer. (a): sub-axis — the direction is trained where the signal exists (Text/Small cells) and tested on all four sub-axes; if it fires only on Text, that is dispute-(a) evidence for the Diagnostician.

### Idea 3 — FLIP-PAIR GRPO with a rung curriculum: reward the decision flip within an (original, edited) pair, bootstrap where pass@k > 0

**Mechanism targeted.** D8 (pass@k for null is 0 at L4 but 50% on class-label removals `[P: pilot]`, 43-49% in the Text/Small cells of GME removals `[P: diagnosis-local §6]`, and CFCamo's base abstained on 88.7% of counterfactuals before training `[VERIFIED: 2606.11231 html]` — so start where on-policy nulls exist and climb), the collapse pair (always-refuse and always-box both score 0 under a flip reward; there is no fixed null bonus at all), D5 (a flip must be accompanied by a hit on the twin, so a "cannot see" abstention earns nothing because it does not flip).

**Method.** Item = pair (I⁺, I⁻, e), expression byte-identical; I⁻ = removal at rung L3, attribute/text edit N0' at rung L2-edit. G rollouts per side; rollout i on I⁺ paired with rollout i on I⁻ (or all G×G pairs averaged for lower variance). Per-pair reward in CFCamo's form `[VERIFIED: 2606.11231 Eq. 5]`: R = Hit(y⁺) − Box(y⁻) + γ[Null(y⁻) − Null(y⁺)] + Hit(y⁺)·Null(y⁻), γ=2, with Hit = box on I⁺ at IoU ≥ 0.5 (CFCamo's Det is replaced by an IoU hit so a wrong box on I⁺ earns nothing); correct pair 4, over-box 0, over-abstain 0. Advantages normalised within the pair-group. Curriculum: stage A on class-label removals (pilot regime); stage B on GME-image removals restricted to GT area > 0.2% (exclude the visibility cells, see traps); stage C on N0' attribute/text edits, where pass@k ≈ 0 at the start — the whole question is whether A+B raise it above 0. Unpaired N0 text flips with a plain r ∈ {0,1} enter only after stage C, never before (they reintroduce the fixed-bonus collapse route). 20-30% plain positives with r = IoU-hit keep the box policy alive (the 10% rule in reverse `[S-B: 2505.13988]`). No SFT cold start (CFCamo used one; testing whether the curriculum replaces it).

**Closest prior work.** (1) CFCamo `[VERIFIED: 2606.11231 html]`: per-pair reward with coupling bonus, γ=2, G=8 per side, Qwen3-VL-4B-Instruct, ObjectClear removals, 4,040 pairs, one epoch SFT then half-epoch RL; PA 1.4% without coupling vs 87.7-88.2% with; S_α 0.844 → 0.875 vs Seg-R1-7B (SurveyB's abstract read of "+3.7 pp" is against a different baseline; the html gives +0.031 S_α — the two are not the same number and I could not reconcile them). Delta: climb from removal (their only rung) to attribute edit; REC not COD; expression held fixed; IoU hit not detection; no SFT cold start. (2) FineCops-Ref pairwise metric `[S-A: 2409.14750]`: positive-negative pairs as a *ranking* eval with no abstention. Delta: the pair is the training signal and abstention is required. (3) GOBL / D-Negation `[S-C: 2603.12606]`: paired positive/negated descriptions on one attribute, positive-safe (+1.1 RefCOCO testA). Delta: image-side pair, MLLM, RL.

**Kill experiment.** *Precondition (instrument, 0.5 GPU-h; = diagnosis-local probes 4 + 6):* base pass@8 for null at T=1 on (i) 100 class-label removals, (ii) 100 GME removals by cell, (iii) 50 N0' edits (needs the B-arm editor). Expected ≈ 50% / 0-45% by cell / ≈ 0. Confirms the ladder has a bottom rung with on-policy signal. *Headline (the rung-transfer number nobody has):* after stages A+B, null pass@8 on held-out N0' edits. **Stop rule: < 10% (from ≈0) → stage C has no on-policy signal, the curriculum does not climb, the idea dies** (fallback is injection, i.e. Idea 2 or RL-DESIGN's forced None). ≥ 10% → run stage C and require GME Rejection ≥ 13 with guardrails. *Cheaper decisive variant:* stage A only on the 316 clean pilot pairs plus their controls (already edited, already verified: zero data cost) for 200 steps, then measure N0' pass@8 — one number, ≈ 5 GPU-h, decides the curriculum.

**Cost table.**

| step | dataset | model | GPU RAM | host RAM | GPU-h | where |
|---|---|---|---|---|---|---|
| precondition | 250 items × 8 samples | 4B | 10 GB | 18 GB | 0.5 | local |
| stage A data | existing 316 pilot pairs + controls; +200 removals | LaMa + SAM 3 | 12-18 GB | 18 GB | 2 | local |
| stage A GRPO | 500 pairs, G=8 × 2 sides, 200 steps, LoRA | 4B + vLLM | 28-31 GB | 20 GB | 5-8 | local |
| stage B/C data | 500 N0' pairs at ~50% clean → 1,000 edits | attribute editor (ask-first download) + SAM 3 | 12-18 GB | 18 GB | 6-10 | local |
| stage C GRPO | 500 pairs, 300 steps | 4B + vLLM | 28-31 GB | 20 GB | 8-12 | local |
| eval | guardrail suite per stage | 4B | 10 GB | 18 GB | 2-3 each | local |
| 8B | all stages | 8B LoRA + vLLM | > 32 GB | — | 15-20 | cloud 1×80 GB |

**Predicted outcome.** Stage A lifts class-label abstention 50 → 85%+ (CFCamo-like, and our own threshold table shows 83% is available at 5% FP `[P: diagnosis-local §5]`). N0' null pass@8 after A+B: 5-15% — I put the stop rule at 50/50, because the removal signal lives in the head-noun gestalt (text string, isolated blob) which an attribute edit leaves intact `[P: diagnosis-local §6]`. If it climbs: GME Rejection after C 8-15, i.e. a real risk of landing under the CI floor. Kill: N0' pass@8 < 10% after A+B.

**Guardrails.** Over-abstention canary on the plain-positive mix and on C-pairs; non-rejection ≤ 2.0; RefCOCOg ≤ 1.0; parse-failure rate; stratified by rung and sub-axis; protocol swap (train `{"boxes": []}`, eval `{"bbox_2d": null}`); eval only on unpaired GME.

**Traps and controls.** (i) "Edited → refuse" shortcut: build 20% of pairs as C-pairs (non-referent edited; the correct pair there is box-box, so a flip is *penalised*). (ii) Visibility reward (D5): a flip on a tiny-object removal rewards "cannot see" — stage A/B restricted to GT area > 0.2%, and Limited/Small false-null tracked. (iii) Pair leakage into eval — GME is unpaired. (iv) Format-scoring — parse failures logged.

**Disputes assumed.** (a): sub-axis — the rung-transfer stop rule is exactly the test of whether removal-type signal reaches clause-type negatives. (b): negative type — the pair reward manufactures the negative-type signal; format is handled by the protocol swap.

### Idea 4 — SELF-VERIFY DISTILLATION: the model's own per-clause yes/no verdicts label its grounding decision; no external negatives, no edits

**Mechanism targeted.** D2 (the check may exist per clause but not enter the box decision — teach the integration), D8 (the yes/no channel has both classes on-policy; the grounding channel is trained by preference optimisation on self-labels, so no null rollouts are needed), D1 (move information from the channel where it exists to the one where it does not).

**Method.** For each (image, expression) in an unlabeled pool (the 9,692-scene instance bank with ≥3 same-category instances `[RL-DESIGN §9.5]`; RefCOCOg train; S0 clausal expressions): (1) decompose e into atomic clauses (text-only LLM, CPU; RL-DESIGN `expr_decompose`). (2) Base grounds e → box b; SAM 3 (or the base's own multi-box output) enumerates same-category candidates. (3) For every candidate and every clause, the *base itself* answers the `verifier_clause_instance` prompt on the red-outlined crop, reading p(no). (4) Self-label: every candidate fails ≥1 clause with p(no) > τ → label null; exactly one candidate passes all → label that box; otherwise discard. (5) Train the grounding turn with DPO/ORPO, not SFT (Visual-Idk: SFT −17.9 IK-IK, ORPO −5.5 `[S-B: 2604.26419]`): on self-null items chosen = null, rejected = the base's own box; on self-box items chosen = base box, rejected = null. Every self-null item is, by construction, a hard negative w.r.t. the current model (the grounder boxed it, the verifier rejects it) — SAM 3's adversarial filter, self-supervised, at zero label cost. Iterate: re-label with the updated model.

**Closest prior work.** (1) MCC / OpenRef `[S-A: 2605.25706]`: training-free count-vs-detect consistency, N3R 38.1 → 91.9 on 2B but 78.7 → 73.0 on GLM-4.6V. Delta: consistency is trained in, per clause not per count, no inference overhead, and the "hurts the strongest model" failure should not recur because a stronger model has a stronger verifier. (2) R-Tuning / Alignment-for-Honesty self-labelling `[S-B: 2311.09677; 2312.07000]`: unknowns labelled from the model's own sampling; prudence 0 → 59 at −0.4 accuracy. Delta: the label comes from a different *channel* of the same model on the same item, not from sampling consistency — SurveyB §5 says sampling cannot find hard negatives because they are consistently wrong with high confidence. (3) Ferret spatial negative mining with yes/no + counter-box and the SESAME cascade (LLaVA verifies, LISA segments, FP-detect 75.6) `[S-D: 2310.07704; S-A: 2312.08366]`. Delta: verifier and grounder are the same weights and the cascade is distilled away.

**Kill experiment.** *Precondition (instrument-only, 0.3-0.8 GPU-h; = diagnosis-local probe 2, and the direct test of D2's "check exists, integration fails" reading):* on 200 S0 N0 items (flipped clause known) and 100 Discriminative positives, per-clause p(no) on the flipped clause vs the unchanged clauses. Need AUROC ≥ 0.80 for flipped-vs-unchanged and false-"no" ≤ 15% on positives' clauses. At chance → there is no teacher → dead. *Secondary precondition:* self-labels vs Gemma4 labels on the same 200 items; agreement < 70% → teacher too noisy → dead. *Headline:* DPO on 3k self-labelled items (≈1.5k self-null, 1.5k self-box), 4B LoRA, 2 epochs. Stop rule: GME Rejection ≥ 13 AND false-null on 804 ≤ 5% AND guardrails. *Cheaper decisive variant:* the precondition; then a 500-item DPO in 1 GPU-h.

**Cost table.**

| step | dataset | model | GPU RAM | host RAM | GPU-h | where |
|---|---|---|---|---|---|---|
| precondition | 200 N0 × ~4 clauses + 100 pos × 4 ≈ 1,200 crop-VQA calls | 4B | 10 GB | 18 GB | 0.3-0.8 | local |
| self-label pool | 5k (image, expr) × (1 ground + ~4 candidates × ~4 clauses ≈ 17 calls) ≈ 85k short calls at ~0.3 s | 4B + SAM 3 (5 GB) | 15 GB | 18 GB | 7-10 | local |
| DPO | 3k pairs × 2 ep, LoRA (reference = adapter disabled) | 4B | ~20 GB | 18 GB | 2-3 | local |
| eval | guardrail suite | 4B | 10 GB | 18 GB | 2-3 per ckpt | local |
| 8B | pool + DPO | 8B | ~30 GB → cloud | — | 12-15 | cloud 1×80 GB |

**Predicted outcome.** Precondition: per-clause AUROC 0.70-0.85 on Text and Appearance clauses, lower on State and Component (FPCO-Dialog's ordering identity 0.56 > attribute 0.41 > location 0.15 `[S-B: 2609.03331]`; FINER ~80% at one clause `[S-A: 2603.17662]`). Headline: Rejection 8-18, false-null 4-8%. Kill: precondition AUROC < 0.70; teacher-Gemma agreement < 70%; Rejection < 13.

**Guardrails.** Over-abstention canary every 100 DPO steps (DPO with null as chosen still moves null into the coordinate stream); self-box labels restricted to items where Molmo2 agrees or RefCOCOg GT exists (a wrong base box as "chosen" is wasted, not harmful, but keep it clean); stratified by sub-axis; protocol swap; parse rate; grey-image control (self-labels on grey images must be near-uniform, else the verifier is a text prior).

**Traps and controls.** (i) Blind-spot reproduction: the policy discards negatives it cannot verify, so the self-null set systematically lacks the hardest cells — measure the sub-axis distribution of self-null labels against Gemma4 labels and report the gap. (ii) Yes-bias in crop-VQA (POPE 2023 style) → few self-nulls: report p(no) calibration on known-false clauses. (iii) The lineage rule (`lineage.py`: the policy must not certify its own data) is deliberately broken here — that is the hypothesis; the Gemma4 agreement number is the safety check.

**Disputes assumed.** (b): channel — the yes/no channel holds what the box channel lacks (64.2 vs 15.8). (a): sub-axis — predicts the per-clause AUROC order Text > Appearance > State ≈ Component, a side-prediction the precondition tests.

### Idea 5 — SFT AT THE FORGETTING FLOOR: rerun the GroundingME negative-mix SFT at lr 1e-6 full-parameter and 10-20% negatives

**Mechanism targeted.** Not a deficiency of the base but of the training cost: rule 3 (off-target regression = format-entropy loss × effective update magnitude `[E: CAP-MECH §4]`) and the monotone negative-fraction rule `[S-B §3]`. If it passes it reprices every idea above: SFT becomes the baseline they must beat.

**Method.** Reproduce the recipe class of GroundingME Table 7 `[S-A: 2512.17495]` (their exact negatives are unpublished: "modifying the description"): RefCOCOg positives + N0 text-flip negatives built with RL-DESIGN's `expr_flip_clause`, certified by Gemma4 (zero-satisfier) and by the difficulty gate (base boxes ≥ 90%, otherwise the arm learns L1 rejection and the −21 fails to reproduce for the wrong reason), labelled `{"bbox_2d": null}` in the official format. 6k items, not 30k (RC-GRPO needed 2k; the paper's 30k is not a requirement). Arms: (i) LoRA r=32, lr 1e-4, 2:1 neg — expected to reproduce the paper's cost; (ii) full-parameter lr 1e-6, 2:1; (iii) full-parameter lr 1e-6, 20% neg; (iv) full-parameter lr 1e-6, 10% neg; (v) = (iii) + 10% general VL replay (DocVQA/AI2D-style) for format-entropy replay. 1 epoch each, plus a 3-epoch version of (iii) because 6k items at lr 1e-6 ≈ 750 steps may under-train (unknown examples are learned slowly `[S-B: 2405.05904]`). Add 1k flips on the S0 long clausal expressions so the length dimension is not entirely untested.

**Closest prior work.** (1) GroundingME Table 7 `[S-A: 2512.17495]`: 1:8 → 2:1 gives Rej 3.5 → 27.9 with Dis 57.4 → 40.2; even 1:8 loses 4 Dis. Delta: same recipe class at the forgetting floor, negatives certified. (2) Hallucination Tax +10% SUM `[S-B: 2505.13988]`: 10% unanswerables restore refusal 0.08 → 0.85 at GSM8K −0.05. Delta: grounding, hard negatives, null in the box stream. (3) The lr ÷10 full-parameter result carried in `[E: CAP-MECH §4 rule 3]` (−0.19 pp on target, 17.6/32.1 pp off-target regression removed, beats LoRA on both axes). Delta: applied to negative SFT.

**Kill experiment.** *Headline directly, no precondition.* Stop rule: **pass** iff arm (iii) or (iv) gives GME Rejection ≥ 13 with Discriminative drop ≤ 2.0 and RefCOCOg ≤ 1.0 — then SFT is the baseline and every RL idea must beat it on the same guardrails. Three informative failure shapes: Rejection ≥ 13 only with Dis drop > 4 at every lr and fraction → the cost is intrinsic to null-in-stream, which is Idea 1's premise; Rejection < 5 at 10-20% and lr 1e-6 → low lr does not learn it in one epoch (run the 3-epoch arm before concluding); arm (i) fails to reproduce −15 or worse → our negatives are not the paper's and the comparison is void. *Cheaper variant:* 2B full-parameter fits locally (~18 GB with 8-bit Adam + checkpointing) but its positives are weak; 4B on one 80 GB card is the honest minimum.

**Cost table.**

| step | dataset | model | GPU RAM | host RAM | GPU-h | where |
|---|---|---|---|---|---|---|
| data | 6k RefCOCOg pos + certified N0 flips + 1k long flips | Qwen3.5-9B + Gemma4-12B + base for the difficulty gate | 19-24 GB | 18 GB | 3-5 | local |
| arm (i) | LoRA | 4B | 16 GB | 18 GB | 1-2 | local |
| arms (ii)-(v) + 3-ep | full-parameter, 8-bit Adam, grad ckpt | 4B | ~40 GB | — | 5 × 2-3 | cloud 1×80 GB (≈ $2-3/h) |
| 8B | arms (iii),(v) | 8B full-parameter | 2×80 GB | — | 2 × 4 | cloud 2×80 GB |
| eval | 6 checkpoints | 4B | 10 GB | 18 GB | 12-18 | local |

**Predicted outcome.** Arm (i): Rej 20-28, Dis −15 to −21 (reproduction). Arm (iii): Rej 8-15, Dis −3 to −6 — I predict the cost shrinks but does not vanish, because null-in-stream at any lr reinforces the D5 meaning of null on Limited (the IDK-token precedent: recall cost persists on the long tail `[S-B: 2412.06676]`). Kill for SFT as a method: Dis drop > 2 at every lr/fraction reaching Rej ≥ 13. Either outcome changes the plan.

**Guardrails.** All six; false-null stratified on Limited/Small (where D5 predicts the cost) and Discriminative (where the paper's cost was); parse-failure rate (should be ≈0 since the literal null is taught — report it); protocol swap: also evaluate under the RL-DESIGN `{"boxes": []}` prompt, a gain that survives the swap is not format compliance; negative type stratified (N0 short vs N0 long).

**Traps and controls.** (i) The reproduction is of the recipe class, not the exact data — say so. (ii) Absurd flips slipping past certification make L1 negatives — the difficulty gate is mandatory. (iii) Under-training at low lr masquerading as "no cost" — the 3-epoch arm. (iv) Parse-failure credit — reported.

**Disputes assumed.** (b): negative type — the SFT supplies the null in the official format directly, so if no arm reaches 13 at any cost, format was never the issue. (a): mostly untested (RefCOCOg expressions are ~8 words); the 1k long flips are the only coverage.

### Idea 6 — NAME-THE-CLAUSE reward: null earns credit only when the rollout names the clause that fails, verified against the edit pipeline's flip record

**Mechanism targeted.** D2 (forces the decision to be computed *from* clause checks; an abstain switch cannot earn the reward), the collapse pair (the null credit is conditional on a verifiable reason, so always-refuse earns ≈0: on positives there is no failing clause, on negatives the wrong clause pays little), D8 in part (the format needs a small cold start; the content is learned by RL).

**Method.** Train-time output: `{"checks": [{"clause": "...", "holds": true|false}, ...], "bbox_2d": [...] | null}` — a fixed short structured block, not free CoT (thinking costs Discriminative in every model, 8B 61.3 → 52.5 `[S-D: 2512.17495 Table 6]`). Rewards. N0/N0' items: r = 1 iff bbox null AND exactly the flipped clause is marked false (string match to the generator's `CHANGED:` line, Gemma4 judge for paraphrase); r = 0.3 if null with the wrong clause named (Abstain-R1's unverified-abstain credit `[S-B: 2604.17073]`); r = 0 if a box. P1 and C items: r = IoU hit; each clause marked false costs −0.5 (false-check penalty, the analogue of Abstain-R1's −1.0 false abstain); null → −λ_refuse. No fixed null bonus beyond the verified-reason credit; α scaling kept. Cold start: 500-item format SFT with the checks block filled from the generator's ground truth (which clause is true or false is known by construction), then GRPO 500 steps. Inference under the official protocol strips the block — **the protocol swap is built in**: train with the block, evaluate with and without it; a gain that needs the block is a CoT gain (Rex-Thinker's kind), still useful but scoped.

**Closest prior work.** (1) Abstain-R1 `[S-B: 2604.17073]`: abstain 0.3 + 0.7 if the clarification is verified, false abstain −1.0; U-Ref 9.4 → 68.1 with answerable accuracy +8.4 on a 3B. Delta: the verified reason is a specific clause checked against an edit record, in grounding. (2) RA-RFT `[S-A: 2511.23151, abstract only]`: query-correction reward in video temporal grounding ("say what would match"). Delta: image REC; the correction target is the flipped clause. (3) Rex-Thinker `[S-A: 2506.04034]`: per-candidate CoT-SFT +13.8 HumanRef rejection at +1.2 DF1, GRPO +0.9. Delta: fixed short check block, reward verifies the check content, and the swap tests whether the decision survives without the block.

**Kill experiment.** *Precondition (instrument, ≈1.5 GPU-h including the cold start):* after the 500-item format SFT, can the policy sample the block with ≥ 1 clause marked false on N0 items at pass@8 ≥ 20%? If not, the D8 wall stands and GRPO has nothing. *Headline:* GRPO with the name-the-clause reward vs RL-DESIGN's set-F1 reward, same 2k items, same steps. Stop rule: named-clause accuracy on held-out N0' ≥ 60% (chance ≈ 1/n_clauses ≈ 25%) AND GME Rejection ≥ 13 AND guardrails. Secondary, free: RL-DESIGN's k>1 acceptance test — a model that checks clauses should move OpenRef Multi without multi-target data; the set-F1 arm is the control. *Cheaper decisive variant:* precondition + 200 GRPO steps on Text-axis items only (text identity is the cheapest check for an OCR-strong model); named-clause accuracy < 60% even there → kill.

**Cost table.**

| step | dataset | model | GPU RAM | host RAM | GPU-h | where |
|---|---|---|---|---|---|---|
| data | 2k P1/N0/N0'/C with `CHANGED:` records (S1 A-arm + B-arm) | as RL-DESIGN S1 | ≤ 24 GB | 20 GB | 10-15 | local |
| format SFT | 500 items | 4B LoRA | 16 GB | 18 GB | 1 | local |
| GRPO 2 arms | 500 steps; rollouts ~3× longer than box-only | 4B + vLLM | 28-31 GB | 20 GB | 2 × 12-18 | local |
| judge | paraphrased clause names (string match first) | Gemma4-12B | 24 GB | 20 GB | 1 | local, sequential |
| eval | guardrail suite + OpenRef Multi | 4B | 10 GB | 18 GB | 3-4 per ckpt | local |
| 8B | both arms | 8B LoRA + vLLM | > 32 GB | — | 25-30 | cloud 1×80 GB |

**Predicted outcome.** Named-clause accuracy 55-70% on N0 (text side), 40-55% on N0' (image side; the FPCO ordering); GME Rejection with the block 12-22; without the block 5-12. Kill: named-clause < 60% on N0' after 500 steps; Rejection < 13 with the block.

**Guardrails.** Over-abstention canary; **false-check rate on positives** (clauses marked false on P1) is the new over-refusal channel and is reported; Discriminative ≤ 2.0 is the binding one (any block is a CoT); RefCOCOg ≤ 1.0; parse-failure rate for the block and for the bbox separately; stratified by sub-axis and N0 vs N0'; swap built in.

**Traps and controls.** (i) Text shortcut: on N0 the false clause is the one that reads oddly — the N0' held-out named-clause accuracy vs N0 is the shortcut size. (ii) Reward hacking by marking every clause false — r = 1 requires exactly the flipped clause false and the rest true. (iii) The block costs Discriminative — the swap arm tells whether it is needed at inference. (iv) A wrong `CHANGED:` record — Gemma4 zero-satisfier certification already required.

**Disputes assumed.** (a): sub-axis — predicts per-axis named-clause accuracy Text > Appearance > Component/State. (b): negative type — verification is the deficit and must be taught with a reason; format is handled by the cold start.

## 3. Seeds I merged or discarded, and why

| seed | disposition |
|---|---|
| separate presence channel (token / probe head / yes-no turn) | Idea 1, as a yes/no turn with its own adapter; the probe-head form belongs to Ideas-2 (inference/architecture) |
| probe-guided off-policy null rollouts | Idea 2 |
| advantage shaped by the item's IoU pass rate (TIAR/KARL), zero fixed bonus | **discarded as stated**: p̂ = "can the model solve this item" labels tiny-object positives as abstain-eligible, which is D5 verbatim and taxes Limited/Small; SurveyB §5 already notes the knowledge-boundary premise does not transfer to hard negatives. The salvageable form — null credit conditioned on the *positive twin's* pass rate — is Idea 3's pair reward at the group level |
| counterfactual paired rewards (CFCamo) | Idea 3, with the curriculum added because our pairs go above CFCamo's removal rung |
| self-distillation from the yes/no channel | Idea 4 |
| curriculum category-absent → text-swap → attribute-false | folded into Idea 3 (rungs A/B/C); as a standalone SFT curriculum it inherits the null-in-stream cost |
| low-lr full-parameter vs LoRA | Idea 5 as a headline, and a control arm on every other idea |
| multi-task "count the candidates" (MCC-style) | Idea 1's count-turn variant; it is what makes the k>1 acceptance test run for free |
| CoT / verified-reason | Idea 6 |

## 4. Rankings

**(i) Information gained per person-week (primary).** Person-weeks include data building; the A-arm infra is largely built `[RL-DESIGN §9.5]`, the attribute editor is not.

| rank | idea | why |
|---|---|---|
| 1 | **5 — SFT at the forgetting floor** | ≈1 person-week (A-arm flips on RefCOCOg + 5 cloud arms), no precondition, and every outcome reprices the whole portfolio: either SFT becomes the baseline to beat or the channel hypothesis (Idea 1) gains its strongest evidence |
| 2 | **4 — self-verify distillation, precondition first** | the 0.8 GPU-h precondition is the direct test of D2's "check exists, integration fails" reading, which the Diagnostician flagged as suggestive-not-strong; the DPO stage (~1.5 weeks) is worth running only if it passes |
| 3 | **2 — steer-rollout GRPO, precondition first** | the 0.5 GPU-h probe settles D1 representation-vs-readout, the biggest open diagnostic question; I expect it to fail (~55%), in which case the RL stage (2-3 weeks) never runs and the whole probe/steer line is closed cheaply |
| 4 | **1 — presence turn** | ~2 weeks (3k mined negatives + 2 GPU-h training); the cleanest headline kill in the set because the positive cost is measured as a single number (false-no rate) with the box policy held at base; expected to pass on rejection, so its information is in the cost, not the gain |
| 5 | **3 — flip-pair curriculum** | stage A on the existing pilot pairs is ~1 week and the rung-transfer number (N0' pass@8 after removal training) is new; stage C needs the ask-first editor |
| 6 | **6 — name-the-clause** | 3-4 weeks (full S1 data, 3× longer rollouts, judge); highest quality if it passes, lowest information per week |

**(ii) Expected paper quality if every kill passes.**

| rank | idea | claim it would support |
|---|---|---|
| 1 | 6 | rejection as verified clause checking: an abstention that must name its reason, with the k>1 transfer as the mechanism-level acceptance test |
| 2 | 1 | the SAM 3 recipe inside an MLLM: hard-negative rejection at a measured, bounded positive cost, the number no MLLM paper has |
| 3 | 3 | a flip reward plus a rung curriculum that climbs from removals to attribute edits without a cold start — the first rung-transfer measurement |
| 4 | 2 | the first REC absence probe, and RL that amplifies an internal signal instead of injecting a string |
| 5 | 4 | self-labelled hard negatives with no external verifier; harder to separate from MCC and R-Tuning in a reviewer's eyes |
| 6 | 5 | a repricing result: one optimizer change removes (or does not remove) the negative-SFT tax — solid, not a headline |

The two rankings invert, which is the usual shape: run the top of (i) first, and let (ii) decide what gets the S1 data budget.

## 5. Disputes and what I could not verify

- CFCamo: SurveyB's abstract read gives "PA 80.0-90.8, S_α +3.7 pp"; my html read gives PA 87.7-88.2 and S_α 0.844 → 0.875 vs Seg-R1-7B, and a base abstention rate of 88.7% on counterfactuals before training (outside the required schema). The reward equation and γ=2 are from the html `[VERIFIED: 2606.11231 html]`; the two S_α numbers are not the same comparison and I could not reconcile them.
- ICPO mechanism and numbers are from a secondary summary page `[LIKELY: 2510.26519]`; the ACL PDF fetch returned binary.
- Every survey-relayed number is marked `[S-x]`; I did not re-read those primaries.
- Dispute with RL-DESIGN §3.3 mechanism one: forced-None replacement in the rollout buffer is off-policy SFT under rule 3 and gives the policy no item-conditional information; Ideas 2 and 3 are the two alternatives, and the three-arm comparison in Idea 2 is the test.
- Dispute with the seed list: pass-rate advantage shaping (TIAR/KARL) applied to grounding rewards abstention on items the model cannot localise, which is D5; it should not be run in its literal form.
- Dispute (a) length vs sub-axis: I side with the Diagnostician on the evidence (ρ=0.06 within non-Text, n=87), and Ideas 2, 4, 6 each carry a per-sub-axis side-prediction that would falsify it.

## 6. Report to lead

File: `D:\Dev\ArcNova\auto-research\VLM-grounding\team\rejection\ideas-1-objective.md` — six ideas, each with mechanism, kill, cost, guardrails, traps; two rankings; the seed list dispositions.

1. Six ideas that go beyond RL-DESIGN/RC-GRPO: (1) PRESENCE-TURN — yes/no turn with its own adapter, coordinate policy stays base weights; (2) STEER-ROLLOUT GRPO — null rollouts from a probe-steered policy replace forced None; (3) FLIP-PAIR GRPO — CFCamo pair reward (verified: R = Hit⁺ − Box⁻ + 2[Null⁻ − Null⁺] + Hit⁺·Null⁻) with a removal → attribute-edit curriculum, no fixed null bonus; (4) SELF-VERIFY DISTILLATION — the model's own per-clause yes/no labels its grounding turn via DPO, no external verifier; (5) SFT AT THE FORGETTING FLOOR — the GroundingME 2:1 recipe at lr 1e-6 full-param and 10-20% negatives; (6) NAME-THE-CLAUSE — null pays only if the rollout names the flipped clause (Abstain-R1 verified-reason reward).
2. Ranking by information per person-week: 5 > 4-precondition > 2-precondition > 1 > 3 > 6. By paper quality if all kills pass: 6 > 1 > 3 > 2 > 4 > 5. The rankings invert; run (i) first.
3. Four preconditions cost ≈ 2 GPU-h total and gate four ideas: yes/no AUROC on GME (Idea 1, probe 1), hidden-state probe (Idea 2, probe 8), per-clause verification (Idea 4, probe 2), null pass@8 by rung (Idea 3, probes 4+6). Probes 8 and 2 are also the direct tests of D1 (representation vs readout) and D2 (check exists vs integration) and are worth running for the diagnosis alone.
4. Idea 5 is the cheapest repricing experiment in the set (≈1 person-week, ~10 cloud GPU-h): if null-in-stream SFT at lr 1e-6 / 20% negatives reaches Rejection ≥ 13 with Dis ≤ 2 drop, every RL idea has a new baseline; if the cost persists at every lr, the channel hypothesis (Idea 1) is strengthened.
5. Discarded seed, with reason: TIAR/KARL pass-rate advantage shaping in its literal form rewards abstention on items the model cannot localise — that is D5 (null = "cannot see") and would tax Limited/Small. Its salvageable form is the pair-conditioned credit of Idea 3.
6. Dispute with RL-DESIGN mechanism one: forced-None replacement is off-policy SFT under CAP-MECH rule 3 and carries no item-conditional information; Idea 2's three-arm comparison (steered / forced / none) is the test.
7. Predictions on record: Idea 2's probe fails (AUROC < 0.70) with ~55%; Idea 3's rung transfer (N0' pass@8 ≥ 10% after removal training) is 50/50; Idea 1 passes on rejection but the information is in the false-no rate on Limited/Small; Idea 5's cost shrinks but does not vanish.
8. Every idea states its dispute assumptions: (a) I side with the Diagnostician (sub-axis, not length) and Ideas 2/4/6 carry per-sub-axis side-predictions that would falsify it; (b) Ideas 1/2/4 assume a channel/readout component, Ideas 3/5/6 assume negative type is binding.
Could not verify: ICPO details (secondary summary only); CFCamo's S_α "+3.7" vs the html's +0.031 (different baselines, unreconciled); all survey-relayed numbers are marked `[S-x]` and not re-read.
