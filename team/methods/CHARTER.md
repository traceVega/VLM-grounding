# Methods for hard samples — team charter (2026-09-16)

Lead: `main`. Workspace: `D:\Dev\ArcNova\auto-research\VLM-grounding\team\methods\`. Today is 2026-09-16; your training data may end before mid-2026, so **use WebSearch / WebFetch for 2025-2026 work** and mark memory-only claims `[LIKELY]`. First action: `ToolSearch` with query `select:WebSearch,WebFetch,SendMessage`. Write files with the Write tool. No GPU jobs.

## The problem (in the user's words, do not re-litigate)

The model fails on **very hard referring-expression samples**: long descriptions (median 54 words, 5-7 details), several same-category candidates in the image, and either exactly one candidate satisfies every detail (positive) or one detail is false for every candidate (negative, answer = "none"). On GroundingME (2512.17495) the base Qwen3-VL-8B gets 41% of positives (338/804 boxes land on a wrong same-category object with full confidence) and 0/201 negatives. The user has looked at the failures: they are genuinely hard; **both the type of detail and the number of details matter**; the question of whether the model can judge a single detail in isolation is closed and must not be re-proposed. The base model's pass@k on these items is about zero (sum of p(null) over the 201 negatives is 3.4e-4; wrong-object positives are answered with p(box) > 0.999), so plain on-policy RL has nothing to reinforce on exactly the items we care about.

**The question: what learning method raises a VLM's capability on especially hard multi-constraint samples?** Survey the newest work on how others attack hard samples, say what each solved and with what evidence, say what is unsolved, and find where we can innovate or improve on an existing method.

## Constraints already established in this project (cite, do not re-derive)

- Off-target forgetting scales with the training mix's format-entropy loss times update magnitude, not with SFT-vs-RL or KL; low learning rate and format-diverse replay are the cheap defenses (`notes/CAPABILITY-MECHANISMS-2026-09-06.md` §4).
- Rejection scorers have artifacts (GroundingME counts unparseable output as correct rejection; zero-box metrics reward over-refusal); every rejection number needs a false-null rate on positives; GroundingME negatives are text-separable at AUROC 0.92 so a gray-image control is mandatory (`notes/REJECTION-DEEP-DIVE-2026-09-16.md` §2.5, §3).
- Negatives that the current model already rejects teach nothing (SAM 3 data engine); LLM-flipped negatives leak through text (SugarCrepe); in-distribution negatives give in-domain 97 and OOD 28 (GroundingME 2:1 SFT) with a 20-point positive collapse; RC-GRPO (2608.04698) is the only rejection RL recipe with an over-refusal penalty and was never run on GroundingME-hard negatives (`team/rejection/survey-A-rec-rejection.md` §3-4).
- Thinking mode on GroundingME: +23 Spatial for 32B/A22B but Discriminative drops in every model (8B 61.3 → 52.5); Rex-Thinker's per-candidate CoT gave +13.8 rejection at +1.2 DF1 on HumanRef; GRPO on top added <1 (`survey-A` §4, `survey-C` §1.2).
- Hardware: local RTX 5090 32 GB (23 GB host in WSL), RunPod for anything bigger; data engine exists (OpenImages instance bank with SAM 3 instances, LaMa removals, cross-family judges Qwen3.5-9B / Molmo2-8B / Gemma4-12B, lineage rules) — see `notes/RL-DESIGN-CANDIDATE-VERIFICATION.md` §9.

## Roster

- `M1` RL algorithms for items the policy cannot solve (LLM and VLM): off-policy guidance / hints / prefix / expert-anchored rollouts, dynamic sampling, difficulty-aware weighting and curricula, pass@k-oriented objectives, exploration, on-policy distillation → `m1-rl-hard-items.md`
- `M2` grounding-specific methods for hard referring expressions: reasoning/thinking grounders, per-candidate verification, zoom / thinking-with-images, structured decomposition, reward shaping for grounding → `m2-grounding-methods.md`
- `M3` data-side methods: hard-sample mining and adversarial data engines, self-play / speaker-listener, LLM synthesis with verification, counterfactual images, error-driven and self-improvement loops, compositional hard negatives → `m3-data-methods.md`
- `M4` verification and test-time compute: process / outcome reward models for grounding, generative verifiers, best-of-N and search, tool use (zoom, OCR, detector), multi-turn re-look, training the verifier into the policy → `m4-verification-tts.md`
- `Innovator` (after M1-M4): unsolved-problem map and 4-6 concrete method proposals with the three-closest-papers-and-delta test → `innovator.md`
- `Skeptic` (after Innovator): adversarial critique → `skeptic.md`
- Lead synthesis → `notes/HARD-SAMPLE-METHODS-2026-09-16.md`

## Per-method entry (mandatory fields)

Method | id | date | what problem it solves (one sentence in plain words) | mechanism (one paragraph) | evidence: base model, benchmark, before → after, and whether the gain was on items the base could not solve at all (pass@k ≈ 0) or on items it solved sometimes | cost on other capabilities if reported | what it does **not** solve | evidence label `[VERIFIED: id]` / `[LIKELY]` / `[SPECULATION]`.

Analytical sections each survey must end with: (1) which methods have evidence of moving capability on **pass@k ≈ 0** items versus only sharpening pass@1 on items already sometimes solved — this distinction is the whole question; (2) the three most transferable ideas for our setting and the concrete obstacle for each; (3) what nobody has done. Final report to `main` via SendMessage, at most 25 lines.
