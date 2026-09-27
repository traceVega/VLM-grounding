# M3 — Data-side methods for hard samples: mining engines, self-play, verified synthesis, counterfactual images, hard-negative preference, error-driven loops, curricula

Author: `M3`. Date: 2026-09-16. Inputs: `team/methods/CHARTER.md`; `team/rejection/ideas-3-data-evaluation.md` §0/§I3-1; `team/rejection/skeptic-survey.md` §5; `team/rejection/survey-A-rec-rejection.md` §3-4; `team/rejection/survey-C-adjacent-domains.md` §1.3; `team/rejection/survey-D-frontier-practice.md` §D.2 rows; `team/ideas/IDEA-13-referring-games.md`; `notes/RL-DESIGN-CANDIDATE-VERIFICATION.md` §9. 51 WebFetch reads of primary arXiv text this session; the shared WebSearch budget ran out after my 22nd query (200/200 for the session), so the last four gap checks were run through arXiv's own search page and are marked as such.

Labels: `[VERIFIED: id]` = primary arXiv HTML/PDF read this session or an earlier verified team extraction (named); `[V-abs: id]` = abstract page only; `[V-2nd: source]` = secondary page (no primary access); `[LIKELY]` = memory; `[P]` = project measurement; `[SPECULATION]`.

**What this file does not repeat.** SAM 3's presence head and §D.4 acceptance rule (`survey-C` §1.3), the L0-L4 negative ladder and the table of every method that moved an L2+ rejection number (`survey-A` §3-4), SugarCrepe's blind-model finding and CounterCurate / VisMin / NaturalBench abstracts (`skeptic-survey` §5), the PAM design (`ideas-3` §I3-1), and IDEA-13's full design and kill experiment. Those are cited, not re-derived; every entry below adds the two fields the charter asks for that the earlier files did not carry: **harder held-out than built on?** and **cost on positives?**

---

## 0. The question through the data lens: three routes, and which ones can touch a pass@k ≈ 0 item

Our items have p(correct) ≈ 0 under the base policy (Σ p(null) over 201 negatives = 3.4e-4; wrong-object positives at p(box) > 0.999 `[P: charter]`). A data method can move such an item only through one of three routes:

- **R1 — label supply.** Put the verified answer in the target (SFT / off-policy distillation). The model never has to sample the answer. Every "0 → 28-58" number in the literature is R1 (`survey-A` §4).
- **R2 — pairwise contrast.** Train on a comparison between two things the model already scores (image vs edited image, expression vs flipped expression, chosen vs rejected response) with a contrastive or DPO-family loss. The gradient is non-zero whenever the two scores differ, whether or not the model could ever sample the right answer. This is the route the compositional-negative literature lives on, and the reason its results do not directly answer our question: in a two-way choice, chance is 50 %, so no item there is ever pass@k ≈ 0 in our sense.
- **R3 — conditional sampling.** Change the sampling distribution so success becomes sampleable (hint, failed-attempt-in-context revision, decomposition into sub-items with non-zero pass rate, curriculum). M1 covers the RL side; the data-side instances are the Group Revision paper (§1.3) and constraint-count curricula (§7).

Self-play (§2) and STaR-style rejection sampling (§6) are, by construction, on-policy: they need pass@k > 0 and their difficulty rewards explicitly target the *middle* of the pass-rate range. That is the single most important fact in this survey for the charter's question, and it is documented in the papers' own reward formulas.

A second fact shapes every "did it generalize" field: with one exception (GroundingME's own 2:1 experiment), no data method in this slice was evaluated on a distribution harder than the one it was built on. In-distribution gains of +30 to +50 and held-out gains of +2 to +7 are the recurring pair.

---

## 1. Adversarial / hard-sample mining engines (against the current model)

### 1.1 SAM 3 data engine, hard-negative mining
- **id / date:** 2511.16719, 2025-11. **Problem:** open-vocabulary detector says "present" for confusable concepts.
- **Mechanism:** ontology siblings + Llama-4 proposals; a candidate negative is kept only if the *current* SAM 3 predicts masks for it that overlap the positive NP's masks; single human label per kept negative; ~30 negatives/image; a global presence token trained with BCE, per-query loss masked on negatives (`survey-C` §1.3 `[VERIFIED: 2511.16719 §D.4, §C.2]`).
- **Evidence:** SA-Co/Gold, 0 / 5 / 15 / 30 negatives per image: cgF1 28.3 / 39.4 / 41.8 / 43.0; IL_MCC 0.44 / 0.62 / 0.67 / 0.68; **pmF1 62.4 / 62.9 / 62.4 / 62.8** `[VERIFIED: 2511.16719 ablation table, HTML read this session; the team's earlier read numbered it Table 9b, the HTML numbers it Table 8]`. The head alone is +1.5 cgF1. Items: by the acceptance rule these are exactly the concept-negatives the current model *accepts*, i.e. pass@k ≈ 0 for "reject" before mining; the target is a separate binary token, so the gradient is trivially non-zero (R1 on a side head, not on the coordinate stream).
- **Harder held-out than built on?** No. SA-Co/Gold is the engine's own distribution; no external hard-negative benchmark (OVDEval, D3-ABS) is reported for the negatives ablation.
- **Cost on positives:** none measurable — pmF1 flat within 0.5 across 0→30 negatives. This is the cleanest published "hard negatives at zero positive cost", and it comes with a separate presence decision and masked positive loss, not a shared output stream.
- **Does not solve:** clause-level falsity on a present object (negatives are noun phrases); nothing about 5-7-detail expressions; detector, not MLLM.
- `[VERIFIED: 2511.16719]`

### 1.2 Motto — online top-K hard negatives in a grounding MLLM
- **id / date:** 2607.24407, 2026-07. **Problem:** free-form grounding MLLM boxes on absent targets (PR-Bench Rejection) and loses precision on dense scenes.
- **Mechanism:** spatially-grounded thought tokens with a detector-style classification loss: focal loss on all matched positive queries plus on the **top-K (K = 20) highest-scoring unmatched queries of the model's own current predictions**, recomputed every step (online, unverified, no human in the loop). `[VERIFIED: 2607.24407 §3.3 Eq. 12, HTML]`. Appendix D.2: K = 20 optimal; "an excessively large K leads to performance drops as easy negatives dominate the training process" `[VERIFIED via survey-D pypdf read of D.2; my two HTML fetches could not locate the appendix]`.
- **Evidence:** base Qwen3-VL-2B. PR-Bench mAcc 71.7, **N-Acc 46.9** (Qwen3-VL base 16.2 `[VERIFIED via survey-D Table 1/11]`), Ref-L4 92.5, HumanRef DF1 83.0 / Rejection 53.4, gRefCOCO N-acc 58.0. (My HTML summariser reported N-Acc 71.7; that is the mAcc column misread — survey D's pypdf extraction is the one to cite.) Items: online top-K by score is by definition the negatives the current model scores highest, i.e. the ones it accepts; no pass@k statistics reported.
- **Harder held-out than built on?** PR-Bench is the authors' own benchmark; HumanRef / gRefCOCO are easier rungs (L1-L2). Not run on GroundingME.
- **Cost on positives:** none — Ref-L4 and PR-Bench mAcc both above base (`survey-D` §D.2). Again a separate classification loss, not the token stream.
- **Does not solve:** L4 negatives (PR-Bench N-Acc 46.9 is the best 2026 number on its rung, and PR-Bench Reject ≤ 47.5 for every model `[P: survey-A ladder]`); negatives are unverified proposals, so false negatives (a "hard negative" query that actually is the target) are trained as negatives — the ARHN finding (§1.4) says that fraction is not small.
- `[VERIFIED: 2607.24407 + survey-D]`

### 1.3 Group Revision — "From Failure to Feedback" (object-level grounding)
- **id / date:** CVPR 2026 (Liu et al.); no arXiv id found. **Problem:** in GRPO on grounding, hard cases give all-zero groups and no gradient.
- **Mechanism:** sample a first response o¹; then sample G *revised* responses under a revision prompt that includes the failed attempt; consolidate with Hungarian matching to ground truth to compute an alignment-cost potential Φ and a relative improvement Δφ = max(0, (Φ(o¹) − Φ(o²))/Φ(o¹)); add ω·Δφ to the reward (ω = 5) and scale the advantage by (1 + Δφ). Feedback source is ground truth, no verifier. `[V-2nd: en.papernotes.org CVPR2026 page; CVF PDF returned 403]`
- **Evidence:** Qwen2.5-VL-7B. ReasonSeg gIoU 56.7 → 61.1; RefCOCOg cIoU 72.6 → 74.3; Pixmo counting 75.7 → 80.0; VisionReasoner mAP@0.5 84.8 → 85.8. Ablation on ReasonSeg val: vanilla GRPO 62.5 → +revision sampling 65.0 → +consolidation 67.0. Items: the method is *about* zero-reward groups, and one figure shows a hard case going 14.7 → 74.6 IoU under revision, but no aggregate "fraction of pass@k = 0 items recovered" is reported.
- **Harder held-out than built on?** No OOD set. RefCOCO+ testB slightly below baseline (redundant boxes).
- **Cost on positives:** slight drop on RefCOCO+ testB; sampling cost doubles.
- **Does not solve:** requires GT labels for Φ; the revision prompt only helps if p(correct | failed attempt in context) > 0, which is not measured; nothing on rejection.
- **Why it is in a data survey:** it is the only grounding paper that turns *the model's own failure* into a training input at the sample level. Relevant to M1's hint/prefix line as well.
- `[V-2nd]`

### 1.4 Retrieval-side mining against the current model: the false-negative correction
- **NV-Retriever positive-aware mining** (2407.15831, 2024-07): drop mined negatives whose score exceeds a fraction of the positive's score, i.e. remove probable false negatives; MTEB retrieval 60.9, first at publication `[V-abs]`.
- **ARHN — answer-centric relabeling of hard negatives** (2604.11092, 2026-04): Qwen3-32B extracts answer snippets from each mined negative and reranks; of **10 mined hard negatives per query, 1.6 are relabeled positive and 2.2 filtered** (~38 % not clean). E5-base BEIR nDCG@10 0.508 → 0.521 in-domain, **0.425 → 0.446 on 7 OOD datasets** — the OOD gain (+2.1) exceeds the in-domain gain (+1.3). Human-LLM κ = 0.373 on the relabeling. `[VERIFIED: 2604.11092 HTML]`
- **Debiased negative mining for OOD detection with VLMs** (2605.23797, 2026-05): corrects the sampling bias of mined negative labels by approximating the negative-label distribution via Monte-Carlo over ID labels and wild corpus; SOTA on OOD detection setups `[V-abs]`.
- **Lesson for us:** a negative mined *against* the current model is, a third of the time, not a negative. Verification is not optional, and the ARHN numbers say the payoff of cleaning is larger out of distribution than in. This is the retrieval-side twin of `survey-C`'s "adversarially filtered against the previous SAM 3" plus SAM 3's single human label per negative.

### 1.5 FineGen — VLM multi-agent attribute hard negatives for fine-grained OVD
- **id / date:** 2606.07645, 2026-06. **Problem:** CLIP-style detectors ignore attributes (FG-OVD Hard split).
- **Mechanism:** generation → verification → correction agents on the Qwen3.5-Plus API; a perturbation function swaps one attribute for a decoy; dual gate (VLM confirms the perturbed attribute is false for the image; sentence stays plausible); 9,997 ImageNet images, 1 positive + ~10 negatives each (99,970 negatives), 96.7 % attribute validity, ~$100 total. Fine-tunes CLIP ViT-B/16 (SubCLIP). Negatives are drawn from attribute vocabularies, **not** against the target model's errors. `[VERIFIED: 2606.07645 HTML]`
- **Evidence:** FG-OVD Hard 20.1 → 34.5 (+14.4), Medium 43.1 → 51.4, Easy 45.6 → 57.6, Trivial 75.8 → 76.1. Items: Hard split is where base is near chance in a multi-way choice; not pass@k ≈ 0 in our sense (R2).
- **Harder held-out than built on?** Images differ (ImageNet train vs FG-OVD), attribute vocabulary is the benchmark's; call it partial.
- **Cost on positives:** not reported (no COCO/LVIS AP).
- **Does not solve:** multi-attribute conjunctions; MLLM outputs.

### 1.6 D-Negation + Grouped Opposition-Based Learning ("Mastering Negation")
- **id / date:** 2603.12606, 2026-03. **Problem:** grounding detectors ignore negation and qualifiers.
- **Mechanism:** COCO single-annotated objects (13,893 images, 139,980 texts); GPT-4V writes, per object and per attribute (color / position / state), four labels — true/false affirmative, true/false negated — 12 texts per object; no human verification stated. Two losses (positive-negation constraint on normalised similarity; text-semantic-opposite L2 push) on MM-GDINO-T and APE, < 10 % parameters, one epoch. `[VERIFIED: 2603.12606 HTML]`
- **Evidence:** D³ absence GDINO 15.9 → 20.9, APE-C 27.3 → 33.0 (full 27.8 → 32.5); D-Negation test APE-D 78.9 → 84.1. Table III: fine-tuning on more *positive* data (Flickr30k-only) did not help negation — "simply increasing the volume of training data does not necessarily enhance the model's ability to handle negative semantics".
- **Harder held-out than built on?** D³ is external and harder; yes, modestly (+5 mAP).
- **Cost on positives:** RefCOCO val −1.5 @1, testA +1.1; the authors call the loss "structured regularization" that "may slightly affect performance in purely affirmative scenarios". A small positive cost, honestly reported.
- **Does not solve:** negatives are not policy-mined; single-attribute texts.

### 1.7 Non-examples and attack-only work
- **Jedi** (2505.13227): 2.67M unmined instruction↔unrelated-screenshot negatives → refusal 7.4 = untrained OS-Atlas; no positive cost, no gain `[VERIFIED via survey-D]`. The control case for "unmined negatives at scale do nothing".
- **REVELIO** (2605.12674, 2026-05): discovers interpretable failure modes (concept compositions) of a VLM by diversity-aware beam search / GP-Thompson sampling; diagnosis only, no data generation from the found modes `[V-abs]`.
- **PEAT** (2506.16157): embedding-guided bidirectional adversarial attack on RES models; attack only, no adversarial training `[V-abs]`. Nobody has closed the loop "attack the grounder with adversarial *expressions*, verify them, train on them".

---

## 2. Self-play and speaker-listener games

### 2.1 VisPlay
- **id / date:** 2511.15661, CVPR 2026. **Problem:** RL for VLM reasoning needs labels.
- **Mechanism:** one base VLM split into an image-conditioned Questioner and a Reasoner, both trained with GRPO. Silver label = majority vote of m Reasoner samples; confidence c = vote share. **Questioner reward r_unc = 1 − |2c − 1|, maximal at c = 0.5**, minus a diversity penalty on cluster size. `[VERIFIED: 2511.15661v2 HTML]`
- **Evidence:** Qwen2.5-VL-3B average 30.6 → 47.3 over three iterations (MMMU 20.0 → 37.1; MathVerse 26.1 → 35.2); 7B 40.4 → 48.6; MiMo-VL-7B 43.6 → 45.7 **with drops** (MM-Vet 59.2 → 56.9, RealWorldQA 78.2 → 71.5). Pseudo-label accuracy falls 72 % → 61 % as questions get harder. Matches supervised GRPO (47.3 vs 47.1) with no labels. The HallusionBench 32.8 → 91.8 row is as extracted and looks like a metric change; treat with care.
- **Items:** by the reward formula, the questioner is *paid to avoid* c ≈ 0 items (r_unc → 0 there), and on such items the majority-vote label is wrong anyway. This loop cannot target pass@k ≈ 0 items; it is designed not to.
- **Harder held-out than built on?** Standard public benchmarks, no OOD split. **Cost:** the MiMo-VL drops above.
- **Does not solve:** anything with an objective geometric label; rejection.

### 2.2 R-Zero (LLM ancestor) and Vision-Zero / Active Zero / self-judge loops
- **R-Zero** (2508.05004): Challenger rewarded "for proposing tasks near the edge of the Solver capability"; Qwen3-4B-Base +6.49 math, +7.54 general `[V-abs]`. Same middle-difficulty targeting as VisPlay.
- **Vision-Zero** (2509.25541): "Who-is-the-spy" games on image pairs where the spy sees a blank image; Iterative-SPO alternates self-play and RLVR; Qwen2.5-VL-7B MathVista 68.2 → 73.1, MathVision 25.4 → 28.9, LogicVista 47.2 → 51.2, MMVP 76.8 → 79.5; pure self-play plateaus (~2 points below Iterative-SPO on LogicVista) `[VERIFIED: 2509.25541 HTML]`. No localization, no held-out claim beyond public benchmarks, no drops reported.
- **Active Zero** (2602.11241): adds a Searcher that retrieves images "aligned with model capabilities" (frontier targeting again); Qwen2.5-VL-7B reasoning 54.0 (+5.7 %), general 59.8 (+3.9 %) `[V-abs]`.
- **Unsupervised self-evolution with self-judge** (2603.21289): self-consistency prior + bounded judge modulation in GRPO; math benchmarks only `[V-abs]`.
- **Common structure:** every 2025-26 VLM self-play loop labels by majority vote or self-consistency and rewards the proposer for ~50 % solver accuracy. None has a geometric win condition; none reports what happened on items the solver started at 0 on.

### 2.3 Reinforced Reference Game (RRG) — the one MLLM speaker-listener game with hard negatives
- **id / date:** 2606.28845, ECCV 2026. **Problem:** personalised MLLMs describe a concept non-discriminatively.
- **Mechanism:** one Qwen2-VL-7B plays speaker (LoRA r = 64, trained with GRPO) and listener (**frozen**); listener answers yes/no per candidate; hard positives = other views of the same instance; hard negatives = the U = 2 CLIP-nearest images of the same class; reward = ρ_target / Σ_u ρ_u if the listener's argmax is the target, else 0. 30 PerVA concepts for the game, 269 held out. `[VERIFIED: 2606.28845 HTML]`
- **Evidence:** captioning F1 PerVA 67.3 → 86.9, MyVLM 89.3 → 95.0, Yo'LLaVA 84.6 → 88.6; recognition +1.1 to +5.4; VQA 92.5 → 95.3; generalises to unseen Yo'LLaVA concept categories. Normalised reward beats binary (95.0 vs 93.4). No hard-vs-random negative ablation.
- **Harder held-out?** Unseen concepts, yes; unseen *difficulty*, no. **Cost:** none reported.
- **Does not solve:** the listener is never trained — the beneficiary is the speaker. This is the pattern in every 2026 speaker-listener paper (also 2609.14207 below): the comprehension side is a frozen reward model.

### 2.4 Speaker trained against an estimated-gaze listener
- **id / date:** 2609.14207, 2026-09. Speaker fine-tuned with a gaze-scanpath listener as reward; expressions shrink 15.4 → 4.0 words while human referential success rises 75.2 → 80.0 %. Speaker only `[V-abs]`. Relevant as the newest evidence that listener-shaped rewards produce *shorter, more discriminative* expressions — the opposite direction from GroundingME's 54-word items — which is what a naive speaker reward would do to our data.

### 2.5 The project's own line: IDEA-13 and the 2017 ancestor
- Yu et al. speaker-listener-reinforcer (1612.09542) trained comprehension against distractor expressions `[VERIFIED via IDEA-13 Verifier]`; IDEA-13 (2026-09-02) designs the MLLM version with a frozen uniqueness verifier U (Qwen3-VL-32B), a speaker bonus for expressions the student listener fails on (pass@8 ≤ 0.5 frontier, pass@8 = 0 hard positive), a non-extremal tier, and a removal-based negative mode; kill experiment unrun (35-45 GPU-h). Against this survey: no published 2026 work rewards a speaker for *listener failure* with an independent verifier — VisPlay rewards uncertainty, RRG rewards listener success. IDEA-13's bonus term is still unoccupied ground (§10, item 2).

---

## 3. LLM synthesis with verification

### 3.1 RefBench-PRO / RefObjects-200k / Ref-R1
- **id / date:** 2512.06276, 2025-12. **Mechanism:** Qwen2.5-VL-72B writes a structured property dictionary per image → Grounding DINO boxes → *region correction* (VLM checklist; keep only fully consistent objects) → rule-based task assignment (Attribute / Position / Interaction / Relation / Commonsense / Reject) → expression generation → two-stage verification (semantic consistency; uniqueness). Reject expressions "contain one or more attributes that do not match any object". Train: 203,985 phrases on 28,541 images (154,839 positive, **49,146 reject**); eval: 6,000 phrases. `[VERIFIED: 2512.06276 HTML]`
- **Evidence:** Qwen2.5-VL-7B Acc_p 57.6 → 64.1 (SFT-CoT) → 69.4 (DyIoU-GRPO); **Reject 3.1 → 58.2**, with RL adding ~0 on Reject (`survey-A` §4); RefCOCO/+/g 88.8 (+2.1); Ref-L4 85.7 (+4.4). Items: the 0 → 58 on Reject is R1 (SFT on 49K verified rejections).
- **Harder held-out than built on?** No — RefBench-PRO is the pipeline's own distribution; not run on GroundingME. **Cost on positives:** none (all positive metrics up).
- **Does not solve:** no policy-hardness gate on the reject set (the writer, verifier and target policy share the Qwen lineage); rung L2 (`survey-A`).

### 3.2 GroundingSuite (GSSculpt)
- **id / date:** 2503.10596, ICCV 2025. **Mechanism:** InternVL2.5 caption → Florence-2 phrase grounding → SAM2 mask → InternVL2.5 writes a disambiguated expression (spatial relations, distinctive features, context; ~16 words) → **EVF-SAM listener filter at IoU ≥ 0.5** against the source mask. 9.56M pairs on 2M SA-1B images. `[VERIFIED: 2503.10596 HTML; V-198]`
- **Evidence:** EVF-SAM trained on it: gRefCOCO 63.5 → 66.4, RefCOCO 60.7 → 63.5, GSEval 62.6 → 77.3; LISA-7B gRefCOCO 29.5 → 31.3, RefCOCOm 29.6 → 37.6.
- **Harder held-out?** GSEval is in-house (+14.7); external gains are +2 to +3; RefCOCOm +8 is the fine-grained-mask metric, not a harder language distribution. **Cost:** none reported.
- **Does not solve:** no distractor control, no negatives, no policy gate; the filter is mask-consistency with the source, not uniqueness among designed distractors (IDEA-13 §1).

### 3.3 Ref-Adv — the verified hard-positive construction with its keep rate
- **id / date:** 2602.23898, ICLR 2026. **Mechanism:** COCO + OpenImages v7 val/test panoptic images with ≥ 3 same-category instances; GPT-4o first lists the discriminators separating a hard-distractor pair, then composes a *minimal* expression from them (direct prompting "produced overspecified descriptions with many redundant descriptors"); three annotators ground blind, then confirm correctness and hard-distractor presence; unanimous only. **LLM-authored keep rate 18.7 %**, ~$0.037 per kept expression; 5,000 items / 2,833 images (public Ref-Adv-s 1,142); 21.25 % use explicit negation. `[VERIFIED: 2602.23898 HTML]`
- **Evidence (eval only, no training):** Qwen2.5-VL-72B RefCOCO 92.7 → Ref-Adv 54.1 (CoT 58.3); GPT-4o + SoM 63.7; bag-of-words shuffle costs 72B 16.8 points.
- **Harder held-out / cost:** n/a (no training). **Why it matters here:** the 18.7 % keep rate is the honest price of *verified* LLM-written hard expressions on multi-instance images — 5× overgeneration before any policy gate. It is also the anti-curriculum: minimal sufficiency, no redundant detail, versus GroundingME's 5-7 details where the falsified one is often decorative (`ideas-3` §0 dispute).

### 3.4 HumanRef-CoT (Rex-Thinker)
- **id / date:** 2506.04034, 2025-06. **Mechanism:** GPT-4o with Set-of-Marks over *all* person boxes writes plan → per-candidate action (check each box against the sub-goal) → summary; **keep only where GPT-4o's final prediction matches the GT label**; 90,824 CoTs; rejection cases are the no-match subset. SFT on Qwen2.5-VL-7B, then GRPO with F1 + format reward. `[VERIFIED: 2506.04034 HTML]`
- **Evidence:** HumanRef DF1 82.3 / Rejection 67.3 after SFT → 83.5 / 68.2 after GRPO; plain-vs-CoT Rejection 53.5 → 68.2 at DF1 +1.2 (`survey-A`); RefCOCOg zero-shot 80.3 → 83.3.
- **Harder held-out than built on?** RefCOCOg is *easier*; "maintains structured CoT on novel categories" is qualitative. **Cost on positives:** DF1 +1.2 (none).
- **Does not solve:** the verification is agreement with the GT *box*, not per-detail correctness of the reasoning; person-only; L2 negatives.

### 3.5 16M-expression engine (no verification) — 2407.14563
- **Mechanism:** PaLI-3 on the cropped target ("ignore the background"), top-5 by confidence, no post-processing; rule-based spatial relations; attribute prompts top-3; 16.2M expressions / 1.1M objects / 512K images (COCO + Objects365) `[VERIFIED: 2407.14563 HTML; V-197]`.
- **Evidence:** zero-shot REC RefCOCO 63.4 vs prior 56.0, **but RefCOCO+ 53.9 vs 55.3 and RefCOCOg 63.3 vs 67.5** — the unverified engine wins on the easy split and loses on the attribute-only and long-expression splits. RES oIoU 41.2 vs 36.1.
- **Harder held-out?** Human RefCOCO family, yes, and it loses on the harder members. **Cost:** n/a (trained from scratch).
- **Reading:** unverified volume buys the easy rung; the two harder rungs are exactly where verification (and distractor awareness) would have to enter.

### 3.6 FINER and FINER-tuning (negatives filtered against a *strong* model)
- **id / date:** 2603.17662, 2026-03. **Mechanism:** from scene graphs (CompreCap human graphs; DOCCI graphs from dense captions) create four plausible negative variants per object / attribute / relation with Qwen3-14B or Gemini-2.0-Flash; **filter with Qwen2.5-VL-72B as discriminator — if it cannot separate positive from negative, regenerate iteratively** (adversarial to a strong model, not to the trained policy). Seven granularity levels. FINER-tuning: DPO with (accepted, rejected) answers for both positive and negative queries, built on Pixmo captions with Phi-4-14B — pipeline-disjoint from the benchmark, COCO and DOCCI-train excluded. `[VERIFIED: 2603.17662 HTML]`
- **Evidence:** base accuracy falls ~80 % → ~20 % from level 1 to levels 5-7 (CompreCap), ~58 % → ~15 % (DOCCI) — constraint count is the difficulty axis. FINER-tuning: InternVL3.5-14B +24.2 Multi-rel; LLaVA-1.6 +23.1 Multi-obj / +25.4 Multi-attr; **held-out** DASH +2.0 (Qwen2.5-VL) / +6.2 (InternVL3.5-8B), AMBER +6.9, MMHal HR to 14 %.
- **Harder held-out than built on?** DASH / AMBER are different distributions but not harder; the gain on FINER's own hard levels is in-pipeline (different writer LLM, same edit grammar). **Cost on positives:** "no alignment tax" — six general benchmarks +0.3 to +1.9. R2 throughout (yes/no queries).
- **Does not solve:** grounding output; policy-adversarial selection; negatives the *trained* model already rejects are not removed.

### 3.7 ScenGround (scenario queries, LLM GT filter, human audit, tag curriculum)
- **id / date:** 2604.02323, 2026-04. **Mechanism:** GPT-4o writes scenario query + reasoning trace + expression + aliases + box on COCO; LLM GT filter then three-annotator audit (κ 0.94); retention 30,000 → 23,802 (SFT), 10,000 → 7,540 (RL); five difficulty tags (uniqueness, clutter, size, overlap, position). Qwen2.5-VL-7B: thought-primed SFT → IC-GRPO with a two-stage mixture (0.70/0.30/0 → 0.20/0.60/0.20 easy/medium/hard). `[VERIFIED: 2604.02323 HTML]`
- **Evidence:** ID Acc@0.5 27.4 → 60.9; **OOD 15.9 → 38.1**; curriculum vs single-stage on OOD: mIoU 38.4 vs 37.7, Acc@0.5 38.1 vs 37.0 — the curriculum is worth ~1 point; SFT-only OOD category accuracy collapses (20.8 → 12.5) while GRPO holds 21.1. Transfer RefCOCO+ val 52.5 → 70.2, RefCOCOg val 52.5 → 78.2 (from a custom-prompt base, so partly format).
- **Harder held-out?** Their OOD split, yes; same generator. **Cost:** none on RefCOCO family; the SFT arm's category-accuracy collapse is a positive-side cost of SFT that RL avoided.

---

## 4. Counterfactual / minimal-change images

### 4.1 CounterCurate
- **id / date:** 2402.13254, 2024-02. **Mechanism:** left/right by flipping; above/below by GLIGEN box-centre swap; counting by GLIGEN inpainting or plant replacement; attributes by GPT-4V caption edits → DALLE-3; 84.7 % human agreement that DALLE-3 images match the negative caption (300 checked); Flickr30k-Entities base. `[VERIFIED: 2402.13254 HTML]`
- **Evidence:** CLIP Flickr30k-Positions 50.6 → 75.9 (L/R), 52.8 → 91.5 (A/B); LLaVA-1.5 61.8 → 95.7 / 56.7 → 96.2; counting CLIP 57.5 → 68.5, LLaVA 44.9 → 50.7; **held-out SugarCrepe CLIP 79.5 → 86.2, LLaVA 89.3 → 94.2** (above GPT-4V's 92.2).
- **Harder held-out than built on?** SugarCrepe is a different (de-biased) distribution; yes for compositional text, and it survived. **Cost on positives:** COCO retrieval flat (text R@1 56.5 → 57.0). Items: base at chance on the position tasks (R2).
- **Does not solve:** no verification that the *positive* still holds after the edit beyond the 84.7 % sample; no grounding output.

### 4.2 VisMin
- **id / date:** 2407.16772, 2024-07. **Mechanism:** Mistral-47B edit instructions; SDXL inpainting (object/attribute), GLIGEN (count/spatial), LMD for synthetic sources; automatic VQA-consistency filter removes 75 %; human 4-step verification with 26 % pass on naturalness/ITM, 80 / 84 / 95 % on the later steps; 64,392 training pairs. `[VERIFIED: 2407.16772 HTML]`
- **Evidence:** CLIP avg 46.4 → 63.7 (counting 37.0 → 82.3); **Idefics2 56.0 → 86.4 with spatial 18.6 → 83.0** (below chance to well above); held-out Winoground text +6; CountBench +9 to +19 over NegCLIP / CounterCurate / SPEC; COCO retrieval up.
- **Harder held-out?** Winoground is harder and different; small positive transfer. **Cost:** **zero-shot ImageNet drops** for both NegCLIP and VisMin-CLIP, VisMin less. Yield: ~6 % of generated pairs survive all filters — the cost of clean minimal-change images.
- **Does not solve:** grounding; multi-detail; no policy gate.

### 4.3 FineCops-Ref (negative images)
- **id / date:** 2409.14750, 2024-09. **Mechanism:** text negatives by LLM replace/swap at two levels (target vs related object); **negative images by PowerPaint inpainting + horizontal flips, filtered by CLIP similarity, DIRE naturalness (0.2) and DINOv2 (best of 10)**. Test: 9,605 pos / 9,814 neg text / 8,507 neg images; **train: 163,792 pos / 80,451 neg text, no negative images**. `[VERIFIED: 2409.14750 HTML]`
- **Evidence:** MM-GDINO-T positive P@1 48.5 → 63.4; text-negative R@1 38.7 → 50.1; **image-negative R@1 43.1 → 57.0** — the image-negative gain comes from text-negative training only. CogVLM (positives only) 64.7 → 78.2, image-neg 47.2 → 54.1, RefCOCOg 89.8 → 90.8.
- **Harder held-out?** No OOD. **Cost:** none (positives up). Zero-shot MLLM on its negatives is L2-rung 5.3 N-acc (`survey-A`).
- **Does not solve:** L4; no policy gate; edited images were never used as training negatives.

### 4.4 CF-VLM
- **id / date:** 2506.17267, 2025-06. **Mechanism:** SDXL edits of one attribute or one causal relation; Qwen2-72B text counterfactuals; three losses (InfoNCE; scenario discrimination margin 0.25; fine-grained causal discrimination margin 0.30); 1:1 factual:counterfactual over CC12M/CC3M. `[VERIFIED: 2506.17267 HTML]`
- **Evidence:** CLIP ViT-B/32 VL-Checklist 74.6 → 88.4, ConMe 54.4 → 59.1; Qwen-VL-7B ARO 86.8 → 93.2, ConMe 82.6 → 87.6; POPE +0.5 to +1.2. **Harder held-out?** No Winoground / SugarCrepe / VisMin reported. **Cost:** ImageNet-1k and retrieval numbers reported without a base column; +20-25 % training compute.

### 4.5 S-VCO and the MVC set — the pairwise-image objective
- **id / date:** 2502.13928, 2025-02. **Mechanism:** 11K minimal-contrast pairs re-used from CounterCurate and FineCops-Ref, filtered DINOv2 < 0.5 (visually different) and CLIP > 0.7 (semantically close, hard); GPT-4o turns captions into conversational QA. Objective: for each image role, an attend term (prefer the matching image over no image) and a reject term (penalise the matching text under the contradictory image), summed symmetrically over both images. `[VERIFIED: 2502.13928 HTML]`
- **Evidence:** LLaVA-1.5-7B MMHal hallucination 57 → 46 %, CV-Bench 59.3 → 63.5; average gain S-VCO +12.2 vs DPO +7.5 vs mDPO +1.4 on the same pairs; LLaVA-Next-Interleave +8.6 vs +4.7 (DPO).
- **Harder held-out?** No. **Cost:** general LLaVABench 66.8 → 67.2 (LV-1.5), 74.8 → 73.9 (LV-INT).
- **Why it is the most transferable objective here:** it is the only published loss that uses a *same-expression, edited-image* pair symmetrically, which is exactly what our LaMa removals + control edits produce (§9-B).

### 4.6 OViP — negatives from the model's own failures, rendered as images
- **id / date:** 2505.15963, 2025-05. **Mechanism:** sample k = 16 responses; LLM scores vs GT; pairs where the gap exceeds max(δ, 2σ); an LLM describes the semantic difference between the positive and the negative response; **FLUX.1-dev renders the negative image from that description**; text-DPO + image-DPO (response fixed, image varied) with an image-free smoothing term; online buffer. `[VERIFIED: 2505.15963 HTML]`
- **Evidence:** LLaVA-1.5-7B MMHal 1.90 → 2.52; AMBER-gen F1 65.0 → 66.7; ObjectHal F1 72.4 → 73.5; LLaVA-Bench 57.2 → 63.1; general +0.88; online beats offline by ~4 HRI in one epoch; image-side loss essential.
- **Harder held-out?** No. **Cost:** none. **Reading:** this is the closest published "failure-driven image negative" loop; the negatives are derived from what the *current* policy got wrong, which is our PAM idea on the image side. Its weakness for us: the generated image is a whole new scene, not a minimal edit of the same scene.

### 4.7 ViPSy — policy-aligned pairs from image variants
- **id / date:** 2606.28401, 2026-06. **Mechanism:** policy self-captions → SD-3.5-L renders N_c variants → judge extracts object-level content recurring across original and variants (visual cue) → policy rollouts conditioned on the cue → judge picks preferred/dispreferred → DPO. `[VERIFIED: 2606.28401v1 HTML]`
- **Evidence:** LLaVA-1.5-7B Object HalBench response 51.3 → 4.0; AMBER hallucination 36.8 → 9.6; MMStar 46.8 → 48.1, CV-Bench 61.9 → 64.4; consistent on Qwen2.5-VL-7B, InternVL3.5-8B, mPLUG-Owl3, Kimi-VL (13.9-24.5 % hallucination reduction); judge agreement 98.4 % with Qwen3-VL-30B.
- **Harder held-out?** No. **Cost:** general benchmarks up. The lesson is "policy-distribution pairs beat intervention pairs" — the same argument OViP makes against predefined negatives.

### 4.8 Evaluation-only paired designs and older DPO image tricks
- **NaturalBench** (2410.14669): semi-automatic collection with CLIP + ChatGPT (adversarial to CLIP, not to the evaluated VLM), 10,000 human-verified items, each question paired with two images that flip the answer; SOTA VLMs 50-70 % below humans; **no fine-tuning reported** `[V-abs]`.
- **GroundBench** (2609.13308, 2026-09): factorised region/identity conditions and counterfactual re-asks for affordance grounding; GPT-5 unchanged with no vision; eval only `[V-abs]`.
- **mDPO** (2406.11839): rejected image = random crop of < 20 %; anchor keeps chosen-response reward positive; Bunny-3B MMHal 2.11 → 2.96, LLaVA-1.5-7B 2.19 → 2.39 `[VERIFIED: HTML]`. **POVID** (2402.11411): GPT-4V-injected hallucinations + diffusion-noised images; LLaVA-1.5-7B POPE 85.9 → 86.9, MMHal 2.42 → 2.69, MMBench +1.9 `[VERIFIED: HTML]`. **LPOI** (2505.21061): mask the critical object and interpolate visibility to make a ranked image list; listwise loss `[V-abs]`. None of these is a *minimal semantic* edit; crops and noise are the "easy image negative" and are the image-side twin of Jedi.
- **Project:** LaMa class-label removals give 50 % abstention, AUROC 0.964, 0/316 false abstentions on positives `[P: notes/ABSTAIN-SIGNAL.md]` — the base already separates whole-object removal; the untested regime is a clause-level edit (one detail changed on one of several same-category instances).
- **Gap check:** an arXiv-search fetch for 2025-26 papers training a grounding/REC model on edited or counterfactual images returned nothing relevant; I found none by any other route either. Absence claimed with the caveat that the search tool was exhausted.

---

## 5. Compositional hard negatives at the encoder and LLM level

### 5.1 NegCLIP (ARO) as judged by SugarCrepe
- **ids:** 2210.01936 (ARO/NegCLIP), 2306.14610 (SugarCrepe). NegCLIP adds word-swapped captions and nearest-neighbour images as in-batch negatives `[LIKELY: memory]`. SugarCrepe removes two exploitable artifacts (a commonsense-plausibility scorer, Vera, and a grammar model) by subsampling to symmetric score-gap distributions on a 100×100 grid; on the de-biased set **"none of the improvements on SugarCrepe is larger than 10 %"** against > 10 % on ARO/CREPE `[VERIFIED: 2306.14610 HTML]`. NegCLIP costs some ImageNet zero-shot accuracy `[LIKELY]`; VisMin confirms the direction (§4.2).
- **Harder held-out than built on?** SugarCrepe *is* the harder held-out, and the gain shrank. **Reading:** most of NegCLIP's headline gain was on the artifacts.
- **CE-CLIP** (2306.08832): intra-modal contrast + ranking of cross-modal hard negatives `[LIKELY: memory; not fetched]`.

### 5.2 HNC — scene-graph hard negative captions at scale
- **id:** 2605.06157 (arXiv posting 2026-05; the paper reads as the EMNLP-2023-era ITM work `[LIKELY]`). 18.1M negatives over 74,942 GQA images (≈ 242 per image), 12 perturbation types incl. AND/XOR; ambiguity and noisy-value heuristics as verification; ITM heads on VilBERT / VisualBERT / LXMERT / UNITER. `[VERIFIED: HTML]`
- **Evidence:** VilBERT HNC test 48.3 → 66.4; **zero-shot VALSE 48.8 → 53.0**; GQA 53.5 → 55.5. Authors: scene-graph negatives "are subject to linguistic and distributional biases which are difficult to combat".
- **Harder held-out?** VALSE, yes, small. **Cost:** not reported. Pre-MLLM.

### 5.3 LLM-level hard-negative preference: what exists
- FINER-tuning (§3.6), POVID / mDPO (§4.8), OViP (§4.6), ViPSy (§4.7), S-VCO (§4.5) are the DPO-family entries. **Anyprefer** (2504.19276): target model and judge as a cooperative two-player game with tools; 58K pairs; vision-language +3.66 % `[V-abs]`. **CSR** (2405.14622): CLIP-score-constrained self-reward, iterative; +7.62 % over ten benchmarks `[V-abs]`.
- **Hard-negative DPO on *box outputs*** (chosen box vs the model's own wrong same-category box, or chosen `null` vs its confident box): an arXiv-search fetch found none for 2025-26; no team file cites one. Every DPO entry above is on free-text responses. This is the missing piece between §4.5's objective and our task (§10, item 3).

---

## 6. Error-driven and self-improvement loops

### 6.1 R3V — rejection sampling that keeps the negatives
- **id / date:** 2411.00855, 2024-11. **Mechanism:** sample 3 CoTs per item with GT answers; positives → SFT; negatives feed a self-refine loss (condition on the wrong rationale, produce the right one) and a self-select loss (pick the correct answer among candidates); 4-5 iterations; GPT-distilled warm-up on 536-1,000 items. `[VERIFIED: 2411.00855 HTML]`
- **Evidence:** Qwen-VL average 48.5 (GPT-distill) → 59.3 (STaR) → 64.4 (R3V); **held-out MMMU 33.7 → 35.6, MathVista 32.7 → 35.1, VCR 45.4 → 50.2**; test-time self-select adds MMMU +2.9. Ablation: no iteration −3.7, no self-select −3.6. Only 8-70 % of *correct-answer* rationales are fully correct, and DPO on them failed — elimination-style selection beat preference ranking.
- **Items:** needs pass@3 > 0 to produce a positive; the negative-conditioned losses are the only part that touches items with no positive sample, and they are not analysed separately by pass rate. **Cost:** none reported.

### 6.2 OpenVLThinker — iterative SFT (distilled) → RL
- **id / date:** 2503.17352, NeurIPS 2025. SFT on captioned-image → text-reasoner distilled CoT, then RL, repeated; the paper's own reading is that "the base model rarely exhibits reasoning behaviors initially, but SFT effectively surfaces these latent actions and narrows the RL search space" `[V-abs + search summary]`. MathVista +3.8, EMMA +2.4, HallusionBench +1.6. **Relevance:** the explicit R1-then-RL recipe for behaviours the base cannot sample; no grounding, no held-out difficulty claim.

### 6.3 Vision-SR1 — self-reward by image-free re-answering
- **id / date:** 2508.19652, ICLR 2026. Model writes a visual description, then re-answers from the description alone; if correct, the description is rewarded (no external RM, +20 % wall-clock). Qwen2.5-VL-7B average 41.5 → 52.2 (MMMU-Pro 33.5 → 52.2; HallusionBench 51.7 → 68.9); language-shortcut rate 10.1 → 9.8 %; spatial OOD sets (MMSI-Bench, OmniSpatial) reported `[VERIFIED: HTML]`. Needs GT answers; rewards descriptions *sufficient* for the answer — for grounding that is a per-detail description, which is the verifier M4 covers.

### 6.4 Reflect-Retry-Reward (LLM) and M2Note (inference-time)
- **RRR** (2505.24726): keep only queries the model failed; reflect, retry, reward the reflection tokens if the retry succeeds; +34.7 % equation writing, +18.1 % function calling on 1.5-7B models `[V-abs]`. R3 on the LLM side; needs a binary verifier and p(success | reflection) > 0.
- **M2Note** (2607.00685, 2026-07): a supervisor writes subject/guidance notes from failures into an external memory retrieved at inference; accept-if-improves per batch; Qwen3-VL-8B MathVista 73.2 → 77.3, MMMU +1.3; $3-7 per run; transferred notebooks "can hurt" `[VERIFIED: HTML]`. Not training; relevant only as a cheap failure-to-guidance loop.

### 6.5 What every loop in §6 has in common
Each needs one of: a GT label (R3V, Group Revision, RRR, Vision-SR1), a majority vote (VisPlay), or self-consistency (2603.21289). On an item where p(correct) ≈ 0 the first exists only if we supply it (our verified engine), the second is wrong, and the third is confidently wrong (338/804 wrong boxes at p > 0.999 `[P]`). No paper in this section reports the pass@k distribution of the items on which it gained.

---

## 7. Curriculum data construction by constraint count

- **Evidence that constraint count is the axis:** FINER's seven levels, ~80 % → ~20 % (§3.6) `[VERIFIED]`; our own GroundingME finding that both detail type and count matter `[P: charter]`; Ref-Adv's bag-of-words drop of 16.8 `[VERIFIED]`.
- **ScenGround's tag curriculum** (§3.7): worth ~1 point OOD over single-stage GRPO; the difficulty tags are scene statistics (uniqueness, clutter, size, overlap, position), not clause count `[VERIFIED]`.
- **E2H Reasoner** (2506.06632): easy-to-hard with fading of easy tasks; sample-complexity bound; gains on 1.5-3B LLMs that "otherwise struggle when trained with vanilla RL alone" `[V-abs]`.
- **Rethinking Easy-to-Hard** (2603.27226, 2026-03): on synthetic arithmetic/logic post-training, SFT and RL, multiple families and schedules, "no robust advantage in difficulty-based sequencing over standard random sampling" `[V-abs]`.
- **Project:** I3-4 support-ratio ladder (clause count × flip type, load-bearing-verified) is designed, unrun.
- **Reading:** curricula are contested where the hard items are *sampleable*; where they are not (ours), the curriculum is doing R3 work — turning a 6-detail item the policy cannot solve into 2-, 3-, 4-detail items it sometimes can. No grounding paper has built such a ladder with per-clause labels, and 2603.27226 is a warning that ordering alone is not the lever; the per-clause *labels* are.

---

## 8. Analysis (1): which data methods moved pass@k ≈ 0 items, versus sharpening pass@1

| route | method (entry) | what the base could do before | gain | held-out harder than built on? | positive cost |
|---|---|---|---|---|---|
| R1 label supply (SFT) | GroundingME 2:1 RefCOCOg_rej `[VERIFIED: 2512.17495 + survey-A]` | Rejection 0 (L4) | in-domain 30.5 → 97.3; **GME 0 → 27.9** | **yes, the only such number** | GME non-rejection 38.8 → ≤ 33.0 (Disc −21, Limited −19), RefCOCOg −5; at **1:8** the same recipe gives in-domain 83.5 with RefCOCOg +2.2 — the collapse is a ratio effect, and the 1:8 row's GME-Rejection value is not printed (open) |
| R1 | Ref-R1 SFT (§3.1); RexSeek / Rex-Thinker CoT-SFT (§3.4) | Reject 3.1 / 53.5 | → 58.2 / 68.2 | no | none |
| R1 on a side head | SAM 3 hard negatives (§1.1); Motto L_cls (§1.2) | IL_MCC 0.44; N-Acc 16.2 | → 0.68; → 46.9 | no | **none** (pmF1 flat; Ref-L4 up) |
| R2 pairwise contrast | VisMin → Idefics2 spatial 18.6 → 83.0 (§4.2); CounterCurate positions ~50 → 96 (§4.1); FINER-tuning +24 (§3.6); S-VCO (§4.5); OViP / ViPSy (§4.6-7) | below/at chance in 2-way choice | large in-pipeline | **small but consistent**: SugarCrepe +5-7, Winoground +6, DASH +2-6, VALSE +4 | ImageNet drop (CLIP); none for MLLM DPO |
| R3 conditional sampling | Group Revision (§1.3) | zero-reward groups | ReasonSeg +4.4 | no | RefCOCO+ testB slight |
| on-policy self-play | VisPlay / R-Zero / Active Zero / Vision-Zero (§2.1-2.2) | reward peaks at 50 % solve rate | +2 to +17 avg | no | MiMo-VL drops |
| on-policy failure loops | R3V, RRR, Vision-SR1 (§6) | need pass@3 > 0 or GT | +5 to +16 | R3V MMMU +2 | none |

Three conclusions.

1. **Only R1 has moved a grounding rejection number from 0**, and only GroundingME's own experiment did it on a distribution harder than the training negatives — at −20 on positives, which its 1:8 row shows to be a mixing-ratio effect rather than a law (positives *rose* at 1:8; whether GME-Rejection was still non-zero there is the cheapest unanswered question in this table). Every other 0 → X is in-distribution.
2. **R2 is the only route with a consistent held-out transfer record**, but every instance is a two-way choice or a free-text response; nobody has run a pairwise objective on box outputs. The evidence that R2 "moves capability" is the below-chance-to-above-chance cases (VisMin spatial for Idefics2), which are the closest analogue to pass@k ≈ 0 the binary-task literature can offer.
3. **Self-play and STaR-style loops are designed around the middle of the pass-rate range** (r_unc = 1 − |2c − 1|; "edge of the Solver capability"; keep-if-failed-then-retry-succeeds). They cannot be the mechanism for our items unless the label comes from outside the loop — which, for grounding, it can (the verified box or the verified `none`).

Where the positive cost comes from, across the table: every recipe that puts `none` into the same autoregressive coordinate stream by SFT at a high negative ratio paid on positives (GroundingME 2:1); every recipe with a separate decision (SAM 3 presence, Motto L_cls, GSVA `[REJ]` per `survey-A`) or a pairwise objective (S-VCO, FINER-tuning) reported no cost. That aligns with the charter's format-entropy account (`CAPABILITY-MECHANISMS` §4): the cost tracks how much of the output distribution the negatives rewrite.

---

## 9. Analysis (2): the three most transferable ideas for our engine, each with its obstacle

Our engine: OpenImages instance bank with SAM 3 instances (9,692 scenes / 311,282 rows), LaMa removals, cross-family judges (Qwen3.5-9B writer, Molmo2-8B listener, Gemma4-12B verifier), lineage rules (`RL-DESIGN` §9).

**A. Policy-adversarial clause-level negatives, verified and de-biased, trained through a separate null decision.**
Composition of three published pieces: SAM 3's acceptance rule ("keep only what the current model accepts", §1.1) applied at clause level (this is PAM, `ideas-3` §I3-1, already designed); ARHN's relabel-or-filter pass on every mined negative (§1.4 — expect ~⅓ of policy-accepted flips to be non-negatives or ambiguous; Ref-Adv's 18.7 % keep rate says human-unanimous hard items are 5× rarer than generated ones); SugarCrepe's blind-detectability gate (§5.1). Then train the `none` decision the way SAM 3 / Motto do — a separate loss that does not rewrite the coordinate stream — rather than 2:1 SFT.
*Obstacle:* an autoregressive MLLM has no side head; the project's harness reads p(null) off the same token stream. Training a presence token with the box loss masked on negatives (`survey-C` transfer (i)) has never been done in an MLLM, and Motto needed a detector-style query head to make top-K online selection meaningful. The cheap version is the ratio knob GroundingME already exposed (1:8 kept positives; 2:1 lost them) plus the format-diverse replay the charter prescribes; whether that reaches L4 without the −20 is precisely what the 1:8 row leaves open.

**B. Same-image pairs with a symmetrical pairwise objective (S-VCO for boxes).**
S-VCO (§4.5) is the only published loss built for (image, edited image, fixed text) pairs, and beat DPO by +5 on the same data; OViP (§4.6) shows failure-derived image negatives beat predefined ones online. Our LaMa removal + control edit already yields (I, I_removed, I_control) triples with objective labels, and GroundingME itself has only 6 same-image present/absent pairs (`ideas-3` §0), so the pairs must be manufactured anyway. Add the text pair (e, e_flip) on a fixed image and the four-cell design is the grounding twin of VisMin.
*Obstacle 1:* every S-VCO / DPO instance scores a free-text response; for us the "response" is a box string or `none`, and pairwise objectives on box tokens are unpublished (§5.3) — the IoU-vs-likelihood mismatch has to be designed, not borrowed. *Obstacle 2:* removal is a whole-object edit, which the base already separates (AUROC 0.964 `[P]`); the GME regime is a *detail* edit on one of several same-category instances, which needs an attribute-level editor and the FineCops-style DIRE/DINOv2 artifact gate, and a control-edit arm so "edited ⇒ none" cannot be learned (IDEA-13 §3.6).

**C. Verified per-candidate CoT hard positives, selected at the policy's frontier.**
HumanRef-CoT's recipe (§3.4: all candidate boxes marked, per-candidate check, keep only if the writer's final answer matches GT) with our cross-family judges in place of GPT-4o, FINER's "regenerate until the strong discriminator separates" gate (§3.6), and IDEA-13's selection rule (hard positive = verifier-unique and student pass@8 = 0). This is R1 for positives — the 41 % of GME positives boxed on a sibling with p > 0.999 are exactly items the base cannot sample and can only be shown.
*Obstacle 1 — judge ceiling:* GPT-4o wrote HumanRef-CoT for 1-2-detail person queries; our 9-12B judges' per-detail accuracy on 5-7-detail items is unmeasured, and the charter records that thinking mode *lowers* Discriminative for every model size (8B 61.3 → 52.5), so a per-candidate CoT format is not free on our target dimension. *Obstacle 2:* the writer and policy share the Qwen lineage (`ideas-3` §I3-1 trap); the Gemma-writer arm is mandatory. *Obstacle 3:* Rex-Thinker's +13.8 rejection came with +1.2 DF1 on an L2 set; on L4 the CoT-SFT positive cost is unknown.

---

## 10. Analysis (3): what nobody has done

1. **A data engine that mines multi-constraint (5-7 detail) expressions against the current policy with per-detail verification.** SAM 3 mines noun phrases against the model (§1.1); Motto mines online but verifies nothing (§1.2); FINER verifies against a *strong* model, not the trained policy, and produces yes/no queries (§3.6); Ref-Adv verifies with humans but builds *minimal* expressions and trains nothing (§3.3); RefBench-PRO / HumanRef-CoT verify against GT boxes with no policy gate (§3.1, §3.4). No published set carries, per item, (clause count, which clause is false, base p(null), which sibling the base boxed). PAM (`ideas-3` §I3-1) is that engine on paper; its precondition kill (≈ 2 GPU-h) is the cheapest way to learn whether the OpenImages bank can produce GME-regime negatives at all.
2. **Self-play where the speaker is paid for expressions the listener fails on and an independent verifier resolves.** VisPlay pays for 50 % uncertainty (§2.1); RRG and the gaze paper freeze the listener and pay for listener *success* (§2.3-2.4); R-Zero and Active Zero target the "edge". IDEA-13's frontier bonus (pass@8 ≤ 0.5, hard positives at pass@8 = 0, acceptance by frozen U) is the only design that rewards failure, and it is unrun. The reason it is possible for grounding and not for VQA: the label (target box) is objective and does not come from the student.
3. **A pairwise objective on box outputs over same-image edited pairs** (§9-B) — no S-VCO / DPO / listwise entry scores a box or a `none`.
4. **A presence / `none` decision in an autoregressive grounding MLLM trained on policy-accepted negatives with the box loss masked on negatives** (SAM 3 Table 10's supervision strategy on an MLLM). Motto is the nearest (detector head on latent tokens); no pure-token MLLM has done it; GroundingME's SFT is the negative control.
5. **False-negative relabeling of VLM hard negatives** (ARHN, §1.4) — every VLM hard-negative pipeline treats its mined negatives as clean; none reports the fraction that a stronger judge would relabel.
6. **Reporting the pass@k distribution of the items a data method gained on.** Not one of the ~35 papers above does it; the charter's question is unanswerable from the literature as written, which is itself the finding.
7. **A clause-count ladder with per-clause labels for grounding** (§7): the FINER curve exists for VQA; no REC set carries verified per-clause labels, and 2603.27226 says ordering without them is not the lever.
8. **Training a grounder on edited-image negatives at all** — FineCops-Ref built 8,507 negative images and used none for training (§4.3); the 2025-26 arXiv search found no successor.

---

## 11. Verification ledger and limits

- **Could not access:** Motto Appendix D.2 text (two HTML fetches, PDF over the fetch size limit) — the K-ablation sentence is cited from survey D's pypdf read; the Group Revision CVPR paper (CVF 403) — all numbers are from a secondary notes page and carry `[V-2nd]`; HNC's PDF (size limit) — HTML used.
- **Memory-only:** NegCLIP's construction and ImageNet cost; CE-CLIP; HNC's original venue/date.
- **Discrepancy resolved by choosing the pypdf read:** Motto PR-Bench N-Acc is 46.9 (survey D, Table 1), not the 71.7 my HTML summariser printed (that is mAcc).
- **Open number I could not extract:** GroundingME Rejection at the 1:8 ratio (only the 2:1 GME row is printed in the text I could read). If the paper's table has it, it decides whether the −20 positive cost and the 27.9 OOD gain are separable.
- **Search coverage:** the shared WebSearch budget (200/200) ran out after my 22nd query; four planned gap checks (detection-side model-in-the-loop negative engines 2026; hard-negative DPO on boxes; self-play REC 2026; edited-pair grounding training 2026) were run as arXiv-search-page fetches and returned nothing relevant. Absence claims in §4.8, §5.3 and §10 items 3 and 8 rest on that plus the team's earlier surveys, not on an exhaustive search.
- **Not re-read:** SAM 3 §D.4 acceptance rule, GroundingME's dimension deltas, Ref-R1's RL-adds-zero finding, Jedi — taken from the named team files, which read the primary PDFs.
