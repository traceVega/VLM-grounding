# Skeptic critique of IDEA-13: Referring games at MLLM scale

Idea file: `ideas/IDEA-13-referring-games.md` (Researcher1). Reviewed version: UNDER REVIEW, last updated 2026-09-01.

## Round 1 (2026-09-02)

**Verdict: MAJOR REVISION.**

Summary judgement. The problem statement is right and the drift controls are the standard ones. Two pre-screen conditions were ignored (the VisPlay-family positioning and the prompted distractor-aware baseline), and the loop has a structural gap: accepted expressions are, by construction, the ones the current listener already solves, so the mechanism by which the next listener learns anything the current one cannot do is unstated. Fixable, but the fix changes the reward.

### Objections

**13-N1 (NOVELTY, pre-screen condition ignored).** Generic image self-play with GRPO on unlabeled images exists and is not cited: VisPlay (questioner and reasoner roles, diversity and difficulty rewards, Qwen2.5-VL-3B average 30.6 to 47.3 over three iterations, CVPR 2026) [VERIFIED: 2511.15661]; Vision-Zero gamified self-play [VERIFIED: 2509.25541 (listing)]; Active Zero [VERIFIED: 2602.11241 (listing)]; unsupervised self-evolution with self-judging [VERIFIED: 2603.21289 (listing)]. A reviewer will summarize IDEA-13 as "VisPlay with a box win condition". The novelty test must be rewritten against VisPlay, not against text-only Language Self-Play and Syn-GRPO, and the delta must be the geometric, listener-verifiable win condition plus the distractor curriculum, stated as such. GREx, which trains generation and comprehension jointly in one model [VERIFIED: 2601.05244], also belongs in Section 5.

**13-M1 (METHOD, the trivial baseline is missing; pre-screen condition ignored).** The pseudo-label baseline ("Grounding DINO boxes plus captioner expressions") does not see the distractors, so it cannot produce discriminative language and is the wrong control. The baseline that tests whether RL self-play is needed is: prompt a strong MLLM (Qwen3-VL-32B, or the same 4B base) once with the target box and the distractor boxes drawn on the image, ask for a unique description, filter with the same frozen listener, and train the listener on that corpus at equal size. If that matches the game, the speaker RL is decoration and the paper is a data-engine paper with a prompting recipe. This baseline goes in the first table and in the kill experiment.

**13-M2 (METHOD, the loop cannot produce signal on what the listener fails).** Accepted expressions are those where L_t succeeds; L_{t+1} is trained on them. Curriculum tiers rise only when success exceeds 80%. So every training example is, by construction, solvable by the current listener, and the only route to improvement is consolidation (self-training on confident predictions) plus the human replay. Yu et al. anchored the game on human expressions [VERIFIED: 1612.09542 (listing)]; here the anchor is Ref-L4 replay, which is not discriminative data. State the mechanism, and add two things: (a) a listener-independent acceptance check, e.g. uniqueness judged by a stronger frozen verifier or by SAM 3 concept matching over the distractor set, so that objectively discriminative expressions the listener *fails* on are kept as hard positives; (b) a "listener self-training without speaker" baseline (listener trains on its own confident predictions on the same images) to separate the speaker's contribution from consolidation.

**13-M3 (METHOD, the game can be won with location shortcuts).** A speaker rewarded for listener success against a weak listener will discover that absolute-position and size descriptors ("the leftmost one", "the big one") are the easiest way to be unique. Those pass the naturalness judge and human listeners, so none of the drift controls catches it. RefCOCO+ banned location words for this reason [LIKELY per R3's history], and Cirik et al. showed RefCOCO-family accuracy is largely bag-of-words plus recognition [VERIFIED per R3: 1805.11818]. The resulting listener learns positional shortcuts, which Ref-Adv is built to punish [VERIFIED: 2602.23898]. Required: report the descriptor-type distribution of accepted expressions per round (attribute, relation, absolute location, size, count), and either constrain the speaker (RefCOCO+-style location-word penalty or a per-type quota) or show that GroundingME-Discriminative gains survive when location-word expressions are removed from the listener's training set.

**13-A1 (ASSUMPTION, "human referring data are structurally short").** Retained from the pre-screen. RefCOCO/+ expressions are indeed short (about 3.5 words [LIKELY: Ref-L4 paper statistics]), but RefCOCOg averages about 8 words [LIKELY], Ref-L4 averages 24.2 words [VERIFIED: 2406.16866] and RSC uses paragraph-length scenario queries [VERIFIED: 2604.02323]. The defensible claim is "human expressions are under-informative relative to hard same-category distractors, because annotators stop when a human would succeed", and it should be shown on GroundingME-Discriminative failure cases (how many have an expression that a human listener finds unique), not asserted.

**13-M4 (METHOD, listener success criterion).** "IoU >= 0.5 and no distractor selected" is one event when the listener emits one box. The informative signal is whether the listener's box overlaps a distractor more than the target; define success as IoU(target) >= 0.5 and max IoU(distractors) < 0.5, and report the distractor-confusion rate separately, since that is the quantity the paper claims to improve.

**13-E1 (EVAL, contamination statement).** OpenImages and Objects365 are in Qwen3-VL's grounding mixture [VERIFIED by Verifier V-095: 2511.21631], so "no COCO" is not "clean". Say which pools are seen by which base; make Molmo2 the primary base for the clean-image claim. SAM 3's SA-Co data may also overlap these pools [LIKELY]; note it.

**13-E2 (EVAL, judge and reward hacking).** The naturalness judge is an open VLM/LLM and is inside the reward; judges inside rewards are gamed [VERIFIED: 2607.09492]. The 500-item human study is the right check, but it is run once; run a 100-item human check per round so drift is caught before the next listener is trained on it.

**13-F1 (FEASIBILITY, minor).** Two rounds at 4B in 150 GPU-h is plausible given Researcher2's numbers. The 200k-expression release depends on SAM 3's licence for derived data; resolve before promising a release.

### What would raise the verdict
1. Rewrite the novelty test against VisPlay and GREx (13-N1).
2. Add the prompted distractor-aware baseline and the listener self-training baseline to the kill experiment and first table (13-M1, 13-M2).
3. Add a listener-independent uniqueness check so hard positives enter training (13-M2).
4. Report descriptor-type composition per round and control location shortcuts (13-M3).
5. Fix the contamination statement (13-E1) and restate the "short expressions" premise with evidence (13-A1).

### The two objections that matter most
13-M2 (the loop only generates what the listener already solves) and 13-M1 (the prompted distractor-aware baseline). If the prompted baseline matches the game, or if the listener-self-training baseline matches it, the contribution is a prompting recipe.

### Addendum (2026-09-02, after reading the pre-round-1 revision of the idea file)
The author revised the file against my pre-screen before I posted round 1, and the two versions crossed. Against the current file:
- 13-N1: RESOLVED. VisPlay, Vision-Zero, Active Zero, GREx and the SAM 3 data engine are cited and the novelty test is rewritten against VisPlay, SLR 2017 and GREx.
- 13-A1: RESOLVED. The premise is restated as "under-informative relative to hard distractors" with the Ref-L4 figure.
- 13-M1: RESOLVED for the baseline (B1 prompted generation is mandatory in the first table and in the kill rule, with the explicit "if the game beats B3 but not B1, stop" clause). 
- 13-E2: PARTIALLY RESOLVED. A 500-item human-listener study per round is now in the file; the judge-gaming point stands as a note, no action needed beyond reporting the judge-vs-human agreement per round.
- Still live and requiring an answer: 13-M2 (the loop only generates what the listener already solves; listener-independent uniqueness check and listener self-training baseline), 13-M3 (location-shortcut language; descriptor-type composition per round and a control), 13-M4 (success criterion and distractor-confusion rate), 13-E1 (OpenImages is in Qwen3-VL's mix; the file now acknowledges this for Ref-L4 but the pool statement should say it too, and Molmo2 should carry the clean-image claim).

### Addendum 2 (2026-09-02, after the Verifier's IDEA-13 report and Researcher3's drift paragraph)
Two objections are added on the strength of `verifications/IDEA-13-verifier.md`, which I endorse:

**13-M5 (METHOD, the incumbent data engine is stronger than B3).** The incumbent is not "Grounding DINO plus captioner". VLM-generated referring-expression engines already exist at scale: 16M model-generated expressions over 1M human-annotated objects with attribute and spatial-relation prompting, giving zero-shot RefCOCO SOTA without human grounding labels [VERIFIED by Verifier: 2407.14563], and GroundingSuite's multi-agent pipeline with 9.56M expressions, ICCV 2025 [VERIFIED by Verifier: 2503.10596]. B3 and the kill experiment must use an equal-size VLM-generated set built with that prompting (attributes, relations) on the same images. The defensible delta is then "listener-verified informativeness against hard distractors beats unverified VLM descriptions", which is also the claim B1 tests from the other side. If neither B1 nor the GroundingSuite-style set is beaten by 2 points with CIs, stop.

**13-M6 (METHOD, the listener SFT control must be difficulty-matched).** The data-centric analysis of RL versus SFT finds RL's OOD advantage comes from implicit filtering to medium-difficulty samples and that difficulty-curated SFT matches or beats RL [VERIFIED by Verifier: 2602.10815]. The listener phase's "SFT-only arm at equal data" must therefore be difficulty-matched (same tier distribution) or the RL-vs-SFT comparison inside the listener is confounded.

**13-A2 (ASSUMPTION, minor).** GroundingME's top model scores Discriminative 69.6, Spatial 49.7, Limited 54.0, Rejection 0.0 [VERIFIED by Verifier V-111], so Spatial and Rejection, not Discriminative, are its weakest dimensions. Section 1's framing should say the game targets Discriminative and Spatial as the dimensions where informative language is the lever, not "the main drivers of 45.1".

Researcher3's drift paragraph (Discussion log, 2026-09-02) supplies the held-out-listener control from a different lineage and the per-round descriptor-type distribution, which together answer 13-M3 and 13-E2 if they are moved from the log into Sections 3 and 6. Andreas and Klein 2016 [VERIFIED by Verifier: 1604.00562] belongs in the pragmatics lineage.

Live objections after this addendum: 13-M2, 13-M3 (paste the controls), 13-M4, 13-M5, 13-M6, 13-E1, 13-A2.

### Discussion pointer
Author responses go in the idea file's Discussion log, one line per objection id. I will re-review after Status is set to REVISED.

## Round 2 (2026-09-02, reviewed version: REVISED, last updated 2026-09-02)

**Verdict: MINOR REVISION (up from MAJOR REVISION).** PROMISING on ACCEPT of 13-M7 and 13-I1 below; on those I have no objection to SURVIVING as a conditional entry whose kill experiment (about 30 to 40 GPU-h) runs before any further commitment.

Round-1 disposition. All seven live objections and the earlier resolved ones are answered with edits I can check in the file. 13-M2: acceptance is a frozen Qwen3-VL-32B verifier U with SAM 3 concept matching as a secondary check; hard positives (U-unique, student pass@8 = 0) enter listener SFT; the speaker gets a frontier bonus; the listener's GRPO runs on pass@8-in-(0,1) bins; B7 isolates consolidation; the original listener-dependent acceptance is kept as an ablation (Sections 2, 3.2, 3.4, 3.5, 6). The partial REBUT is accepted on both points: the negative mode's ground truth never depended on the listener, and greedy-solvable is not sampling-solvable; my round-1 sentence should have read "solvable by the current listener under the acceptance decoding". 13-M3: non-extremal-target tier, descriptor tagger validated on 200 human tags per round, location-word penalty arm, removal ablation (3.1, 3.7). 13-M4: one-box rule by format, the two-part success criterion, distractor-confusion rate on every instance-annotated set (3.3). 13-M5: B3 is the 2407.14563 / GroundingSuite-style set on the same instances, unfiltered and U-filtered, in the first table and the kill rule; 2407.14563 is the third closest paper (Sections 1, 4, 6). 13-M6: B8 difficulty-matched on tier and pass@8 bin (3.5, 6). 13-E1: SA-1B primary, OpenImages flagged, Molmo2 with the V-112 wording (3.1). 13-A2: per-dimension GroundingME scores and the corrected framing (Section 1). 13-E2 and 13-F1: per-round human check with the round-0 stop rule; SAM licence stated as contingent and the release not promised.

### New objections

**13-M7 (METHOD, B1 must receive the same hard-positive selection).** The game's corpus is now U-filtered *and* enriched toward the student's frontier (hard positives, with a bonus that steers the speaker to them). B1 is U-filtered only. If the game beats B1, the reason may be the selection step (keep what U resolves and L_t fails on) rather than speaker RL, and selection is free. Required: a B1-HP variant, the prompted 32B corpus with the same U filter and the same hard-positive enrichment at equal size (and the same treatment for U-filtered B3), in the first table and in the kill rule's "best of B1 and B3". If the game beats B1 but not B1-HP, the contribution is a selection recipe; write that outcome into the kill rule next to the existing "beats B3 but not B1" clause.

**13-I1 (IMPACT, the verifier is a same-lineage 32B teacher, so the loop is distillation unless shown otherwise).** With U = Qwen3-VL-32B and the student = Qwen3-VL-4B, hard positives are by definition items the 32B resolves and the 4B does not; the corpus cannot contain discrimination that U lacks, which Section 8 already concedes. The paper is then "a 4B listener approaches a 32B listener through a speaker-generated, verifier-filtered corpus", which is verifier-in-the-loop distillation with a data engine, not self-play in the sense of VisPlay or SLR 2017, and the novelty sentence in Section 4 should say which it is. Required: (a) U's own scores on every benchmark as a row of the first table, so the student's headroom is visible; (b) a pre-registered U-exceedance test: does the round-2 student beat U on GroundingME-Discriminative or Ref-Adv 2026? If yes, the loop generated signal beyond its verifier, which is the self-play claim and would be the paper's strongest result; if no, the framing is "verifier-in-the-loop data engine with distractor conditioning" and the delta over 2407.14563 is distractor conditioning plus verification plus frontier selection; (c) one run with U at the student's own size (a frozen Qwen3-VL-4B base as U), the true self-play configuration, so the reader can see what the 32B verifier buys.

**13-E3 (EVAL, the kill rule's 2-point margin on a small split).** GroundingME-Discriminative is a few hundred items, so a 2-point difference between 25k-expression listeners at one seed is inside its noise. Make Ref-Adv 2026 (larger) the primary kill benchmark, state n and the paired MDE at three seeds, treat GroundingME-Discriminative as secondary in the kill decision, and add the distractor-confusion rate on the game's own held-out set as a third and larger instrument.

**13-M8 (METHOD, minor).** Section 2 defines a hard positive as pass@8 = 0 while the reward bonus fires at pass@8 <= 0.5; name the two thresholds differently (hard positive versus frontier item) or use one, and report the fraction of accepted expressions in each pass@8 bin per round.

### The two objections that matter most
13-M7 (a free selection step may be the whole effect) and 13-I1 (the paper must know whether it is self-play or distillation before it claims either).

### Discussion pointer
Answer 13-M7, 13-I1, 13-E3, 13-M8 in the Discussion log; on ACCEPT of 13-M7 and 13-I1 I will record PROMISING as final without a further round.

### Round-2 addendum (2026-09-02, after the Verifier's `FINAL-evidence-summary.md` and claims V-195, V-198)
Two items join round 2; the verdict stays MINOR REVISION and 13-N2 joins the conditions for PROMISING because it narrows the delta sentence.

**13-N2 (NOVELTY, the "incumbents use no listener" delta is REFUTED for GroundingSuite).** Section 1 says that none of the incumbent engines "verifies that a listener can resolve" its expressions. The Verifier finds that GroundingSuite filters its generated expressions with EVF-SAM at IoU 0.5 [VERIFIED by Verifier V-198: 2503.10596], which is a segmentation-model listener filter. The U filter alone is therefore not new; what remains unclaimed is verification against a *controlled distractor set* (uniqueness among same-category instances, not IoU with the source mask), the non-extremal and similarity tiers, hard-positive selection toward the student's frontier, and the rejection mode. Required: rewrite the Section 1 sentence and the Section 4 delta accordingly; make B3's U-filtered variant explicitly the GroundingSuite-style "filtered incumbent" so the first table separates "filtered" from "distractor-verified and frontier-selected".

**13-E4 (EVAL, citation label, minor).** SPARK (2605.05546) is text-only self-play over knowledge graphs, not VLM self-play [VERIFIED by Verifier V-195]; remove it from the VLM self-play list in Section 5 or relabel it.

Conditions for PROMISING now: ACCEPT of 13-M7, 13-I1 and 13-N2.

## Final verdict after round 2 (2026-09-02)

**PROMISING. No objection to SURVIVING as a conditional entry whose kill experiment runs first.** Checked in the current file: 13-M7 (B1-HP and B3-HP at equal size in the first table; the kill rule compares against the best of B1, B1-HP, B3, B3-HP with the selection-recipe stop clause); 13-I1 (U scores as row B0-prime; pre-registered U-exceedance test on Ref-Adv 2026 and GroundingME-Discriminative with paired CIs per round; one configuration with U = the frozen 4B base as the true self-play setting; Section 4 commits to the self-play framing only if the student exceeds its verifier or the frozen-4B configuration shows gains, otherwise a verifier-in-the-loop data-engine paper); 13-N2 (B3 carries the EVF-SAM IoU-0.5 filter as a mandatory variant and the delta against GroundingSuite is narrowed to the controlled distractor set, speaker training on the listener signal, curriculum, rejection mode and the listener-independent check); 13-E3 (Ref-Adv primary with n and the paired MDE stated, pooled with OpenRef multi-target and a held-out game set if too small; GroundingME-Discriminative secondary; distractor-confusion rate reported); 13-M8 (hard positive versus frontier item defined separately); 13-E4 (SPARK relabelled). Ranking unchanged: sixth, P 0.15; clarity 4. The most probable outcome remains the kill rule firing, which is why it stays a conditional entry.

Closure (2026-09-02): 13-N2 is applied beyond what the final record describes. Sections 1 and 4 no longer claim the U filter as a delta; B3 is split into B3-U (unfiltered, the 2407.14563 recipe), B3-F (the GroundingSuite-style EVF-SAM filter at IoU 0.5 against the source instance), B3-V (the same corpus under the distractor-uniqueness criterion) and B3-HP, and the kill rule names B1, B1-HP, B3-F, B3-V and B3-HP. That ladder (unfiltered, filtered, distractor-verified, frontier-selected, game) is exactly the decomposition the first table needs; 13-E4 applied. Final verdict unchanged: PROMISING, SURVIVING as a conditional entry.
