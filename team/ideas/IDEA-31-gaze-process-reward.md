# IDEA-31: HINT revisited: human gaze as process supervision and faithfulness metric for hard-attention (crop) policies
Status: ABANDONED (2026-09-02, before review; reasons below; two components folded into IDEA-91)
Author: Researcher3    Last updated: 2026-09-02
Origin: `landscape/r3-history.md` Section 3, NF-3.

## 1. Research problem (as proposed)
Attention supervision from human attention maps (VQA-HAT 2016; HINT and SCR 2019) was shown in 2020 to act as a regularizer rather than as grounding: random cues gave the same gains `[VERIFIED: arXiv 2004.05704, Verifier V-069]`. The proposed explanation is architectural: soft attention is not an information bottleneck, so a model can attend correctly and still answer from priors. Crop-and-zoom policies make region selection a hard bottleneck, which reopens the question of whether human process supervision (fixations, in order) helps, and whether human scanpaths can serve as a process-level faithfulness metric for thinking-with-images models.

## 2. Core insight (as proposed)
Supervising *where to crop* is load-bearing by construction; gaze is the only grounding signal that carries an order; the Shrestha random-region control is trivial to run for a crop policy.

## 3. Why it is abandoned

Three independent assessments converged, and I agree with them.

1. **Data (Researcher2, `r2-practical.md` 8.7 and 8.12).** Eye-tracking VQA sets are tiny and all on COCO / Visual Genome images, which are contaminated for grounding: AiR 1,422 questions and 20 subjects `[VERIFIED: arXiv 2007.14419 / 2204.09774]`, VQA-MHUG 3,990 stimuli `[VERIFIED per Researcher2: arXiv 2109.13116]`, VQA-HAT is mouse-driven de-blurring rather than eye tracking `[LIKELY]`. A new collection is not blocked by money (2-4k USD for 10k webcam items) but by timeline (IRB plus 6-8 weeks), by image contamination (must be collected on non-COCO images), and by the precision mismatch between webcam gaze (about 100 px) and the crops being supervised. The click-based "mark the evidence" proxy is cheap but changes the claim from gaze to human evidence marks, which DeFacto-1.5K and answer-grounding datasets already provide.
2. **Crowdedness (Skeptic, Section 6.3 NF-3).** GazeVLM uses "gaze" for *internal* attention control trained with GRPO and reports about +4% on HR-Bench with no human gaze `[VERIFIED: arXiv 2605.07817 abstract]`; a reviewer will conflate the two. GazeLLM (arXiv 2504.00221) and gaze-driven efficient VLMs (arXiv 2509.16476) already put human gaze into MLLMs `[VERIFIED per Skeptic: listings]`. Models already know where to look without supervision `[VERIFIED: arXiv 2502.17422]`. Human fixations are not crop targets: they include reading and confirmation fixations and have low inter-annotator agreement `[LIKELY per Skeptic]`, so the paper would first have to show fixation-derived crops beat saliency-derived crops.
3. **Shape of the result (my own assessment).** With the random-control arm as the headline, the likely outcome is a negative result about process supervision at about 1.4k items: publishable, short, and not what the team was asked to find. With existing evidence annotations instead of gaze, the direction collapses into evidence-consistency training, which DeFacto already does.

## 4. What is preserved (folded into IDEA-91)
- **Human evidence data as the validation set for the interventional score.** AiR, VQA-MHUG and DeFacto-1.5K evidence regions, and especially COCO-Search18's 6,202 target-present / target-absent images with fixations `[VERIFIED per Researcher2: cocosearch site]`, are used in IDEA-91 Sections 3.3-3.4 as human-labelled anchors and as *natural* absent-target pairs, which is what the Skeptic asked IDEA-91 to add.
- **Scanpath faithfulness as an optional evaluation.** For zoom-based reasoners in IDEA-91's audit, the sequence of crops is compared with COCO-Search18 fixation order (sequence alignment); reported as a secondary column, not a claim.
- **The random-region control** survives as the interpretation rule for null results in IDEA-91 Section 3.2 (decorative boxes versus unread crops).

## 5. Conditions under which it could be reopened
A partner lab with a tracker and a subject pool, a collection on non-COCO images with expression-conditioned targets, and a first result that fixation-derived crops beat saliency-derived crops on a small-object suite. None of these exists on the team's timeline.

## Discussion log
[2026-09-02] Researcher3 -> Skeptic: NF-3 verdict (CROWDED-adjacent and small; fold or abandon): CONCEDE. Abandoned; scanpath metric and human-evidence validation folded into IDEA-91 as described in Section 4.
[2026-09-02] Researcher3 -> Researcher2: 8.7 tier-C verdict: ACCEPT; the click-proxy fallback is noted but not pursued because it duplicates DeFacto-1.5K-style evidence marks.
