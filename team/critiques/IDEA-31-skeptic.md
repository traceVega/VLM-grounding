# Skeptic record for IDEA-31: Gaze as process reward for crop policies

Idea file: `ideas/IDEA-31-gaze-process-reward.md` (Researcher3). Status at review time: ABANDONED (2026-09-02, before review).

## Round 1 (2026-09-02)

**Verdict: none required; ABANDONED by the author (CONCEDE to pre-screen NF-3).**

Record. The pre-screen verdict (landscape/skeptic-crowded-map.md Section 6.3, NF-3) was CROWDED-adjacent and small: GazeVLM uses the word "gaze" for internal attention control trained with GRPO [VERIFIED: 2605.07817]; GazeLLM and gaze-driven efficient VLMs already put human gaze into MLLMs [VERIFIED: 2504.00221, 2509.16476 (listings)]; models already know where to look without supervision [VERIFIED: 2502.17422]; and Researcher2's Section 8.7 and 8.12 found that no gaze dataset exists at the needed scale on non-COCO images, that webcam gaze error (about 100 px) exceeds the crop precision being supervised, and that a new collection is a six-to-eight-week data project. Researcher3 conceded and folded two components into IDEA-91: the scanpath-faithfulness metric and the human-evidence validation. I agree with the abandonment; the reasoning is recorded in the idea file and no further review is needed.

One note for IDEA-91: if the human-evidence validation is implemented as click-based evidence marks (Researcher2's proxy, about 10 px precision), it must be described as human evidence marks, not gaze, and its agreement with the interventional score should be reported as a validation of the score, not as a training signal.
