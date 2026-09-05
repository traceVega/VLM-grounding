# The model always knows; greedy decoding cannot act on it

Run 2026-09-04, Qwen3-VL-8B-Instruct at `0c351dd0`, 332 human-clean removals
(339 less the box review's drops) times three conditions, 996 prompts, one
attempt after the checkpointing fix. Exploratory, `--non-kill`: class labels
rather than referring expressions, one model; no number here may enter a kill
table or change P8 to P21. Data in `tables/pilot_abstain.jsonl`.

Half of the verified-clean removals get an abstention and half get a box. That
rate cannot distinguish a model that cannot see the object left from one that
can and answers anyway, and the two have opposite costs to fix. So what is
recorded is the decision rather than the output: the prompt ends at a two-way
choice, and the token after `{"bbox_2d":` is either ` null` or ` [`.

Ground truth comes from the pair, not a judge. Every image runs in both edited
conditions -- REMOVE takes the referent, CONTROL_OBJ takes a different object of
matched size -- same editor, same hole, same artifact, differing only in whether
the referent survived. P3's control doing what it was designed for.

## The separation is eleven orders of magnitude

| condition | the referent | median p(null) |
|---|---|---:|
| ORIGINAL | present, unedited | answers, 0 abstentions |
| CONTROL_OBJ | present, something else removed | **1.3e-18** |
| REMOVE | gone | **0.62** |

**AUROC 0.964** (95% CI 0.950 to 0.976, n=316 vs 316, ORIGINAL-correct items).
No ties: every value is strictly inside (0, 1), so this is margin, not
tie-breaking.

The half of removals where the model still emits a box are not a separate
population. Their p(null) has a log10 median of **-6.6** against the controls'
**-17.9**. Even while answering, the model registers the removal by eleven
orders of magnitude. It never abstains wrongly -- 0 of 316 controls -- so the
operating point is not miscalibrated in both directions; it is maximally
conservative in one.

## What moving the threshold buys

| false-abstention budget | correct abstention |
|---|---:|
| today (the model's own decoding) | 50% |
| 1% | 70% |
| 5% | **83%** |
| 10% | 91% |

Thirty-three points for a 5% cost. But the thresholds live at 1e-4 and below,
where bf16 and a change of image distribution both matter, so this is the size
of the prize rather than a recipe. What it establishes is that the information
is present and the failure is in the policy that reads it, which is the half of
the problem that training can move.

## V, measured for the first time, is perfect

Removing a non-referent object must not move the box.

| | |
|---|---|
| box held at IoU >= 0.70 | **316 / 316** |
| abstained on a control | 0 / 316 |

Verified not to be an artifact of identical images: the control edits change
0.5% to 14% of pixels, comparable to the removals, and 214 of 332 produced a
*different* box that still fell inside IoU 0.70.

Consequences. P16(b') cannot fire: the edits do not confuse this model. And
since the headline counts predictions that fail `N+` **or** `V`, and `V` never
fails here, the 45% from the box review stands as the whole of it.

## The open confound

Both conditions carry an inpainting artifact, so "was this edited" is
uninformative -- but *where* the artifact sits is not. REMOVE puts it on the
referent and CONTROL_OBJ puts it elsewhere, so the model could be detecting an
artifact over the region it was asked about rather than the object's absence.

The cell that separates them exists and is already labelled: the 127 removals
the author marked `not_clean`, where the inpainting ran on the referent and the
object survived anyway.

| | referent present | referent gone |
|---|---|---|
| no artifact on it | ORIGINAL | -- |
| **artifact on it** | **not_clean (unmeasured)** | REMOVE |
| artifact elsewhere | CONTROL_OBJ | -- |

High p(null) on `not_clean` means the artifact is the cue and this result is
about edit detection. Low p(null) means the model is reading absence. 127
prompts, about six minutes.
