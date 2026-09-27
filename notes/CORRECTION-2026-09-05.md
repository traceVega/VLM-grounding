> **Superseded in part by `REJECTION-DEEP-DIVE-2026-09-16.md` §3/§5** — the "expression length" variable in §1 is confounded with the Discriminative/Text sub-axis (the 28 short items are exactly the 28 Text items); sub-axis vs dilution is open pending probes P1/P10.

# Correction, 2026-09-05: three claims in the two notes below were wrong

Found by an adversarial panel (16 agents) over the results and confirmed from
files on disk. Supersedes the affected passages in `ABSTAIN-SIGNAL.md` and
`GME-PRECHECK.md`; those files are kept as written for the record.

## 1. "4% abstention, AUROC 0.69 on GroundingME removals" was a selection artifact

The first 78 items reviewed are the first 78 by item id. They are all
Discriminative and their expressions run a median of 48 words. The 255
unreviewed items run a median of 19. Over all 333 removals, by expression
length:

| words | n | declines | median shift (orders) | AUROC |
|---|---:|---:|---:|---:|
| < 15 | 92 | 14% | +2.3 | 0.708 |
| 15–30 | 78 | **28%** | +3.4 | 0.713 |
| 30–60 | 97 | **0%** | +1.0 | 0.629 |
| ≥ 60 | 66 | **0%** | +0.6 | 0.610 |

Within Discriminative alone, which holds the image distribution and the
editor fixed:

| words | n | declines | median shift | AUROC |
|---|---:|---:|---:|---:|
| ≤ 20 | 28 | **43%** | **+12.6** | **0.972** |
| 21–35 | 17 | 12% | +1.8 | 0.758 |
| > 35 | 79 | **0%** | +1.3 | 0.685 |

The short Discriminative items are a head noun plus one discrete, checkable
clause: *"a vehicle, with number '2500' on its body"*, *"a building. It has
text 'RAUMEN' on its side"*. The long ones are graded appearance. **On the
same benchmark images, with the same removal, the class-label regime (16
orders, half the time abstaining) comes back when the expression is a single
checkable clause and vanishes when it is a paragraph of appearance.** The
variable is the expression, not the scene. Spatial is flat at every length
(0% declines, +0.3 to +0.5 orders).

The unreviewed items have not been checked for removal cleanliness. An unclean
removal lowers p(null) (the not_clean cell sits 1.4 orders above control, not
12), so it cannot manufacture a +12.6-order rise; but the 43% should be
confirmed by review before it is quoted.

## 2. The head-noun probe (`gme_headnoun_probe.py`) was invalid as designed

Under "the lamp" on a Discriminative image there is, by construction, another
lamp. The short-prompt ORIGINAL box hit the ground-truth referent on 13/50; the
probe mostly grounded a different object, for which `null` is the wrong
answer. Its result (0/50 declines, AUROC 0.429) measures nothing about the
referent and must not be read as "shortening the prompt does not help." The
within-Discriminative length split above is the valid form of that test.

## 3. "The model never wrongly abstains" holds on OpenImages only

On GroundingME's 804 positives it abstains on 41 (5.1%), 40 of them in
Limited, where referents have a median area of 0.1% of the image. Across the
benchmark, AUROC of p(null) for absent (Rejection) against present (all
positives) is **0.298**: p(null) is *lower* when the object is absent than
when it is present, because it tracks how hard the object is to see, not
whether it is there. The 0/316 figure is real but does not generalise.

## What stands unchanged

Same-box 0/48 on long expressions (the box follows the referent; the audit
premise is dead). K1 editor gate at 0.50. V 316/316 on OpenImages. Class-label
removals: 16 orders, AUROC 0.964, 50% declines. GroundingME ORIGINAL 41%
against a published 45.1%; Rejection 0/201.
