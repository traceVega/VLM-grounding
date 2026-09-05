# Human review of the removal bank, and what the model does on it

740 mask REMOVE edits from the OpenImages K1 pool, every one judged by the
author on 2026-09-04: **did the object leave the picture?** One pass, one
question, four keys. This replaces the Gemma4-12B verifier, which agreed with
the author on 28% of a 32-item check (60% false rejection, 92% false
acceptance) and cannot carry a denominator.

Exploratory. The pool is class labels on OpenImages under `--non-kill`, not
GroundingME referring expressions under P12, so no number here is P15's and
none may be quoted as a K2 result.

## The bank

| label | n | share | meaning |
|---|---:|---:|---|
| clean | 339 | 46% | the object is gone |
| bad_item | 159 | 22% | the box never held the whole referent, or another instance remains |
| not_clean | 127 | 17% | still there, or only blurred |
| unsure | 115 | 16% | could not be called either way |

`bad_item` is a judgement about the item, not the edit. OpenImages labels are
hierarchical and often box a part: the first panel was "Toy" with a box on a
fallen shell, the radio-controlled car itself standing beside it. The shell
came out cleanly and "is the Toy gone?" is still no. Such an item cannot be
scored in either direction and leaves the pool.

## What Qwen3-VL-8B does, by label

Items where the ORIGINAL box was correct (IoU >= 0.5 against ground truth),
crossed with the pilot's 400 inference records.

| | clean (n=182) | not_clean (n=59) | bad_item (n=23) |
|---|---:|---:|---:|
| says `none` | 48% | 3% | 17% |
| returns a box | 52% | 97% | 83% |
| **same box (IoU >= 0.5)** | **13%** | **81%** | 9% |
| median IoU of the boxes it does return | 0.098 | 0.725 | 0.000 |

Two readings.

**The labels hold.** The only variable between the first two columns is whether
the object is actually in the picture, and the model's behaviour separates by 68
points. A noisy review would not produce that gap.

**The necessity relation works, and the audit headline is smaller than the idea
assumed.** Same-box on verified-clean removals is 13%, under P16(c)'s 20%. The
author's own labels on 32 cases put the true rate lower still: several same-box
cases are legitimate, the referent's position being recoverable from what the
removal left behind (a neck under a removed head, a car around a removed wheel).

## The constraint this exposes

Not every referent is removable in principle. Where the removal leaves the
information that fixes the object's position, a surviving box is correct rather
than hallucinated. `bad_item` catches the extreme form; the mild form -- parts
of wholes, and referents whose place is implied by their surroundings -- is not
caught by P8's sampling rule, which selects on mask quality alone. K2 is not
frozen (B0b unsigned), so this is the moment to encode it.

## Where the labels live

`data/human/openimages_removal_labels.csv`, one row per `image_id`. Written by
`scripts/review_removals.py --serve`, which renders the panel a judgement was
made from and can re-render it from the edit index at any time.

---

# Second pass: where the box went

The first review asked whether the object left the picture. This one asks, on
the 182 clean removals whose ORIGINAL box was correct, whether the answer the
model gave instead of `none` was acceptable -- a real instance inside the box,
or nothing at all. Labels in `data/human/openimages_box_labels.csv`.

The referent is verifiably gone and its class occurs once, so `none` is the only
valid output. `N+` counts a box as a pass only when a new region genuinely
satisfies the expression.

| | n | |
|---|---:|---|
| denominator after the pass's own drops | 175 | 6 dropped, 1 unsure |
| the model declined | 87 | 50% |
| the model gave a box | 88 | 50% |
| ...of which held a real instance | 9 | 10% of the boxes |
| **`N+` failure** | **79** | **45%** |

**When this model answers after a clean removal it is wrong nine times in ten.**

## The failure has two mechanisms, and they are not the one that was assumed

The author's own taxonomy over the 182, with the geometry attached:

| | n | share of failures |
|---|---:|---:|
| box still covers the hole | 28 | 35% |
| box moved elsewhere | 51 | 65% |
| (of the failures, essentially the same box) | 20 | 25% |

**No visual evidence at all** (the 35%): a licence plate boxed where plates
belong on a car, a face boxed above the neck that is still there, or simply the
same box again. A prior standing in for pixels.

**Evidence that does not meet the bar** (the 65%): the model looks, finds
something, and accepts it. A desk clock for a removed watch; an arm for a
removed hand; a few dark pixels near a cloud for a kite. Not a wrong prior -- a
threshold far below a person's.

The original claim -- that the model re-predicts the same box -- is the 20
cases, 11% of the denominator. The larger and better-evidenced claim is that the
model cannot take `none` for an answer even when the prompt spells out how to
give it.

## Why this needs all four relations

The two mechanisms are caught by different relations, which is why the two
statistics measured on the same items differ by a factor of nearly three:

| mechanism | caught by | rate |
|---|---|---|
| prior at the old place | `N` alone | 17% |
| everything, including a wrong object accepted | `N+`, which needs `S` | 45% |

`N` alone sees only a box that failed to move. Deciding that a desk clock is not
a watch takes the judge, so the 28-point gap between the two numbers is the part
of the phenomenon that `S` exists to measure. That is the case for `IC` being a
conjunction of relations rather than any single one.
