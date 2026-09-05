# The GroundingME removal set, built and ready to review

Built 2026-09-05 overnight, unattended, four stages supervised on progress.
Exploratory, `--non-kill`: nothing here may enter a kill table or change P8 to
P21. The K1 bank is untouched -- these edits live in their own root,
`~/vlmg-data/edits_gme`, because the K1 bank is frozen at B0a.

| stage | result |
|---|---|
| REMOVE edits, SAM 3 box prompt on the window + big-LaMa | **333 / 333** (322 mask holes, 8 rectangular, flagged) |
| REMOVE inference, Qwen3-VL-8B, p(null) recorded | **333 / 333** |
| Chinese rendering of each expression, Qwen3.5-9B | **333 / 333** |
| review panels | **333** |

Items are the ones whose ORIGINAL box was already correct, which is the only
frame in which "it stopped looking" can be said at all: Discriminative 124,
Limited 123, Spatial 86.

## English and Chinese

Every experiment renders its prompt from the English `expr`, GroundingME's own
words, per P12. The Chinese in `tables/gme_expr_zh.jsonl` has exactly one
consumer, the review page, and reaches no model. A translated prompt would be a
different prompt.

## Provisional, before the review filters anything

Failed removals and items where something else now satisfies the description are
both still in here, so these are not results.

| dimension | n | declined | same box | median p(null) |
|---|---:|---:|---:|---:|
| Discriminative | 124 | 11% | 2% | 5.0e-15 |
| Limited | 123 | 17% | 6% | 3.3e-13 |
| Spatial | 86 | **0%** | 8% | 7.7e-15 |

Two things to watch once the review has cleaned the denominator. Abstention is
far below the 50% the class-label removals produced and nowhere near the 100%
these items call for, and p(null) sits at 1e-13 to 1e-15 rather than the 0.62 of
the class-label set -- the direction the Rejection row already pointed. And
`Spatial` declines on none of its 86, the stratum whose expressions name
landmarks that survive the removal.

If those hold after the review, the reading is that the abstention signal is a
property of short expressions, and `IC`'s necessity relation is weak where it
would be needed most.

## Reviewing

    bash scripts/serve_gme_review.sh      # brings it back up after a WSL reap

http://127.0.0.1:8902 -- keys `1` gone, `2` still there, `3` something else now
matches, `0` unsure, left arrow undoes. Panels are grouped by dimension and the
header names it. Labels append to `~/vlmg-data/gme_review/gme_labels.csv` on
every press, so a closed tab costs nothing.
