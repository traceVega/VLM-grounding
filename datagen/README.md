# datagen — hard-sample data generation (arm A, text-side negatives)

Turns one crowded OpenImages scene (an image with 3–8 instances of one category)
into one master record: a target expression with 4–7 checkable details, a sibling
expression for the instance the base model confuses it with, a one-detail-false
negative, per-instance detail verdicts from a cross-family checker, listener
uniqueness and load-bearing results, the base policy's own boxes and decision-token
probabilities, and the gate flags.  Design: `notes/HARD-SAMPLE-METHODS-2026-09-16.md`
§3.1 and `notes/RL-DESIGN-CANDIDATE-VERIFICATION.md` §1, §9.

## Stages (one model resident per process; every stage is resumable)

| stage | model | env | output | what it does |
|---|---|---|---|---|
| `select` | none | vlmg-env | `scenes.jsonl` | crowded whole-object scenes from the instance bank; candidate hygiene (no nested / duplicate / overlapping candidates, no sliver or border-truncated targets) |
| `write` | Qwen3.5-9B (judge_a) | vlmg-env | `write.jsonl` | target expression (outlined scene + close-up), atomic decomposition, one-detail flip with the plain image, decomposition of the flip, flipped-detail index |
| `policy --which target,neg` | Qwen3-VL-8B-Instruct | vlmg-env | `policy.jsonl` | the GroundingME prompt byte for byte; box, p(null) / p(box) at the decision token, which candidate the box landed on |
| `sibling` | Qwen3.5-9B | vlmg-env | `sibling.jsonl` | expression for the sibling the base boxed (or the largest sibling), same detail template |
| `policy --which sibling` | Qwen3-VL-8B | vlmg-env | `policy.jsonl` | box on the sibling expression |
| `blind` | Qwen3.5-9B, text only | vlmg-env | `blind.jsonl` | can the altered sentence be told from the original without the image (SugarCrepe gate) |
| `check` | Gemma4-12B (judge_b) | vlmg-env | `checker.jsonl` | every detail × every candidate (outlined scene + close-up); the flipped detail on every candidate (zero-satisfier) |
| `check --checker qwen3vl --target-only` | Qwen3-VL-8B | vlmg-env | `checker2.jsonl` | second opinion on the target row (checker disagreement estimate) |
| `listen` | Molmo2-8B | **molmo-env** (transformers 4.57) | `listener.jsonl` | "Point to the {expr}." — uniqueness of the target and sibling expressions; drop-one-detail ablation (load-bearing) |
| `report` | none | vlmg-env | `records.jsonl`, `gates.md`, `review/` | master records, gate table, human review page (copied to `<repo>/review/<run>/`) |

```bash
# WSL, from the repo root
P=~/vlmg-env/bin/python
$P -m datagen.run select  --run test50c --n 50 --per-label 5 --seed 2
$P -m datagen.run write   --run test50c
$P -m datagen.run policy  --run test50c --which target,neg
$P -m datagen.run sibling --run test50c
$P -m datagen.run policy  --run test50c --which sibling
$P -m datagen.run blind   --run test50c
$P -m datagen.run check   --run test50c
$P -m datagen.run check   --run test50c --checker qwen3vl --target-only
~/molmo-env/bin/python -m datagen.run listen --run test50c
$P -m datagen.run report  --run test50c
```

Outputs live on the ext4 volume at `~/vlmg-data/datagen/<run>/`; the review page is
copied to `review/<run>/index.html` on the Windows side (git-ignored).

## Gates (per scene, in `gates.md`)

| gate | meaning | who decides |
|---|---|---|
| `flip_identified` | the negative differs from the target in exactly one decomposed detail | decomposer diff |
| `writer_target_all_yes` / `_at_most_one_no` | the checker confirms the target's details on the target | Gemma4 |
| `checkers_agree_on_target` | Gemma4 and Qwen3-VL agree on ≥ 80% of the target's details | both checkers |
| `siblings_distinct` | every sibling fails at least one target detail | Gemma4 |
| `listener_unique_hit` | Molmo2 points inside the target mask from the expression alone | Molmo2 |
| `zero_satisfier` | the flipped detail holds for no candidate (unclear counts as not-no) | Gemma4 |
| `base_boxes_on_neg` | the base still boxes a candidate on the negative with p(box) ≥ 0.99 (hard negative) | policy |
| `base_wrong_on_target` | the base boxes a sibling on the target expression (hard positive) | policy |
| `blind_judge_fooled` | a text-only judge cannot tell the altered sentence | Qwen3.5 text only |
| `negative_usable` | all of: flip identified, writer confirmed, listener unique, zero-satisfier, base boxes on neg | — |
| `positive_hard` | base wrong on target, listener unique, writer confirmed | — |

## Known limits (2026-09-21)

- **Resolution.** The pool images are 1024 px on the long side (0.75 MP) against
  GroundingME's median 2,250 px (3.4 MP).  Details on ~100 px objects are
  hallucinated by the writer and unreadable by the checker; the close-up crop
  helps but cannot add pixels.  Flickr originals (median 2,832 px, ~80% of links
  alive) are the fix: re-fetch and re-run SAM 3 on the crowded subset.
- **Checker strictness.** Gemma4-12B answers "no" to roughly half of the target's
  details even when they hold (seen by eye on test50); the writer also
  hallucinates some.  `checker2.jsonl` bounds the disagreement; a frontier API
  checker is the planned upgrade.
- **Molmo2 under transformers 5.16** loads with four shims but generates
  degenerate text with no points; vLLM 0.28 cannot start on this WSL host ("UVA
  is not available").  The listener therefore runs from `~/molmo-env`
  (transformers 4.57.1, torch 2.14 cu130).
- **Policy model.** The gate runs on the 8B; the training target is the 4B,
  which is not downloaded yet.  Gate 3 must be re-run on the 4B before training.
- The instance bank's `n_head_noun_instances` column is 1 everywhere (K1 pool);
  crowding is counted from the `same_class_other_instance` rows.

## Added 2026-09-22

- `check --checker gemini` (`checker_api.py`): the same per-detail verification through Gemini 3.1 Pro (`GEMINI_API_KEY`, WSL `~/.vlmg-secrets` sourced from `~/.profile`); writes `checker_gemini.jsonl`. `gemini-flash` = gemini-3.8-flash. ~$0.13/scene at one call per detail; batch the details per candidate before scaling.
- `report --primary-checker gemini`: gates and page from Gemini's matrix; outputs are suffixed (`records-gemini.jsonl`, `gates-gemini.md`, `review/<run>-gemini/`).
- New gate `zero_satisfier_sentence` (+ `negative_usable_sentence`): a candidate only counts as satisfying e_T⁻ if the flipped detail AND every other detail are yes/unclear on it.
- `datagen/reviews/<run>.json`: reviewer buckets + per-scene notes; `report` groups the page by bucket when the file exists.
- Every review card carries a contact sheet of each candidate's close-up (`img/<id>_cands.jpg`); review at that resolution, not on the 1024 px overview.
- Flip guards: whole-word substitution, NEW values containing OLD are rejected, an unchanged "flipped" clause marks `flip_failed`.
