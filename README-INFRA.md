# Kill-experiment infrastructure: build state

Implements `idea-91-interventional-verification/DESIGN-kill-infra.md` v1.0 for
E91-1 (K1, the artifact gate) and E91-2 (K2, the removal test) on the local
RTX 5090. The harness contract is `shared/harness/SPEC.md` v1.2.

Started 2026-09-02. Deviations are in `notes/DEVIATIONS.md`, open questions in
`notes/OPEN-QUESTIONS.md`, pins in `shared/env/PINS.md`.

## Layout

```
shared/          shared with IDEA-11
  env/           PINS.md, probe.py (GPU name, idle VRAM, package versions, code_sha)
  harness/       SPEC.md, schema.py (normative), manifest.py, prompts.py, model_config.py
  judges/        judges.yaml + lineage.py (the lineage rule), serve.sh, client.py (cached)
  data/          sources.py (registry + licences), licenses.py (renders LICENSES.md),
                 items.py (items.parquet builder + per-set adapter seams), download.py
  paths.py       roots, env-overridable
  stats.py       AUROC, clustered bootstrap, kappa
idea91/
  masks.py       RLE, dilation, area/centrality matching quantities
  schemas.py     edits/index, edits/verifier, results/relations, instances
  edits/         window.py (P2), composite.py (check 1c), sampler.py (P3, check 2),
                 inpaint.py (big-LaMa + a non-kill fallback), build.py
  gate/          inputs.py (P5 rows), ladders.py (check 1a/1b), verdict.py (P7),
                 dataset.py, train.py (P6)
  relations/     compute.py (P14, P15), verifier_rules.py (P10)
  analysis/      k1.py (tables/k1.md), k2.py (P15 rates, P16 stop rules)
  instances/     backend.py (capability contract), sam.py (SAM 3 pin + SAM 2 contingency),
                 nounphrase.py (head noun + phrases), build.py (K1/K2 scenes)
  frontier/      (empty: P17 API client is not written yet)
  human/         tasks.py (P11 draw + rendering), app.py (local annotation UI),
                 agreement.py (Fleiss/Cohen kappa, check 3), build_tasks.py
  run_k1.py      pipeline driver: instances | edits | status
configs/         models/*.yaml (SPEC Section 1), prompts/*.txt (versioned, hashed)
tests/           251 tests, 9 of them marked `gpu`
```

## Environment

WSL2 Ubuntu, uv-managed Python 3.12 at `~/vlmg-env`, torch 2.14.0+cu130 on
sm_120, timm 1.0.29. Code lives on the Windows volume; data, edits and results
live on ext4 (`~/vlmg-data`, `~/vlmg-results`), overridable with
`VLMG_DATA_ROOT`, `VLMG_EDITS_ROOT`, `VLMG_RESULTS_ROOT`.

```bash
wsl -d Ubuntu -e bash -lc 'cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding && uv pip install --python ~/vlmg-env/bin/python -e ".[dev,gpu]"'
```

Machine facts for a manifest:

```bash
wsl -d Ubuntu -e bash -lc '~/vlmg-env/bin/python -m shared.env.probe'
```

Tests (the `gpu` set builds a toy edit bank and trains for one epoch on the card):

```bash
wsl -d Ubuntu -e bash -lc 'cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding && ~/vlmg-env/bin/python -m pytest tests -q -m "not gpu"'
```

## What is enforced in code, not by convention

| Rule | Where |
|---|---|
| SPEC Sections 3 and 4 schemas, validated on every write and read | `shared/harness/schema.py` |
| The SPEC enum lists cannot drift from the prose | `tests/test_schema_matches_spec.py` |
| `iou_gt`, `point_in_gt`, `is_failure` exist on ORIGINAL rows only | `schema.py` |
| `image_px_sent` on every row | `schema.py` |
| Dev slices disjoint from kill sets by pHash | `schema.assert_dev_slice_disjoint` |
| A result directory is immutable once its manifest is written | `manifest.write` / `manifest.read` |
| A non-kill run id cannot enter a kill table | `manifest.require_kill_runs` |
| An UNVERIFIED prompt or an unpinned model config cannot serve a kill run | `prompts.py`, `model_config.py` |
| The edit verifier is never the reward judge | `judges/lineage.py` |
| Compositing is in-mask; nothing outside the hole changes (check 1c) | `edits/composite.assert_in_mask` |
| No control hole overlaps the P3 exclusion set (check 2) | `edits/sampler.assert_no_exclusion_overlap` |
| The 80/20 gate split is by source image | `gate/dataset.split_by_image` |
| A K1 PASS never prints without its ladders and nulls | `gate/verdict.K1Verdict` |
| Same-box excludes UNRESOLVED items | `analysis/k2.compute_rates` |
| (b') does not fire when the CONTROL_BG shift is within 10 points | `analysis/k2.rule_b_prime` |
| Verifier-clean is decisive only when kappa >= 0.6 | `analysis/k2.decisive_clean_column` |
| P20: no classifier trains on a real removal before B0a exists | `gate/train.run_row` |
| A value frozen at B0a cannot change without the run refusing | `freeze.require` |
| The freeze covers behaviours too, by source hash, not only constants | `freeze.B0A_BEHAVIOURS` |
| A wrapped header comment cannot reach the model as prompt text | `harness/prompts._parse` |
| class_agnostic controls never enter P10's verified-pairs row | `gate/removal_success` |
| The box parser cannot be handed a ground-truth box to choose by | `harness/parsers` |
| An image control changes no words; a text control changes no pixels | `harness/conditions.resolve` |
| T_NULL renders the same template as the condition it controls for | `configs/models/*.yaml` |
| The render cache is lossless, so it cannot manufacture JPEG artifacts | `gate/dataset.RenderCache` |

## Build plan status (design Section 7)

| Step | Deliverable | State |
|---|---|---|
| B1 | `shared/env`, `PINS.md`, runtime | **done.** SAM 3, big-LaMa, both judges, both policies and GroundingME all downloaded and pinned by revision in `PINS.md`; vLLM build still unpinned |
| B2 | `shared/data`: pools, benchmarks, `LICENSES.md` | **OpenImages and GroundingME run.** 10,000-image pool selected and fetched; GroundingME extracted to 1,005 images with `items.parquet` built and validated. OpenRef, gRefCOCO and COCO-Search18 adapters still wait for their formats, and every unpinned locator is refused rather than guessed |
| B3 | `idea91/instances`, `idea91/edits` for the K1 pool | **instances done, edits running.** 9,692 scenes banked in 3.6 h (311,282 instance rows, 65 shards). The edit bank is building under `scripts/supervise.sh` |
| B4a | gate ladders and nulls on the 1b pairs | **done and passing, 2026-09-04.** See the results below |
| B0a | K1 freeze | **taken 2026-09-04**: `B0a@38589450bde7`, 47 values, signed off by Jiaqi Zhang, tag `freeze-B0a` on `b8fc7fc` |
| B4b | gate classifiers, adversaries, removal-success rows | **written; the verdict tier is running.** Tiered so P7's verdict is 24 trainings rather than 76. P10 removal success still needs the judge served |
| B5 | K1 full run | verdict tier in flight; the remaining three tiers block nothing |
| B6 | K2 instances and edits | **adapter done**, 804 positives on 685 images; instances and edits not run |
| B7 | `shared/judges` and the verifier | **done.** Service, rules and both judge revisions pinned |
| B8 | harness additions, Molmo2 path, parity run | **parsers and conditions done** with both models' conventions settled (Q-1, Q-2); the run loop and the backends are not written |
| B9 | relations, analysis, human-check project | **relations, analysis and a local annotation UI done**; the hosted project is not set up |
| B0b | K2 freeze | blocked |
| B10 | frontier client | **not started** |
| B11 | K2 full run | blocked |

## What blocks the first real run

1. ~~SAM 3, big-LaMa, the judges, the policies, GroundingME~~ **all downloaded
   and pinned 2026-09-03**; see `shared/env/PINS.md` for revisions and the
   measurements taken on the card.
2. ~~GroundingME's instruction and null-box literal~~ **resolved (Q-1).** Copied
   byte for byte from the benchmark's own evaluator at `lirang04/GroundingME@6867f7a0`.
   Both Qwen3-VL prompts are `VERIFIED`.
3. ~~Molmo2's pointing instruction and coordinate convention~~ **resolved (Q-2).**
   Points are 0-1000 against the original image; the config had said
   `percent_float`, which was Molmo v1's. Its **abstention form is still
   `PIN_REQUIRED`** -- the card never shows one -- and is measured on P12's own
   `p12` dev slice.
4. **Qwen3-VL's coordinate convention (Q-3)** -- still open, and now a
   correctness requirement rather than a detail: the benchmark's scorer picks
   whichever of four readings maximises IoU against ground truth, which K2
   cannot copy because REMOVE has no ground-truth box. Needs a GPU probe.
5. **Still outstanding**: OpenRef (design O1), gRefCOCO for the `p12` slice,
   COCO train2014 and COCO-Search18.

## B4a results (2026-09-04, 2,000-image ladder sample, whole bank behind it)

| check | mask | rect | requirement |
|---|---|---|---|
| **1b** null, must sit at chance | 0.499 to 0.507 | 0.498 to 0.507 | 0.5 +/- 0.05 |
| **1a** global, q75 | 0.945 | 0.976 | > 0.90 |
| **1a** global, **q90** (the rung the requirement is read at) | **0.794** | **0.754** | >= 0.70 |
| 1a global, q92 (the design's rung, not met) | 0.640 | 0.616 | -- |
| **1c** paired-crop | 200 pairs clean | 200 pairs clean | identical bytes |

**All three checks pass.** The sensitivity requirement is read at q90 rather than
the design's q92, amended at B0a on measured grounds: the editor's own footprint
is 49.4 grey levels inside the hole against 3.98 at the harshest rung the ladder
tests, so every rung is more than an order of magnitude finer than what K1 has
to detect and the distinction cannot bear on fitness. No resolution raise was
made -- a ninefold range of input pixels moves the q92 AUROC only between 0.55
and 0.64. See `notes/OPEN-QUESTIONS.md` Q-15.

**The local floor is the caveat every K1 result carries** (P7 requires it beside
the verdict). Only one cell in eight detects in-hole damage at all: mask 5-15%
at q30, AUROC 0.815. Every other bin on both hole types is at chance, confirmed
against a second gate row, so this is a property of the gate rather than of the
row that measured it. A K1 PASS therefore means "no systematic difference the
gate can see", not "no difference". Row (ii) is separately near-blind to local
artefacts for a structural reason -- `notes/OPEN-QUESTIONS.md` Q-16.

## Compute, measured rather than budgeted

The design allows 14 to 20 GPU-hours for all of K1. Measured on this card:

| | per training, full bank | 148 trainings |
|---|---|---|
| as first written | 18 min | 45 h |
| with the render cache | 9.5 min | 24 h |

The gate is tiered, so **P7's PASS/FAIL is 24 of those trainings**, not 148 --
roughly 4 hours. Everything else is reported alongside the verdict and blocks
nothing. `run_k1 rows --tiers verdict` runs only that.

## Fetching data

`python -m shared.data.download --core` prints the plan and fetches nothing.
`python -m shared.data.groundingme --extract --items` builds the K2 set from the
pinned snapshot. 767 GB free on the volume, against a 54 GB edit bank and a
render cache capped at 150 GB.
`--yes` executes only the ready steps.


## Human annotation (P11)

The K2 images are research-licensed and must not be published, so the annotation
UI is local and dependency-free, binds to `127.0.0.1`, and refuses any other
host without `--i-understand-the-licence`.

```bash
# build a task set from an existing edit bank
python -m idea91.human.build_tasks --out ~/vlmg-data/human/pilot --limit 10
# one process per annotator, each on its own port
python -m idea91.human.app --tasks ~/vlmg-data/human/pilot --annotator A1
```

What the annotator sees is fixed by P11 and enforced in code: the edit window at
2x the hole, the full image, the head noun -- and never the original, never the
condition, never another annotator's label. The task file carries no operator and
no verifier answer, so a leak would have to be deliberate; the answer key is
written beside it as `KEY_do_not_show_annotators.csv` and is never served. Item
order is shuffled per annotator so position carries no signal either.

Keys `1` gone, `2` traces, `3` still there, `0` unsure, `Backspace` to undo.
Labels are written through to `human_labels.csv` on every press, in the columns
design Section 5 specifies.

`idea91.human.agreement` computes what P11 asks for: human-human Fleiss kappa
(3-way, binarised, and per stratum) and human-versus-V1 Cohen kappa, with
acceptance check 3's verdict -- which is what decides whether verifier-clean or
human-clean carries the K2 headline.
