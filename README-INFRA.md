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
configs/         models/*.yaml (SPEC Section 1), prompts/*.txt (versioned, hashed)
tests/           200 tests, 7 of them marked `gpu`
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

## Build plan status (design Section 7)

| Step | Deliverable | State |
|---|---|---|
| B1 | `shared/env`, `PINS.md`, runtime | **SAM 3 and big-LaMa done and verified on the card**; vLLM build and judge weights still unpinned |
| B2 | `shared/data`: pools, benchmarks, `LICENSES.md` | **written, not run** -- registry, licence renderer, `items.parquet` builder and the download plan are done and tested; each set's annotation adapter waits for its data, and every unpinned locator is refused rather than guessed |
| B3 | `idea91/instances`, `idea91/edits` for the K1 pool | **written, not run** -- edits and the K1/K2 scene builders are done and tested against a fake segmenter; the SAM 3 adapter runs on transformers' `Sam3Model`/`Sam3Processor` with its plumbing unit-tested against fakes, and SAM 2 is wired as the design's contingency, which stamps `kill_grade=False` on everything it makes. Waiting on an HF token |
| B4a | gate ladders and nulls on the 1b pairs | **logic done** -- ladder operators, floors and verdicts written and tested; not yet run on a real bank |
| B0a | K1 freeze | blocked on B2/B3 |
| B4b | gate classifiers, adversaries, removal-success rows | **classifiers done** (trained end to end on the card in the smoke test); adversary and removal-success wiring pending |
| B5 | K1 full run | blocked |
| B6 | K2 instances and edits | blocked on an HF token, then the GroundingME adapter |
| B7 | `shared/judges` and the verifier | **service and rules done**, weights unpinned |
| B8 | harness additions, Molmo2 path, parity run | **contract done** (configs, prompts, manifests); the backends and parsers are not written |
| B9 | relations, analysis, human-check project | **relations and analysis done**; the Label Studio project is not set up |
| B0b | K2 freeze | blocked |
| B10 | frontier client | **not started** |
| B11 | K2 full run | blocked |

## What blocks the first real run

1. ~~SAM 3~~ **done 2026-09-03.** `facebook/sam3` @ `3c879f39826c`, downloaded
   and verified: concept prompt IoU 0.99, absent concept returns nothing, box
   prompt IoU 0.99, ~0.2 s per prompt, 2.13 GB VRAM.
2. ~~big-LaMa weights~~ **done 2026-09-03.** Apache-2.0, downloaded, generator
   vendored at `advimman/lama@786f5936`, verified: 0.029 s per 512x512 fill,
   0.45 GB VRAM. Set `VLMG_LAMA_DIR` to the unpacked `big-lama` directory.
3. **GroundingME's instruction and null-box literal**, and Molmo2's native
   pointing instruction and abstention (OPEN-QUESTIONS Q-1, Q-2). Both prompts
   are `UNVERIFIED` and refused by a kill run.
4. **Qwen3-VL's coordinate convention** (Q-3).
5. **Downloads still outstanding**: the 10,000 K1 pool images (~3 GB, pinned and
   selected, awaiting a go), GroundingME, COCO train2014, COCO-Search18.

Resolved since: the repository exists (branch `main`), so the freeze points have
something to tag and manifests carry a real `code_sha`.

## Fetching data

`python -m shared.data.download --core` prints the plan and fetches nothing.
Today 13.2 GB is fetchable (COCO train2014 and the 2017 annotations) and 44.2 GB
is blocked on unread licences or unpinned locators; 879 GB free on the volume.
`--yes` executes only the ready steps.
