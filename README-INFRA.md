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
  data/          (empty: the download pipeline is not written yet)
  paths.py       roots, env-overridable
  stats.py       AUROC, clustered bootstrap, kappa
idea91/
  masks.py       RLE, dilation, area/centrality matching quantities
  schemas.py     edits/index, edits/verifier, results/relations
  edits/         window.py (P2), composite.py (check 1c), sampler.py (P3, check 2),
                 inpaint.py (big-LaMa + a non-kill fallback), build.py
  gate/          inputs.py (P5 rows), ladders.py (check 1a/1b), verdict.py (P7),
                 dataset.py, train.py (P6)
  relations/     compute.py (P14, P15), verifier_rules.py (P10)
  analysis/      k1.py (tables/k1.md), k2.py (P15 rates, P16 stop rules)
  instances/     (empty: the SAM 3 adapter is not written yet)
  frontier/      (empty: P17 API client is not written yet)
configs/         models/*.yaml (SPEC Section 1), prompts/*.txt (versioned, hashed)
tests/           129 tests, plus 4 marked `gpu`
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
| B1 | `shared/env`, `PINS.md`, runtime | **partial** -- runtime and probe done; SAM 3, big-LaMa, vLLM and judge weights unpinned (gated / not downloaded) |
| B2 | `shared/data`: pools, benchmarks, `LICENSES.md` | **not started** -- needs the downloads |
| B3 | `idea91/instances`, `idea91/edits` for the K1 pool | **edits done, instances not started** -- window, compositing, samplers, index and the builder are written and tested; the SAM 3 adapter is missing |
| B4a | gate ladders and nulls on the 1b pairs | **logic done** -- ladder operators, floors and verdicts written and tested; not yet run on a real bank |
| B0a | K1 freeze | blocked on B2/B3 |
| B4b | gate classifiers, adversaries, removal-success rows | **classifiers done** (trained end to end on the card in the smoke test); adversary and removal-success wiring pending |
| B5 | K1 full run | blocked |
| B6 | K2 instances and edits | blocked on SAM 3 |
| B7 | `shared/judges` and the verifier | **service and rules done**, weights unpinned |
| B8 | harness additions, Molmo2 path, parity run | **contract done** (configs, prompts, manifests); the backends and parsers are not written |
| B9 | relations, analysis, human-check project | **relations and analysis done**; the Label Studio project is not set up |
| B0b | K2 freeze | blocked |
| B10 | frontier client | **not started** |
| B11 | K2 full run | blocked |

## What blocks the first real run

1. **SAM 3 access** (design 4.1, E91-0 day one) -- gated checkpoint and a licence
   text behind the gate (O7). The host already has `sam2.1_hiera_large.pt`, which
   is the design's own contingency: K1 can start on SAM 2 masks prompted by the
   OpenImages boxes, labelled as such.
2. **big-LaMa weights** -- `VLMG_LAMA_PT` points at `big-lama.pt`; until then the
   only inpainter is OpenCV Telea, which is marked not kill-grade.
3. **GroundingME's instruction and null-box literal**, and Molmo2's native
   pointing instruction and abstention (OPEN-QUESTIONS Q-1, Q-2). Both prompts
   are `UNVERIFIED` and refused by a kill run.
4. **Qwen3-VL's coordinate convention** (Q-3).
5. **Downloads**: OpenImages subset (K1 pool), GroundingME, COCO train2014,
   COCO-Search18. About 60 GB core, 883 GB free on the ext4 volume.
6. **A git repository** -- the freeze points B0a and B0b are tags, and every
   manifest carries `code_sha`, which currently falls back to a source-tree hash
   written as `tree:<hex>` (Q-9).
