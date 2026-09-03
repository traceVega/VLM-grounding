# IDEA-11: The latent grounding gap

Working folder for the direction "MLLMs localize better than they say, and why, and can the text decoder be taught to read its own map?"

Created 2026-09-02 from the team's Phase 3 output. Nothing in `../team/` is modified from here; this folder holds the execution plan, code, results and notes for this one direction.

## Source of record

- Idea file: `../team/ideas/IDEA-11-latent-grounding-gap.md` (SURVIVING; Skeptic final PROMISING; Verifier grade B)
- Critique: `../team/critiques/IDEA-11-skeptic.md`
- Verification: `../team/verifications/IDEA-11-verifier.md`
- Budget rows and kill order: `../team/PLAN.md` Sections 2 (step 3) and 4
- Independent evaluation with the cross-idea kill order (this idea is KE-1 there): https://claude.ai/code/artifact/e81d2ae1-c577-4baa-8240-4b44ba5c14f9

## The question

Coordinate-emitting MLLMs are scored on the boxes they write, but the same forward pass carries an implicit localization signal readable from attention, gradients or block ablation. The paper is an audit (emitted versus read-out localization, same model, same forward pass, across REC and GUI, at 8B and 32B, with every read-out's label budget stated and a matched-label supervised head as the control) plus a cause test (decoding artefact, perception limit, or a trainable decoder). Training is a gated minimum at 4B, run only after the core and only if two gates pass.

## Decision rules carried over from the idea file

- Kill (Section 6): on Qwen3-VL-8B and 32B, over emitted-box failures on GroundingME, OpenRef and the Ref-L4 small bin, stop the idea if at both scales the rescue rate is below 15% and the 0.15-IoU-gap rate is below 20%.
- Stage 4 stop: the matched-label supervised head within 2 IoU points of the best read-out, or Gate 1 headroom below 0.05 IoU on the training pool.
- Gate 2: presence-probe referent-mask drop minus control-mask drop must reach AUROC 0.7 for groundability, else the necessity term is replaced by IDEA-91's edit-pair negatives.
- Stage 5 stop: agreement does not beat coordinate-token likelihood on AURC on at least two of four benchmarks.

## What to run first

`E11-0` (setup and hooks) then `E11-1` and `E11-2` on the local 5090, about two days of GPU time, then `E11-3` on a cloud card for one afternoon to confirm at 32B. The full list with the local-versus-cloud split and machine specs is in `EXPERIMENTS.md`.

## Proposed layout

- `EXPERIMENTS.md`: the ordered experiment list, where each runs, and what each needs
- `configs/`: one file per model with its coordinate convention, prompt and max_pixels, plus one loader per benchmark
- `scripts/`: harness, eager-attention hooks, read-outs, probes, heads
- `results/`: per-experiment outputs; keep 64 by 64 read-out masks and per-item tables, never raw attention maps
- `notes/`: decisions and any deviation from the idea file, dated
