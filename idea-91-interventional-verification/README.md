# IDEA-91: Interventional grounding verification

Working folder for the direction "Right box, wrong reason: treat the model's own region as a hypothesis and test it by editing the image."

Created 2026-09-02 from the team's Phase 3 output. Nothing in `../team/` is modified from here; this folder holds the execution plan, code, results and notes for this one direction.

## Source of record

- Idea file: `../team/ideas/IDEA-91-interventional-grounding-verification.md` (SURVIVING; Skeptic final PROMISING; Verifier grade A) and Researcher1's edit-server section `../team/ideas/IDEA-91-R1-section-edit-server.md`
- Shared notation (relations, removal-region rule, duplicates rule): `../team/landscape/r3-history.md` Section 7
- Edit-bank engineering and the artifact-leakage protocol: `../team/landscape/r2-practical.md` Sections 8.2 and 8.10
- Critique: `../team/critiques/IDEA-91-skeptic.md`
- Verification: `../team/verifications/IDEA-91-verifier.md`
- Budget rows and kill order: `../team/PLAN.md` Sections 2 (steps 1, 2, 6) and 4
- Independent evaluation with the cross-idea kill order (this idea is KE-2, KE-3 and KE-6 there): https://claude.ai/code/artifact/e81d2ae1-c577-4baa-8240-4b44ba5c14f9

## The question

IoU against a reference box cannot tell a model that found the referent from one that found the only salient object. Anchor a realistic edit on the model's own predicted region, pair it with an artifact-matched control edit elsewhere, and require the localization output itself to respond: invariance (edits outside the expression's support must not move the box), necessity and necessity-plus (removing the region must yield `none` or a valid fallback), sufficiency (a frozen judge confirms the expression from the crop alone), and text-side controls. Their mean is a label-free interventional consistency score that can be computed on images uploaded after every model's training cutoff. Three contributions in a fixed order: the audit, the metric, and only if a matched-compute comparison justifies it, the training signal.

## Decision rules carried over from the idea file

- K1 (first, 2 GPU-hours): referent-removed versus control-edited images must not be separable by a ResNet-18 or ViT-S/16 with the referent location hidden; AUROC above 0.6 stops everything downstream until the editor or control design changes.
- K2 (10 GPU-hours): stop the training half if the two 8B models already output `none` on more than 60% of removals, or hallucinate a box on more than 80% of removals while abstaining on more than 30% of controls; keep only the rejection half if same-box re-prediction on removals is below 20%.
- K3 (about 500 dollars): stop the metric contribution if the score's AUROC for human "box correct" is below 0.75 or kappa between the judge and people is below 0.5.
- K4 (inference only): stop the benchmark claim if the item-level AUROC is below 0.75, if the score does not beat the judge alone by 0.03 (0.05 on the judge-wrong subset), or does not beat a two-seed disagreement predictor by 0.05.
- K5 (300 to 400 GPU-hours): drop the training contribution if any cheap counterfactual (gray mask, blur-ghosting, attention mask) is within 2 points on natural-pair necessity-plus or GroundingME rejection.

## What to run first

`E91-0` (setup: SAM 3 access on day one) then `E91-1` and `E91-2` on the local 5090, about two days of GPU time plus 100 dollars for the human removal check and 50 to 150 dollars of API calls for the frontier subset. Nothing is trained before the audit exists. The full list with the local-versus-cloud split and machine specs is in `EXPERIMENTS.md`.

## Proposed layout

- `EXPERIMENTS.md`: the ordered experiment list, where each runs, and what each needs
- `configs/`: editor settings, judge lineage assignments (judge A for rewards, judge B held out, generator of a different lineage from the judge), benchmark loaders
- `edits/`: the edit-bank builder (SAM 3 instances, LaMa removals, 4B-editor swaps, copy-paste insertions, matched controls, verifier outcomes), keyed by image and instance
- `relations/`: the score as code: invariance, necessity, necessity-plus with the hole masked from the judge, sufficiency with the text-only prior, text-side controls, the duplicates rule
- `results/`: the audit table, one row per model and item, with REDUNDANT, UNRESOLVED and box-inpaint flags
- `notes/`: decisions, licence checks (SAM 3, COCO-Search18, Objects365), and any deviation from the idea file, dated
