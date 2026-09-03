# Verifier report: IDEA-22 (does grounding accuracy buy task success: grounder swap and error injection inside computer-use agents)

Verifier, 2026-09-02 (Phase 2, after round 1). File checked: `ideas/IDEA-22-grounding-utility-decoupling.md` (status REVISED, author Researcher2). **P** = primary source read (arXiv abstract or HTML body, official repo or blog); **S** = secondary. Claim ids refer to `verifications/claims-log.md` (V-084, V-151, V-161 to V-165, V-180 to V-191).

## 0. Summary verdict

- **Environment and agent facts: VERIFIED.** OSWorld has 369 tasks across Ubuntu, Windows and macOS (V-180); its repo is Apache-2.0 with VMware, VirtualBox, Docker, AWS, Modal and Daytona providers and a multi-environment runner (V-162); OSWorld-Verified is a documented in-place upgrade (300+ fixes, AWS, up to 50 parallel environments; the blog does not restate the task count) (V-181); AndroidWorld has 116 tasks across 20 apps (V-182); Jedi's GPT-4o-planner setup uses screenshots only and its 5.0 to 24.0 to 27.0 to 51.0 ladder is as quoted (V-161, V-084); UI-Vision, SafeGround and the GUI uncertainty benchmark exist as described (V-191, V-151).
- **REFUTED: §1 "The one verified data point is Jedi ... three grounders and no control".** At least three other papers vary the grounder under a fixed planner and measure closed-loop task success: Agent S2 Figure 6 (UI-TARS-7B-DPO, UI-TARS-72B-DPO, UGround-V1-7B, Claude-3.7-Sonnet as grounders, OSWorld 15-step; V-184); MMBench-GUI Finding 2 (fixed GPT-4o planner, grounders UGround-V1-7B to UI-TARS-1.5-7B on online L3 tasks across five platforms: "improving localization alone led to a 2.8x (Delta=17.25) increase in SR" versus 1.15x (Delta=3.58) from a planner swap, and the figure caption "Task success grows roughly linearly with visual-grounding accuracy"; V-185); GroundCUA Table 4 (fixed o3 planner on OSWorld-Verified: GroundNext-3B 50.6, Jedi-7B 51.0, OpenCUA-72B 46.1; V-186). The Skeptic's framing (map §0, §4 item 4) that no GUI-grounding paper shows the grounding-to-success link in a controlled way is therefore also contradicted by MMBench-GUI, at least at the level of a small ladder. The ladder half of IDEA-22 is a replication with more rungs and a shared harness; the file's own decision to make the ladder descriptive (22-M1) already fits this, but §1 and §4 must be rewritten.
- **Error-injection half: no parametric grounder-error injection study found, but a close neighbour exists and must be positioned.** GUI-RobustEval (2605.29447, May 2026) builds 1,216 executable test cases from real failed trajectories of 12 agents on OSWorld: experts locate the root-cause step, repair the prefix, and the error-free prefix plus the erroneous step are replayed in live OSWorld before the agent takes over at depths 0, 1, 3 or 5; 11 error types including wrong UI elements and incorrect parameters; metrics are error-awareness rate and post-error success rate (V-187). It differs from IDEA-22 in every dimension the idea cares about (real errors at fixed depth, no perturbation magnitude or rate sweep, no isolation of one grounding property, no grounder swap, no abstention arm, no metric-to-success surface), so the delta survives, but 22-N1 must be answered with this citation. Adjacent and not overlapping: OSWorld-Noisy (environment-side interruptions; V-188), GUI-Perturbed (offline grounding perturbations; V-189), "Do GUI Agents Believe Their Eyes?" (single-channel interventions, mostly single-step; V-190), "Naive Visual Memory is Not Enough" (observational taxonomy with a grounding-error mode; V-190), GUI-Reflection (training-side error construction; V-190).
- **Failure-attribution precedents (22-N1(a)):** the OSWorld abstract names "GUI grounding and operational knowledge" as the major deficiencies (VERIFIED as a precedent at abstract level; V-180); Aguvis has a two-stage grounding-then-planning pipeline but no stage-wise failure attribution in its abstract (PARTIALLY; V-183).
- **Researcher2's three requests: (1) VERIFIED, (2) VERIFIED, (3) VERIFIED with a correction** (Section C): Agent S2 does report grounder ablations, so §5's description of it as only "a baseline agent" is incomplete.
- **Section 9 [LIKELY] items: all resolved** (Section D). One useful fact for the planner choice: the Qwen3.5-27B card reports OSWorld-Verified 56.2 and AndroidWorld 64.2 as an agent (V-077), so the open planner the file names is not in the GPT-4o-era planner-bound regime.

## A. Novelty test (three closest papers)

| Paper named by author | Says what the author says? | Does the stated delta hold? |
|---|---|---|
| Jedi / OSWorld-G (2505.13227) | Yes, P (body): planner (GPT-4o or o3) sees the screenshot and history, emits a natural-language instruction; Jedi predicts the action "purely from visual perception"; OSWorld 5.0 to 24.0 (3B) to 27.0 (7B), 51.0 with o3 (V-161, V-084). | Yes for injection and recoverability. Not for "eight grounders" as a first: see Section B. |
| GTA1 (2507.05791) | Yes, P (abstract): test-time scaling with a judge over sampled action proposals; RL-trained grounder (V-164). | Yes. |
| GUI-G1 (2505.15810) | Mostly: hit and IoU rewards pull box size in opposite directions is PARTIALLY supported on magnitudes (V-061). | Yes: no downstream measurement there. |

## B. Fixed-planner grounder comparisons the file missed

| Paper | Design | Numbers | Injection? |
|---|---|---|---|
| Agent S2 (2504.00906), P (body) | Figure 6: planner fixed, grounder swapped among UI-TARS-7B-DPO, UI-TARS-72B-DPO, UGround-V1-7B, Claude-3.7-Sonnet; OSWorld 15-step | "smaller specialist models ... can outperform large generalist models like Claude-3.7-Sonnet"; Mixture-of-Grounding ablation 27.69 to 30.77 (15-step), 33.85 to 38.46 (50-step) (V-184) | No |
| MMBench-GUI (2507.19478), P (body) | Finding 2 on L3 online single-app tasks (Windows, macOS, Linux, Android, Web): planner fixed (GPT-4o) with grounders UGround-V1-7B to UI-TARS-1.5-7B; then grounder fixed (UI-TARS-1.5-7B) with planners varied | grounder improvement 2.8x (Delta=17.25); planner improvement 1.15x (Delta=3.58); "Task success grows roughly linearly with visual-grounding accuracy" (V-185). The figure's per-model points could not be extracted from the HTML; the ladder has at least two rungs. | No |
| GroundCUA / GroundNext (2511.07332, ICLR 2026), P (body) | Table 4: o3 planner fixed on OSWorld-Verified; grounders GroundNext-3B, Jedi-7B, OpenCUA variants, proprietary APIs | GroundNext-3B 50.6, Jedi-7B 51.0, OpenCUA-72B 46.1 (V-186) | No |
| Jedi (2505.13227), P | above | above | No |

Consequence: "grounder swap under a fixed planner" is an established measurement (four papers, 2025 to 2026). The novel part of IDEA-22 is (i) the parametric injection of one grounding property at a time, (ii) the empirical-error replay arm, (iii) the abstention-versus-hallucination comparison under three re-plan policies, and (iv) reading which offline metric predicts success. §1, §4 and §8 ("Jedi already showed grounding helps") should be rewritten around MMBench-GUI's "roughly linear" claim as the thing to test with controlled error types, since a linear relation across two or three grounders cannot separate hit rate, IoU, identity errors and refusal.

## C. Researcher2's Section 9 requests

| Request | Verdict | Evidence |
|---|---|---|
| (1) OSWorld task count and current provider list | **VERIFIED.** 369 tasks (abstract, V-180); providers VMware / VMware Fusion, VirtualBox, Docker (KVM), AWS, Modal, Daytona; `run_multienv.py`; Apache-2.0 (README, V-162). OSWorld-Verified: AWS with up to 50 environments; task count not restated (V-181). | P |
| (2) Jedi 5.0 to 27.0: accessibility tree or screenshots only | **VERIFIED: screenshots only.** "purely from visual perception without access to the GUI's underlying code or APIs"; 1080p screenshots (V-161). | P |
| (3) Agent S2 arXiv id and grounder ablations | **VERIFIED id (2504.00906) and it DOES report grounder ablations** (Figure 6, four grounders under a fixed planner) plus a Mixture-of-Grounding ablation (V-184). Update §5. | P |

## D. Section 9 [LIKELY] items and other checked claims

| Claim | Verdict | Evidence |
|---|---|---|
| OSWorld task count 369 | VERIFIED | V-180 |
| OSWorld licence Apache-2.0 | VERIFIED | V-162 |
| OSWorld-Verified exists; checkers | VERIFIED (exists, in-place upgrade) / PARTIALLY (369 retained not stated) | V-181 |
| Agent S2 id | VERIFIED | V-163, V-184 |
| OSWorld 2404.07972 and AndroidWorld 2405.14573 ids | VERIFIED | V-180, V-182 |
| AndroidWorld 116 tasks, 20 apps, emulator | VERIFIED (Pixel 6 / API 33 from README not re-checked) | V-182 |
| UI-Vision availability | VERIFIED (licence-permissive, 83 apps, open-sourced) | V-191 |
| OSWorld run time with 10 environments "within 1 hour" | PARTIALLY | README claim for the maintainers' agent (V-162); OSWorld-Verified blog: "10+ hours to minutes" on AWS with up to 50 environments (V-181). Your agent's wall time depends on planner latency. |
| Aguvis (2412.04454) as stage-wise analysis precedent | PARTIALLY | two-stage training verified; no failure attribution in the abstract (V-183) |
| OSWorld's own error analysis (2404.07972) | VERIFIED (abstract level) | "GUI grounding and operational knowledge" (V-180) |
| ScreenSpot-Pro: point inside box; 0.07% targets; 1,581 items | VERIFIED | V-060, V-173 |
| OSWorld-G: 564 samples incl. 54 refusal cases; about 71 top | VERIFIED | anchors A1 |
| Grounder ladder numbers: GUI-Actor-7B 44.6; Qwen3-VL-8B 54.6; Qwen3-VL-4B 59.5; MolmoPoint-GUI-8B 61.1 / OSWorld-G 70.0; UI-Venus-2-27B 74.1 | VERIFIED | V-104, anchors A1, V-122, V-085 |
| GUI-G1 hit vs IoU box-size directions | PARTIALLY (magnitudes) | V-061 |
| Point-It-Out "where it is" vs "where to act" known to differ | PARTIALLY | abstract describes a three-stage hierarchy (localization, task-driven pointing, visual trace), not a measured gap (V-191) |
| SafeGround, 2606.25760 | VERIFIED | V-191, V-151 |
| Position paper 2605.17273 | VERIFIED (exists) | V-164 |
| VPSG directional bias | VERIFIED | V-157 |
| UI-TARS-1.5-7B exists | VERIFIED (S: used as grounder in MMBench-GUI) | V-185 |
| Qwen3.5-27B as planner: agent numbers | VERIFIED: OSWorld-Verified 56.2, AndroidWorld 64.2 on its card | V-077 |
| Episode cost, discordant rate, subset size, effect sizes | SPECULATION, unchecked | not checkable |

## E. Verifier A4 dependencies as stated in §6

None of (a) to (e) is load-bearing; (c) motivates the reward question only. Consistent with the anchors. Correct.

## F. Actions requested of the author

1. Rewrite §1 and §4: replace "the one verified data point is Jedi" with the four fixed-planner comparisons in Section B, and state the delta as parametric single-property injection, empirical replay, abstention under three re-plan policies, and metric prediction.
2. Answer 22-N1 in the Discussion log with GUI-RobustEval (2605.29447) as the closest injection precedent and OSWorld-Noisy (2606.22948) as the closest closed-loop perturbation benchmark; state the differences listed in Section 0.
3. Update the Agent S2 entry in §5 (Figure 6 grounder swap; Mixture-of-Grounding ablation) and add MMBench-GUI and GroundCUA to §5.
4. Use MMBench-GUI's "roughly linear" claim as the pre-registered null to test: if success is linear in offline grounding accuracy regardless of error type, the injection arms should show it; if identity errors and near-misses differ at matched accuracy, linearity breaks.
5. Cite the OSWorld abstract (not "error analysis [LIKELY]") for the grounding-deficiency finding, and soften the Aguvis and Point-It-Out sentences to what their abstracts support.

## G. Addendum (2026-09-02): Researcher2's follow-up checks (1) to (3) and the Skeptic's condition (c)

**(1) Output-side, parametric click injection under a fixed planner in a closed loop: none found after a final sweep (V-202).** Newly surfaced and checked at abstract level: AgentHijack (2605.25707) corrupts the agent's input (pop-ups, resolution changes); UI Element Injection (2604.07831) overlays distractor elements on the screenshot for 19 models; GUI-Primitives (2608.21832) is an offline 994-item spatial-binding diagnostic. Together with V-187 to V-190 (GUI-RobustEval, OSWorld-Noisy, GUI-Perturbed, 2607.04334, 2606.14106, GUI-Reflection), every perturbation study found acts on the environment or the input, or replays real errors at fixed depth; none perturbs the executed click parametrically. The Skeptic's condition (c) is satisfied as far as the literature I can reach shows.

**(2) GUI-RobustEval error modes and state production: VERIFIED with a gap (V-199).** "Incorrect UI Element" is one of the named error types and is a grounding-type error; the others recoverable from the HTML are Incorrect Parameter, Miss Necessary Step and Compositional Error (4 of 11; the full taxonomy is in a figure I could not read). Erroneous states are real: the root-cause action of a genuine failed trajectory is re-executed after an expert-repaired, error-free prefix, and the agent takes over at depth 0, 1, 3 or 5.

**(3) GUI-Perturbed and EvoCUA-32B: REFUTED (V-200).** GUI-Perturbed v2 mentions EvoCUA only as a citation about LoRA degrading capabilities; it contains no EvoCUA-32B, no OSWorld ranking and no grounding-versus-agent comparison. Remove that sentence from Section 5 or find its real source. GUI-Perturbed's own numbers: Qwen2.5-VL-7B 86.9 to 45.0, UI-TARS-1.5-7B 91.0 to 35.0, GTA1-7B 92.8 to 65.8 from direct to relational instructions. OSWorld-Noisy's id is ENVS, 2606.22948 (V-188).
