# Team Charter — Research ideation on VLM grounding

Date of kickoff: 2026-09-01. Your training data may end before this date: use WebSearch/WebFetch to find 2025–2026 work.

## Goal (from the user, verbatim)
Generate concrete and potentially high-impact research ideas that could meaningfully advance the field of VLM grounding and have the potential to become strong top-tier AI conference papers.
Do not merely summarize existing work. The main objective is to understand the field deeply, identify important open problems and opportunities, and develop novel research ideas with concrete methodology.
Final output: a selective ranked shortlist. Prefer a few genuinely promising, well-investigated ideas over many superficial ones.

## Scope: what "VLM grounding" means here
Broadly: the ability of vision-language models (VLMs / MLLMs) to anchor language to specific visual evidence and to reason faithfully from it. Non-exhaustive sub-areas:
- referring expression comprehension / segmentation, phrase grounding, open-vocabulary detection inside VLMs
- pointing / box / mask / polygon output from MLLMs; coordinate representations
- grounded captioning and grounded description; attribution of generated text to regions
- grounded reasoning ("thinking with images", zoom/crop/tool use, visual chain-of-thought)
- hallucination and faithfulness treated as grounding failures
- GUI / screen grounding for computer-use agents
- video temporal and spatio-temporal grounding; 3D / embodied / robotic grounding; spatial reasoning
- training methods for grounding (SFT, RL with verifiable rewards, data synthesis, self-improvement)
- evaluation: benchmarks, saturation, contamination, what current metrics miss
The team should decide early which sub-areas hold the most opportunity instead of covering all equally.

## Roster (SendMessage addresses)
- `main` — Lead (coordinates, maintains TASKS.md, arbitrates, writes final synthesis)
- `Researcher1` — Frontier Explorer
- `Researcher2` — Builder / Experimentalist
- `Researcher3` — Field Explorer / Historian
- `Skeptic` — Adversarial Reviewer
- `Verifier` — Independent Checker

## Tools
- First action in every phase: `ToolSearch` with query `select:SendMessage,WebSearch,WebFetch` to load messaging and web tools.
- Message a teammate: `SendMessage({to: "Skeptic", message: "..."})`. Message the lead with `to: "main"`.
- Shared workspace (durable record): `D:\Dev\ArcNova\auto-research\VLM-grounding\team\`
  - `CHARTER.md` — this file
  - `TASKS.md` — shared task list maintained by the lead; teammates append status lines under their own tasks
  - `landscape/` — Phase 1 exploration outputs (one file per teammate)
  - `ideas/IDEA-<nn>-<slug>.md` — research proposals, following `ideas/TEMPLATE.md`
  - `critiques/IDEA-<nn>-skeptic.md` — Skeptic reviews (one per idea, updated across rounds)
  - `verifications/claims-log.md` — Verifier's running log; `verifications/IDEA-<nn>-verifier.md` for per-idea checks
  - `FINAL.md` — lead's synthesis (written only after Phase 3)
- Files are the record; messages are for debate, alerts and requests. Every message should point to a file section where possible and say what you want the recipient to do.
- Read files with the Read tool or Bash `cat`; write with the Write tool (heredocs with apostrophes have failed in this shell). Prefer appending to existing files over rewriting other people's files. Never delete another teammate's file.

## Evidence labels (mandatory on every non-trivial factual claim)
- `[VERIFIED: <url or arXiv id>]` — you read the source
- `[LIKELY: <reason>]` — strong recollection or indirect evidence, not checked
- `[SPECULATION]` — your own conjecture
Cite papers with title + arXiv id or URL. A number without a source is a `[LIKELY]` at best.

## Idea numbering (avoid collisions)
Researcher1: IDEA-11, 12, 13, ... Researcher2: IDEA-21, 22, ... Researcher3: IDEA-31, 32, ... Merged ideas: IDEA-9x (lead assigns).

## Idea status vocabulary
DRAFT → UNDER REVIEW → REVISED → one of SURVIVING / ABANDONED / MERGED (into IDEA-xx). Status is the first line of the idea file.

## Skeptic verdict vocabulary
Per idea: `REJECT` / `MAJOR REVISION` / `MINOR REVISION` / `PROMISING`. Each objection is numbered and typed: NOVELTY (must cite the closest prior work), ASSUMPTION, METHOD, EVAL, IMPACT, FEASIBILITY. A NOVELTY objection without a concrete citation is not allowed.

## Verifier verdict vocabulary
Per claim: `VERIFIED` / `REFUTED` / `PARTIALLY` / `UNVERIFIABLE`, each with the evidence consulted and a one-line reason.

## Debate rules
1. Authors must answer every Skeptic objection in the idea file's **Discussion log**: `ACCEPT` (and revise the idea), `REBUT` (with evidence), or `CONCEDE` (and narrow or abandon). Silence is not an answer.
2. The Skeptic re-reviews after revisions; verdicts can go up or down. Two rounds minimum for any idea that reaches SURVIVING.
3. Disagreement is expected and valuable. Do not soften a critique to be polite, and do not accept a critique you think is wrong: rebut it with evidence and, if unresolved, ask `main` to arbitrate (with both sides' evidence in the file).
4. Any teammate may ask the Verifier to check a specific claim. Keep requests specific: the claim, why it matters, where it lives.
5. Abandoning an idea is a good outcome. Record why in the file so the reasoning is not lost.
6. Novelty test: for every idea, name the 3 closest papers and state in one sentence what this idea does that each of them does not. If you cannot, the idea is not ready.
7. Before finishing any phase: re-read `TASKS.md` and any file that mentions you, answer outstanding objections, then send `main` a short final report (what you produced, file paths, open disputes, what you recommend next).

## Phases
- Phase 1 — Landscape: independent exploration; each teammate writes their `landscape/` file and candidate directions.
- Phase 2 — Proposals and debate: researchers write IDEA files; Skeptic critiques; Verifier checks; authors respond and revise; cross-researcher discussion.
- Phase 3 — Convergence: surviving ideas get concrete experimental plans (Researcher2 leads), final Skeptic verdicts, final Verifier checks.
- Phase 4 — Lead synthesis (`FINAL.md`).
