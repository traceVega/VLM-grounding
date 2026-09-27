# train/ — the PoC trainers and evaluation suite (2026-09-23/24)

Everything runs in WSL (`wsl.exe -d Ubuntu`), env `~/vlmg-env`, data under `~/vlmg-data/train/`.
Plan: `notes/POC-TRAIN-PLAN-2026-09-23.md`. Results: `notes/POC-TRAIN-RESULTS-2026-09-24.md`.

| File | What it does |
|---|---|
| `data.py` | exported items (`~/vlmg-data/datagen/exports/poc_v1.jsonl`), scene-level split (18 val scenes, seed 0), answer text in the base model's own format (fenced JSON, 0–1000 coords), eval sets (GroundingME parquet, RefCOCO/+/g val shards, gray images). `NULL_TYPES` = the parser's null output types (`"none"`, `"refusal"`). |
| `sft_lora.py` | LoRA SFT with the unlock curve (`--eval-every`). Options: `--decision-weight`, `--null-unlikelihood` (−log(1−p(null)) on positives), `--neg-repeat`, `--extra <items.jsonl>`, `--init-adapter`, `--drop-4b-refused`. |
| `grpo_lora.py` | hand-written GRPO (no TRL). `--inject-gt neg` = RC-GRPO forced null when a negative's group has none; `--inject-gt all` = ViSurf ground truth in every group with reward smoothing; `--neg-adv-scale` (α), `--r-pos-null` (over-refusal penalty), `--beta` (KL to the start adapter, loaded as adapter "ref"); starts from `--init-adapter` or a fresh LoRA. |
| `eval_suite.py` | sets `gme` (all 1,005 or `--gme-pos-n N` stratified positives + all 201 Rejection), `gmegray` (GME gray-image control), `own`, `gray`, `refcoco`; resumable per item; `--prompt` for probes, `--max-new`. `summary.json` per tag under `~/vlmg-data/train/eval/<tag>/`. |
| `screen_compare.py` | prints screening tags next to the baseline recomputed on exactly the same items. `compare.py` is the plain side-by-side for full evals. |
| `curve.py` | prints a run's unlock curve. |
| `augment.py` | cross-scene negatives: `build` → `verify` (batched Gemini, ~0.03 USD/pair) → `export` (`~/vlmg-data/train/augment/cross_v1.jsonl`; 193/200 usable). |
| `run_queue.sh`, `queues/` | run a queue file sequentially with three retries per line. Launch detached so a session restart cannot kill it: `setsid nohup bash train/run_queue.sh train/queues/x.txt > ~/vlmg-results/x.log 2>&1 < /dev/null &`, plus a hidden `wsl.exe -d Ubuntu -- sleep infinity` holder. |
| `run_poc.sh` | the original three-stage chain (baseline eval → SFT → eval). |

Gotchas that cost time: the harness parser reports null as `output_type == "none"`; Qwen3-VL's
`mm_token_type_ids` must be extended with zeros when appending completion tokens for a
training forward; SFT holds ~27 GB and starves a concurrent eval (13 s/item), so run stages
sequentially; screening eval ≈ 20 min, SFT ≈ 10 min, GRPO (260 prompts, G=8) ≈ 30 min/epoch.
