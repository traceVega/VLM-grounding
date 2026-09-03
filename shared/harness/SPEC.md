# Evaluation harness contract, SPEC v1.2

Status: DRAFT v1.2, 2026-09-02. This is the single normative definition of the model configuration, the run contract, the result store and the backend parity rule used by IDEA-11 and IDEA-91. Both design documents cite this file by version and carry no copy. A change bumps the version and updates both design documents in the same commit; `schema.py` next to this file is the executable form and validates every parquet file on write and read.

Changes in v1.1: the parity slice is named (Section 5); `resolution_policy` is a model-config field and `image_px_sent` is mandatory on every row (Sections 1 and 4); coordinate mapping uses per-axis factors (Section 1). Changes in v1.2: the YAML carries no override values (Section 2 wording) and the manifest records `effective_resolution` read back from the processor (Section 2).

## 1. Model configuration (`configs/models/<model_id>.yaml`)

| Field | Type | Meaning |
|---|---|---|
| `model_id` | string | stable name used in results, for example `qwen3vl-8b-instruct` |
| `hf_path` | string | Hugging Face repository |
| `revision` | string | commit SHA, never a branch name |
| `dtype` | `bfloat16` | the only value used |
| `attn_impl` | string or map | `sdpa`, `eager`, or a map per sub-config such as `{text_config: eager_capture, vision_config: sdpa}` |
| `backend_default` | `hf` or `vllm` | the backend a kill run uses unless overridden; a backend other than the default requires the parity record of Section 5 |
| `resolution_policy` | map | how the model's input resolution is controlled: for Qwen-family processors `{kind: pixels, fields: [min_pixels, max_pixels]}` with values supplied per run as overrides; for Molmo2 `{kind: crops, max_crops: <int>, crop_size: <int>}` pinned to the processor defaults and recorded; the harness fills `image_px_sent` from the processor's actual output on every row |
| `prompt_templates` | map | `grounding`, `presence`, and benchmark-specific overrides such as `groundingme`; values are paths to versioned prompt files under `configs/prompts/` |
| `coordinate_convention` | enum | `relative_1000` (0 to 1000 relative to the model's input image; mapped to original pixels per axis using the resized size from the processor, since the resize can be anisotropic by a few percent), `absolute_resized` (pixels of the resized input, mapped per axis), `percent_float` (percent of width and height), `loc_tokens` (PaliGemma-style 1,024 bins, y-first) |
| `parser` | string | name of the parser function in `shared/harness/parsers.py` |
| `abstain_protocol` | map | `primary` and optionally `secondary`, each with `prompt_suffix` (string, may be empty), `none_patterns` (list of literal strings or regexes that mean "no target"), and `null_box` (the literal null-box output the benchmark protocol expects, if any) |
| `output_type_support` | list | subset of `box`, `point`, `set`, `none` |
| `tokens` | map | pinned token ids the read-outs need, such as `yes_ids` and `no_ids` |

## 2. Run contract

`harness run --model <model_id> --items <set or slice> --conditions <list> [--override min_pixels=.. max_pixels=.. abstain_protocol=primary|secondary backend=hf|vllm] --run-id <id> [--non-kill]`

- Run-level overrides are recorded in the manifest; the per-model YAML carries no override values (it carries the resolution policy and, for crop-based processors, the pinned crop settings).
- `--non-kill` marks acceptance, parity, smoke and dev-slice runs; analysis refuses to join non-kill run ids into kill tables.
- Output: `results/<run_id>/outputs.parquet` and `results/<run_id>/manifest.json`; the directory is immutable after the manifest is written.

Manifest fields: `run_id`, `non_kill` (bool), `prereg_version`, `code_sha`, `config_hash`, `prompt_hashes` (map), `model_id`, `model_revision`, `dataset_revisions` (map), `overrides` (map), `effective_resolution` (the resolved `min_pixels` and `max_pixels`, or the crop policy, read back from the processor after overrides are applied, so a field left at a processor default is still recorded), `backend`, `server_args` (for vLLM runs), `rejected_requests` (int), `seeds` (map including `bootstrap_seed` where analysis writes it), `package_versions` (map), `gpu_name`, `idle_vram_mb`, `started_at`, `ended_at`, `validity_run_ids` (the acceptance-check and parity run ids this run relies on), `file_sha256` (map of every file in the directory, written last).

## 3. `items.parquet`

| Column | Type | Meaning |
|---|---|---|
| `item_id` | string | unique across sets |
| `set` | string | `groundingme`, `openref`, `ref_l4`, `grefcoco`, `cocosearch18`, `openimages_pool`, `sa1b_pool`, `synthetic` |
| `split_half` | string or null | `coco` or `objects365` for Ref-L4 |
| `image_path` | string | relative to the data root |
| `image_sha256` | string | of the file on disk |
| `image_phash` | string | 64-bit perceptual hash, hex |
| `expr` | string | the referring expression or category prompt |
| `gt_boxes_xyxy_px` | list of list of float | ground-truth boxes in original pixels; empty for no-target items |
| `n_gt` | int | number of ground-truth boxes |
| `size_bin` | string or null | `tiny`, `small`, `medium`, `large`, `xl` by longer side of the first ground-truth box |
| `dimension` | string or null | GroundingME dimension |
| `in_kill_set` | bool | member of a kill set |
| `dev_slice` | string or null | `a`, `b`, `p12` or null |
| `pair_id` | string or null | K2 pair identifier |
| `ceiling_iou_d4` | float or null | grid ceiling at the 2.4 Mpx fixed setting (IDEA-11 D5) |
| `downscale_factor_d4` | float or null | descriptive: original longer side over resized longer side at the fixed setting |

Disjointness of dev slices from kill sets is asserted on `image_phash` (Hamming distance at most 4) at build time.

## 4. `outputs.parquet`

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | |
| `model_id` | string | for API models, the response model string |
| `item_id` | string | |
| `condition` | enum | `ORIGINAL`, `REMOVE`, `CONTROL_OBJ`, `CONTROL_BG`, `T_NULL`, `T_HEAD`, `T_ATTR` |
| `pair_id` | string or null | |
| `abstain_protocol` | enum | `primary`, `secondary` |
| `prompt_version` | string | hash of the prompt file |
| `raw_text` | string | the model's full text output |
| `parse_ok` | bool | |
| `output_type` | enum | `box`, `point`, `set`, `none`, `invalid`, `refusal` |
| `n_boxes` | int | number of boxes or points parsed; for single-box tasks a value above 1 sets `output_type` to `set` and the first box is stored |
| `box_xyxy_px` | list of float or null | first box in original pixels |
| `point_xy_px` | list of float or null | first point in original pixels |
| `iou_gt` | float or null | defined for `ORIGINAL` rows only: IoU with the first ground-truth box; 0 for `none`, `invalid`, `refusal` |
| `point_in_gt` | bool or null | defined for `ORIGINAL` rows only |
| `is_failure` | bool or null | defined for `ORIGINAL` rows only: `parse_ok` false, or `output_type` in `none`, `invalid`, `refusal`, or `iou_gt` below 0.5 (for points: not inside the box) |
| `image_px_sent` | list of int | width and height actually given to the model after the processor's resize, or sent to the API; mandatory on every row |
| `latency_ms` | int | |
| `tokens_in`, `tokens_out` | int or null | API rows |

Relations for the non-ORIGINAL conditions are computed by the idea-specific relation code from joins on `pair_id` and `condition`, never stored as `iou_gt`.

## 5. Backend parity rule

A backend other than `backend_default` may serve a kill run only after a parity run on dev slice (a) as materialized by `shared/data` from Ref-L4 (IDEA-11 D2(a), 1,000 items), under the same override set as the kill run it serves (IDEA-11's fixed `min = max` setting and IDEA-91's cap-only setting are different override sets and need their own parity runs): `output_type` agreement at least 99%, and IoU between the two backends' boxes at least 0.95 on at least 95% of items where both produced a box. The parity run id is written into the kill run's manifest under `validity_run_ids`. Models with no Hugging Face-native implementation (Molmo2) use vLLM as `backend_default` and a batch-one `trust_remote_code` spot check as the parity record.

## 6. Determinism tolerance

Two runs of the same read-out pass on dev slice (b) must agree exactly on B0 boxes for at least 98% of items and within IoU 0.9 for the rest; the tolerance is printed in every kill table.

## 7. Validation

`schema.py` exports the pyarrow schemas of Sections 3 and 4 and a `validate(path)` function; every writer calls it before closing a file and every reader on open. A test in CI diffs the enum lists in this file against `schema.py`.
