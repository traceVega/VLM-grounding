# Deviations from the design documents, dated

Every entry is a place where the implementation does something the design does
not say verbatim, with the reason. Nothing here changes a pre-registered value
(P1 to P21); those can only change through the freeze rule P20.

## 2026-09-02, at first implementation

### D-1. Python 3.12, not 3.11 (IDEA-11 design 4.1)

IDEA-11's 4.1 says Python 3.11. The host already runs a proven Blackwell
(sm_120) stack on Python 3.12.13 with `torch 2.13/2.14+cu130` and `vLLM 0.28`
(`~/ptr1-env`), and the same wheels resolve on 3.12 without a second CUDA
toolchain. The env for this repo is therefore pinned to 3.12 in
`pyproject.toml`. `shared/env` is shared with IDEA-11, so this applies to both.
Nothing in either design depends on a 3.11-only feature.

### D-2. `shared/stats.py` is a new shared module

The design's code homes list `shared/` as environment, data, harness, result
store and judge service. AUROC and the clustered bootstrap are needed by
IDEA-91's gate (P6) *and* by its relations (P15), and IDEA-11's tables quote
intervals of the same shape, so the implementation puts them in `shared/stats.py`
rather than duplicating them per idea.

### D-3. The WSL memory cap stays at 24 GB

IDEA-11 4.1 asks for `.wslconfig` `memory=28GB`. The host's existing
`.wslconfig` sets `memory=24GB` with a comment explaining that another project
(PointTrack-R1) OOMs Windows above that. K1 and K2 are GPU-bound and stream
edits to disk, so 24 GB is not expected to bind; the file was left alone rather
than changed under another project's feet. Raise it if the SAM 3 pass over the
K1 pool turns out to be host-RAM bound.

### D-4. Code on the Windows volume, data and results on ext4

4.1 says data, results and model caches live on the WSL ext4 volume and never
under `/mnt/c`. The source tree lives in the Windows project folder (where the
design documents already are) and everything else is on ext4, resolved through
`shared/paths.py` and overridable by `VLMG_DATA_ROOT`, `VLMG_EDITS_ROOT`,
`VLMG_RESULTS_ROOT`. The same code then runs unchanged on a RunPod volume.

### D-5. The window is square before clipping (P2)

P2 gives the window as "4 times the hole's longer side, at least 768 px and at
most 2,048 px on the longer side, clipped to the image". The implementation
builds a square of the clamped side and then clips, so a stored window is
non-square only where the image edge cut it. The window is slid inside the frame
rather than shrunk when the hole sits near an edge.

### D-6. The local ladder's reference class (check 1a)

The global ladder is explicit: one edit at q75/q90/q92 against the other at q95.
The local ladder names only the damage qualities (q75/q50/q30). The
implementation mirrors the global structure -- the other edit of the same pair,
re-encoded in-hole at q95 -- so the two arms differ only in where the damage is
applied. `reference_quality` is a parameter, so an alternative reading (an
untouched reference) is one argument away. Recorded in `OPEN-QUESTIONS.md`.

### D-7. Hole-area bins for the local floor

Check 1a reports the local floor "per hole-area bin" without naming the bins. P1
admits referents covering 0.5% to 15% of the image, so the implementation uses
0.5-1%, 1-2%, 2-5%, 5-15% (`idea91/gate/ladders.py`). These must be frozen with
the rest of P5 to P7 at the K1 freeze.

### D-8. `relations.parquet` carries five columns beyond the design's list

`item_id`, `set`, `size_bin`, `image_id` and `undetermined` are added to the
field list in design Section 5. The first four are what P15 stratifies by ("per
model, protocol, set and size bin") and `image_id` is the bootstrap's clustering
key; `undetermined` records P10's third outcome, whose share P15 requires to be
reported. Without them every analysis would have to re-join `items.parquet`.

### D-9. ViT rows pad inside the model

P6 asks for "ViT-S/16 (dynamic image size)". timm's dynamic image size still
requires a multiple of the patch size, and the native rows hand it arbitrary
sizes (1,024 px on the longer side gives 1024x683). The models are created with
`dynamic_img_pad=True`, so the padding happens inside the patch embedding and
the gate still sees native pixels rather than a resized image.

### D-10. Tiles shorter than 512 px are reflect-padded

Only an image smaller than the tile produces a short tile; padding it keeps one
shape per batch. Interior tiles are always full because `tile_positions` pulls
the last tile back to the edge instead of padding it.

### D-11. Placeholders are refused rather than guessed

Two prompts (P12's Qwen3-VL primary protocol, which is "GroundingME's own
instruction and null-box convention", and Molmo2's native pointing instruction
and abstention) and several model-config values (commit SHAs, coordinate
convention, `none_patterns`) cannot be filled in without the benchmark
repository and the model cards. They carry `status: UNVERIFIED` and
`PIN_REQUIRED`, and `shared/harness/prompts.py` and
`shared/harness/model_config.py` refuse to serve them to a run that is not
`--non-kill`. A guessed instruction that silently became "the benchmark's
protocol" is the failure the pre-registration exists to prevent.

## 2026-09-03, instance stack and data registry

### D-12. `instances.parquet` is a new store

Design Section 5 lists `edits/index.parquet`, `edits/verifier.parquet`,
`edits/human_labels.csv` and `results/<run_id>/relations.parquet`, but design
4.3 describes instances as a product of `idea91/instances` ("masks as RLE with
area fraction, bounding box, centre, `control_source` and box-to-mask IoU").
They are written to `instances.parquet` with a schema in `idea91/schemas.py`, so
the SAM pass runs once per image rather than once per edit. It also carries
`kill_grade`, which is False whenever a contingency backend or the heuristic
parser produced the scene.

### D-13. Instance `source` has three values beyond P3's `control_source`

`referent`, `same_class_other_instance` (K1) and `head_noun_neighbour` (K2).
All three belong in the P3 exclusion set but are never control candidates, and
none reaches `index.control_source`, whose enum is unchanged.

### D-14. A noun phrase containing the head noun is not a control candidate

P3 samples a K2 control from "a SAM 3 instance of a non-referent noun phrase".
The parser returns the referent's own phrase among the noun phrases ("red cup"
for head noun "cup"), and comparing phrases literally would have made it a
non-referent phrase. `NounPhraseParse.other_phrases` drops any phrase containing
the head noun as a word, so a control edit can never remove something the
expression names. Those instances still reach the exclusion set through the
head-noun concept prompt.

### D-15. `data/LICENSES.md` is generated, not hand-written

Design 4.2 asks for a single file with one row per asset. It is rendered from
`shared/data/sources.py` by `python -m shared.data.licenses --write`, so the
licence table cannot drift from the registry the download and build code reads.

### D-16. SAM 3 is pinned to `facebook/sam3`, not `facebook/sam3.1`

Design 4.1 says "SAM 3.1 if that is the current release at pin time; recorded".
SAM 3.1 *is* released, but the two repositories ship different formats:
`facebook/sam3` has `model.safetensors` (3.44 GB) and loads with
`Sam3Model.from_pretrained`, while `facebook/sam3.1` ships only
`sam3.1_multiplex.pt` (3.50 GB) with no safetensors, which the transformers
integration cannot read. The pin is therefore `facebook/sam3` at
`3c879f39826c...`, recorded here and in `shared/env/PINS.md`. Revisit if a
safetensors conversion of 3.1 appears; the multiplex checkpoint would otherwise
need a conversion step of our own, which is a new unpinned artefact in the
middle of the edit stack.

### D-17. SAM 3 runs through transformers, not a `sam3` package

transformers 5.16.1 ships `Sam3Model` and `Sam3Processor`, and the processor
takes `text=` for a concept prompt and `input_boxes=`/`input_boxes_labels=` for
a box prompt -- exactly the two capabilities design 4.3 needs. The adapter was
first written against a hypothetical standalone `sam3` package; it now uses the
transformers classes, and both adapters accept injected processor/model objects
so the tensor plumbing is unit-tested without the gated weights.

### D-18. Two SAM 3 call conventions found by running it, not by reading it

`Sam3Processor.__call__` types `input_boxes_labels` as three levels
(`list[list[list[int]]]`), but its own validator rejects three and requires two
(`[image][box]`). And box coordinates arrive as float32 while the model runs in
bfloat16, so the geometry encoder raises `mat1 and mat2 must have the same
dtype`; the adapter casts floating-point inputs to the model dtype. Text prompts
hit neither path, which is why the concept prompt worked before the box prompt
did. Both are covered by tests.

### D-19. The K1 pool is the Open Images *validation* split

P1 says "10,000 OpenImages images (CC-BY ...)" without naming a split. The
validation split is used: 41,620 images, **all of them CC BY 2.0** (checked, not
assumed), with a 24 MB box file against 2.15 GB for train, and 20,535 images
carrying a box inside P1's 0.5-15% band -- twice the pool P1 asks for, so there
is headroom for images the SAM 3 pass then drops. The v7 train bbox URL is a 403;
v5 validation is what serves. Nothing in K1 depends on which split the pixels
came from: it is a pool, not an evaluation set, and no model under test is
trained on it. The saving is 38 MB of metadata instead of 2.8 GB.

Images come from the CVDF mirror
(`https://open-images-dataset.s3.amazonaws.com/validation/<id>.jpg`), one request
per pool image, so only the 10,000 selected are fetched (~3 GB) rather than the
whole split. No AWS credentials and no `downloader.py` are needed.

Two pre-filters run before SAM 3 is loaded: `IsGroupOf` and `IsDepiction` boxes
are dropped (P1 wants an object instance), and boxes under 0.5% of the image are
dropped. The second is *sound* rather than heuristic -- a mask sits inside its
box, so a box under 0.5% cannot yield a mask over 0.5% -- and there is no upper
pre-filter, because a large box can still hold a small mask.

### D-20. Attribution lives beside the pool, not in `items.parquet`

P1 requires author, URL and licence recorded per image. `items.parquet`'s columns
are fixed by SPEC Section 3 and have no room for them, so they are written to
`prepared/openimages_pool/attribution.csv`, which must travel with the K1 bank if
it is ever released. `pool.parquet` beside it carries the per-image class labels
and boxes the instance pass prompts with.

### D-21. LaMa's generator is vendored, not installed, and loaded from the original checkpoint

`inpaint.py` first assumed a TorchScript `big-lama.pt`. The released artefact is
`big-lama.zip`: a hydra `config.yaml` and a PyTorch Lightning `best.ckpt`.
Rather than install `pytorch-lightning`, `hydra-core` and `webdataset` beside
torch 2.14 for code paths that only matter during training, the generator's
import closure (six files, ~28 KB) is vendored verbatim at
`advimman/lama@786f5936` under `idea91/edits/vendor/`, with `utils.py` reduced to
the one function `ffc.py` imports. The checkpoint is unpickled behind a
meta-path finder that stubs the training packages the pickle names, since the
only thing wanted from it is `state_dict['generator.*']`, and the config's
`${a.b.c}` interpolations are resolved by a 20-line resolver instead of
omegaconf. `kornia` *is* installed: LaMa's `spatial_transform` imports it at
module level, though the big-LaMa generator config does not use the wrapper.

Verified against the real weights: FFC-ResNet, 51.1 M parameters, 0.029 s per
512x512 fill after load, 0.45 GB VRAM, and the in-mask compositing assert holds
on a real edit.

### D-22. big-LaMa is Apache-2.0, so the K1 bank stays releasable

The licence was an open question, because a non-commercial or share-alike clause
on the editor would have undercut P1's reason for using CC-BY images at all. The
upstream repository is Apache-2.0 and its README states no separate terms for the
weights, whose download URL the README itself gives. Recorded in
`data/LICENSES.md` with the date read.

### D-23. Class-agnostic masks are capped at half the image

P1 gives class-agnostic masks a floor ("above 0.5% area") and no ceiling. SAM 2's
automatic generator readily returns the *background* as a high-confidence
"object" -- measured at 81% of a test scene. Putting that in the P3 exclusion set
leaves nowhere for a CONTROL_BG hole and rejects nearly every CONTROL_OBJ
candidate, so masks covering more than half the image are treated as scene rather
than object. Recorded as OPEN-QUESTIONS Q-11; the ceiling must be frozen with
P1 to P7 at the K1 freeze.

### D-24. De-duplication uses mask IoU, not box IoU

SAM's own automatic generator de-duplicates by box NMS. That fails here: a
candidate mask often carries a few stray pixels far from the object, and those
pixels blow up its bounding box without changing the mask. Measured on a test
scene, two candidates had mask IoU 1.000 and box IoU 0.348, so box NMS kept both
and the same object was returned twice. `mask_nms` compares masks directly, on
the 256 px decoder output so the comparison stays cheap. It also made the pass
four times faster, because de-duplication now happens before upscaling.

### D-25. SAM 2 comes from transformers, not the `sam2` PyPI package

The `sam2` 1.1.0 sdist on PyPI declares no dependencies at all, which does not
match the upstream `facebookresearch/sam2` project, and installing an unvetted
sdist runs its `setup.py`. `Sam2Model`/`Sam2Processor` from transformers 5.16.1
serve instead, with `facebook/sam2.1-hiera-large` at `665f8e2ad61c` (ungated).
The automatic mask generator is therefore ours
(`idea91/instances/automask.py`), following SAM's published algorithm: a point
grid, three candidate masks per point, then predicted-IoU, stability and area
filters, then de-duplication.

### D-26. `CompositeSegmenter` is how O6 resolves

SAM 3 answers concept and box prompts; it has no generate-everything call, so
the class-agnostic masks come from SAM 2. `CompositeSegmenter` pairs them and
keeps `is_pinned_backend=True`, because every mask that decides a referent or a
control instance comes from SAM 3 and SAM 2 only contributes obstacles to the
exclusion set, which can only make the sampler more conservative. Both backends
are named in the index row.

### D-27. P3's exclusion rule: reading B (Q-12)

The referent is absolute -- a control hole may not touch its dilated mask at all
-- and every other excluded instance tolerates a nick of at most 10% of its area,
measured outside the candidate. The literal reading yielded CONTROL_OBJ on 8% of
real images because SAM 2's automatic generator returns a near-complete scene
partition and the exclusion set then covers a median 80% of the image; reading B
yields 84%. Chosen by the user on 2026-09-03 after the four readings were
measured on 25 images. Acceptance check 2 asserts the same rule, so it remains an
independent second pass rather than a restatement of the sampler.

### D-28. SAM 3 vision embeddings are cached per image

P1 prompts every class label of an image and K2 every noun phrase, and
`Sam3Model.forward` documents `vision_embeds` as reusable. Encoding once per
image instead of once per prompt took the instance pass from about 4.3 s to
1.08 s per image -- roughly 14 hours to 3 for the 10,000-image pool -- with
identical outputs (verified prompt by prompt). The cache key is a content hash
rather than `id(image)`, because array ids are reused after garbage collection
and serving one image's embeddings for another would be silent corruption.

### D-29. CONTROL_BG shares the reading-B rule (Q-13)

`sample_control_bg` takes the instance list and applies the same rule as
CONTROL_OBJ: the referent absolute, no other instance bitten by more than 10%.
The strict rule (no overlap with the exclusion set at all) placed a background
hole on 27.5% of 400 pool images against 77.2% for reading B. The deciding
argument is not the yield but P16 (b'), which compares the CONTROL_OBJ shift with
the CONTROL_BG shift: two controls placed under two different rules would not be
comparable. Acceptance check 2 asserts whichever rule was used.

### D-30. The K1 bank carries both hole types, and is larger than the design budgeted

P7 requires the gate on both hole types, so the bank holds eight operators per
image (REMOVE, CONTROL_OBJ, CONTROL_OBJ_2, CONTROL_BG and their RECT_ variants)
rather than the four the design's 4.3 budget line counts. Measured on the smoke
run, a stored window averages 0.78 MB, against the design's implied 0.4 MB
(16 GB for 40,000). At roughly 7 edits per image over 10,000 images the bank is
therefore about 54 GB rather than 16 GB. There is 870 GB free, so this is a
budget correction, not a problem.

### D-31. P20 was breached by a throughput probe, before the freeze existed

**2026-09-03, ~20:20.** While pricing the remaining GPU work I ran a scratchpad
script that called `idea91.gate.train.run_row` directly on the contrast
`REMOVE_vs_CONTROL_OBJ`, for two rows at one seed each, on the edit bank as it
stood at 17% (1,340 images, 268 in the held-out half). P20 places the K1 freeze
"before the first classifier trains on real removals". No freeze existed. The
run therefore breached P20.

What was seen: AUROC 0.504 (`iii_resnet18_1024`) and 0.508
(`i_resnet18_full_cap`).

Why it happened: the guard was in the CLI stage (`run_k1 rows`), not in the
function that trains. A script calling `run_row` walked past it. The intent was
to measure seconds per training, and nothing about the sample contents was
considered.

Containment, in the order it matters:

1. **The frozen values could not have been influenced.** Every P1 to P7 constant
   was committed before the probe -- the freeze registry at `f823aed`, and the
   last commit touching gate behaviour at `4c1d878` (20:16), against the
   probe's result at roughly 20:22. `git log` is the evidence rather than an
   assurance.
2. **Nothing persisted.** The probe called `run_row`, not `train_cached`, so no
   result reached `gate_cache/` and none can be reused by the real run. The
   script itself is deleted.
3. **The guard moved.** `run_row` now calls `freeze.require` itself for any
   contrast beginning `REMOVE_vs_`, with an explicit `allow_unfrozen` for the
   ladders and nulls P20 permits pre-freeze. Guarded by default: a contrast
   added later is protected without anyone remembering to protect it.

Residual risk: the two numbers are known before the freeze is signed. They come
from a 17% bank at one seed and are not a K1 result, but the sign-off should
discount them, and this entry exists so a reader knows they were seen. If that
is judged insufficient, the remedy under P20 is to fork the document and report
both versions; that call belongs to whoever signs B0a, not to the
implementation.

### D-32. The gate trains with no DataLoader workers, because forked ones crash CUDA

**2026-09-04.** The checks stage died twice on its first row with
`torch.AcceleratorError: CUDA error: unknown error`, the driver reporting
`Returning 999 (CUDA_ERROR_UNKNOWN) from cuMemcpyHtoDAsync_v2` on the
host-to-device copy in `train_one`.

Not the code, and not the card. Immediately afterwards, a plain script did
thirty host-to-device copies of the same shape (64x3x1024x1024 uint8) and a
20-step bf16 matmul without a fault, and `nvidia-smi` showed the GPU idle and
healthy. The distinguishing factor is that the stage runs four DataLoader
workers, forked *after* the model has already initialised CUDA -- the classic
way to inherit a broken context, and evidently one this host's WSL GPU stack
does not tolerate. It is the same neighbourhood as the `dxg` ioctl failures in
the kernel log (see the WSL memory note).

With `--workers 0` the identical command completed thirteen rows with no fault,
including the four check-1b nulls and eight ladder rows.

**Cost:** about 17%. A fully cached row took 25 s at 500 images, which scales to
roughly 89 s at 1,787, against 76 s measured with four workers. That is a small
price now that the render cache made loading an `.npy` read and normalising
moved to the device -- and it is measured against a stage that otherwise does
not finish.

`--workers` still exists and can be raised on a host that tolerates it; the
default is 0. An untested alternative that would keep the parallelism is a
`spawn` multiprocessing context on the DataLoader, which avoids inheriting the
parent's CUDA state; it was not tried, because robustness at one in the morning
was worth more than an unmeasured speedup.
