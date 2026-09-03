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
