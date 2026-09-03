# Open questions raised by implementing the design

Design Section 9 already tracks O1 to O8. These are additional questions the
code ran into, each with what the implementation does in the meantime. Anything
touching P1 to P21 must be settled before the corresponding freeze (P20).

## Blocking a kill run

### Q-1. GroundingME's own instruction and null-box convention (P12)
The primary protocol is pinned to the benchmark's own words, so the K2 numbers
can be described as "under the benchmark's protocol". Needs the GroundingME
repository at a recorded commit. Until then
`configs/prompts/grounding_qwen3vl_primary.txt` is `UNVERIFIED` and any run
without `--non-kill` refuses it. **Blocks B8 and B11.**

### Q-2. Molmo2's native pointing instruction and abstention (P12, P19)
Same, from the Molmo2-8B model card at the pinned revision: the exact
instruction string, the form of the no-such-object sentence, and how an empty
point list is emitted, which becomes `none_patterns` in the model YAML.
**Blocks B8 and B11.**

### Q-3. Qwen3-VL's coordinate convention (SPEC Section 1)
SPEC allows `relative_1000`, `absolute_resized`, `percent_float` or
`loc_tokens`. Qwen2-VL used 0-1000 relative; Qwen2.5-VL moved to absolute
coordinates of the resized input. Which applies to Qwen3-VL-8B-Instruct must be
read off the processor and confirmed on the dev slice, not assumed: the whole
box-mapping path depends on it. `PIN_REQUIRED` in
`configs/models/qwen3vl-8b-instruct.yaml`. **Blocks B8.**

### Q-4. SAM 3 access -- RESOLVED 2026-09-03
Access approved, token in place, weights downloaded and the adapter verified
against them: concept prompt IoU 0.99, an absent concept returns zero instances
(which is what P13's T_HEAD selection relies on), box prompt IoU 0.99,
box-to-mask IoU 0.97, ~0.2 s per prompt, peak 2.13 GB VRAM. Pin is
`facebook/sam3` at `3c879f39826c...` (DEVIATIONS D-16). Still outstanding under
design O7: **read the licence text** before anything derived from the bank is
released. Class-agnostic masks stay with SAM 2 (Q-10).

## Not blocking, but to settle before the K1 freeze

### Q-5. The local ladder's reference class (check 1a)
See `DEVIATIONS.md` D-6. The implementation mirrors the global ladder (the other
edit of the pair, re-encoded in-hole at q95). An untouched reference would make
the local ladder slightly easier; either is defensible, but it must be fixed
before the freeze because it sets the local floor.

### Q-6. Hole-area bin edges for the local floor
See D-7. Must be frozen with P5 to P7.

### Q-7. Tile budget for gate row (ii)
P5 aggregates ViT-S/16 tile scores per image by the maximum. Over an OpenImages
pool most images give 4 to 9 tiles, which is affordable, but the design's
budget (gate classifiers 4 to 6 h) assumes something. `TileDataset` draws one
tile per image per epoch at train time and uses every tile at eval, so an epoch
is one pass over images. If the pool turns out to hold large images, the train
sampler is the knob, and the choice belongs in the frozen text.

### Q-10. SAM 3 has no generic-object mode (closes half of design O6)
`Sam3Processor.__call__` accepts `text` and `input_boxes` only, and the model's
outputs post-process through `post_process_instance_segmentation`; there is no
generate-everything entry point in transformers 5.16.1. So the class-agnostic
masks of P1/P3 come from SAM 2's automatic mask generator, which is the
fallback design O6 already names. Worth confirming against the SAM 3 card once
the gate is open, in case the released model exposes one another way.

### Q-8. Which forensic detector (P6)
"TruFor if its weights are obtainable under research terms, else an RGB-domain
detector of the PSCC-Net or MVSS-Net class". Undecided until the weights are
requested; `PINS.md` has the row waiting. It is an adversary row and does not
gate, so it does not block the freeze, only the completeness of the table.

## Process

### Q-9. Git repository -- RESOLVED 2026-09-03
Initialised at `VLM-grounding/` on branch `main`; the freeze points B0a and B0b
now have something to tag and `shared/env/probe.py` reports a real `code_sha`
(with a `-dirty` suffix when the tree has uncommitted changes).
