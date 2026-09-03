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

### Q-11. The class-agnostic area ceiling (DEVIATIONS D-23)
0.50 is ours, not P1's. Must be frozen with P1 to P7.

### Q-12. P3's overlap rule against a dense segmentation -- RESOLVED 2026-09-03: reading B
P3 rejects a CONTROL_OBJ candidate whose hole shows "no overlap with the
exclusion set after the candidate's own mask is removed from it", and acceptance
check 2 asserts the same on the *dilated* hole. The design appears to assume
class-agnostic masks are a modest addition covering unlabelled objects. In
practice SAM 2's automatic generator returns a near-complete scene partition
(20 to 31 masks per image), and the exclusion set then covers a median **80.3%**
of the image (min 19.5%, max 97.7%), so almost every candidate's 5 px dilation
ring touches something.

Measured on 25 real pool images, 228 area- and centrality-matched candidates:

| reading | CONTROL_OBJ yield |
|---|---|
| A, literal (as implemented) | **8%** |
| B, referent absolute + no other object bitten by more than 10% | **84%** |
| C, class-agnostic masks left out of the exclusion set | 36% |
| D, tested on the undilated mask instead of the hole | 100% |

A cannot produce the 10,000-image bank P1 asks for. D is close to vacuous, since
candidates rarely overlap each other's cores and the dilation ring is exactly
what would damage a neighbour. C discards the protection P1 asks for by name.
B keeps P3's purpose -- the control removes one object without materially
damaging another, and never touches the referent -- while remaining satisfiable.

**Chosen: B** (user decision, 2026-09-03), implemented as
`sampler.exclusion_violation` with `CONTROL_BITE_TOLERANCE = 0.10`, and asserted
by acceptance check 2 so the assert tests the rule the sampler sampled under.
Re-measured on the same 25 images: CONTROL_OBJ yield 84%, control mix 11
labelled_other_class and 27 class_agnostic. Must be frozen with P1 to P7 at B0a.

### Q-13. CONTROL_BG placement -- RESOLVED 2026-09-03: reading B, same rule as CONTROL_OBJ
CONTROL_BG keeps P3's strict rule -- the referent's hole shape translated to a
position with *no* overlap with the exclusion set -- and with that set covering a
median 80% of the image, a position is found on only 9 of 21 images (43%).
CONTROL_BG is secondary in P3 and feeds only *reported* comparison rows in K1
(P7: "REMOVE versus CONTROL_BG is reported on the same rows"), so it does not
block the gate, but it does thin those rows, and in K2 it is the instability
guard of stop rule (b'). The likely fix is that a background hole should avoid
*objects* rather than the whole exclusion set: SAM's class-agnostic masks include
sky, ground and wall regions, which is exactly where a background hole belongs.
**Resolved by measurement on 400 banked pool images** (CPU only, off the
instance store):

| CONTROL_BG placement rule | images placed |
|---|---|
| strict, P3 as written | 110/400 = 27.5% |
| reading B (referent absolute, 10% nick) | 309/400 = **77.2%** |
| objects only (class-agnostic regions allowed) | 303/400 = 75.8% |

The two alternatives are within 1.4 points, so the tiebreaker is methodological:
P16's stop rule (b') compares the box shift under CONTROL_OBJ against the shift
under CONTROL_BG. Placing the two controls under different rules would confound
"object versus background" with "rule A versus rule B". Reading B is therefore
applied to both, and `sample_control_bg` takes `instances` to do it; the strict
rule remains as the no-instances path. Must be frozen with P1 to P7 at B0a.

The same run gives the real CONTROL_OBJ figures on 400 images rather than 25:
**86.5% yield**, control mix 139 labelled_other_class and 207 class_agnostic,
exclusion-set coverage median 85.6% (p10 41.6%, p90 97.0%).

### Q-14. Rectangular holes yield less than mask holes
P7 requires the gate verdict on both hole types. A rectangular hole is the mask's
bounding box, so it is larger and meets the exclusion set more often: on a
30-image smoke run, CONTROL_OBJ was found for 83% of images with mask holes and
70% with rect holes. Not a blocker -- both hole types get their own count row and
their own gate verdict -- but the rect arm will rest on a smaller bank, and the
count table has to say so.

## Process

### Q-9. Git repository -- RESOLVED 2026-09-03
Initialised at `VLM-grounding/` on branch `main`; the freeze points B0a and B0b
now have something to tag and `shared/env/probe.py` reports a real `code_sha`
(with a `-dirty` suffix when the tree has uncommitted changes).
