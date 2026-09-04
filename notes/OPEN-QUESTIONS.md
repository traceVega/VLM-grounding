# Open questions raised by implementing the design

Design Section 9 already tracks O1 to O8. These are additional questions the
code ran into, each with what the implementation does in the meantime. Anything
touching P1 to P21 must be settled before the corresponding freeze (P20).

## Blocking a kill run

### Q-1. GroundingME's instruction and null-box convention -- RESOLVED 2026-09-03
Read from `lmms_eval_task/groundingme/utils.py` (the module-level `PROMPT`, line
95) and `evaluate.py` at `lirang04/GroundingME@6867f7a0`, recorded in `PINS.md`.
Copied byte for byte into `grounding_qwen3vl_primary.txt`, now `VERIFIED`; the
secondary is the same string plus P12's sentence, and a test asserts it is a
strict prefix.

The dataset itself: 1,005 items, of which **804 are positive single-box** and
**201 are `Rejection`, whose `bbox` is null**. `bbox` is absolute `xyxy` in
original-image pixels (100% of positives are consistent with `xyxy`, 15% with
`xywh`). `detection_type` carries a head noun for 100% of items, so P10's
verifier question needs no noun-phrase parse on this set. Every positive is
already within P8's 30% area limit, so that filter excludes nothing here.

Three consequences the pre-registration should absorb:

**(a) P8's 2,000 pairs are unreachable from GroundingME alone.** 804 positives
is the ceiling before the SAM 3 box-to-mask IoU filter. Without OpenRef, design
contingency O1 applies -- K2 becomes every eligible GroundingME item and P15's
CI half-widths are restated against a denominator near 800, not 2,000.

**(b) The primary protocol already has a rejection channel.** The benchmark's
own instruction ends "If no matching object is found, output `{"bbox_2d":
null}`". Deviation (3) in the design's preamble assumed the opposite -- that
under the benchmark's protocol the models have no way to abstain, which is why
the box-shift form (b') was pre-registered for P16. That reasoning needs
revisiting: abstention is available under the primary, so P16 (b) may be
fireable there after all, and P12's secondary ("output none") adds a *second*,
differently-formatted channel rather than introducing the first one. The parser
must accept both under the secondary or a primary-style abstention would be
scored as an unparseable box. Implemented as pre-registered; flagged for the
freeze, not silently changed.

**(c) The benchmark's scorer is oracle-assisted and K2 cannot copy it.** See
Q-3.

### Q-1b. P21's cap share on GroundingME -- MEASURED 2026-09-03
**99.3% of GroundingME images exceed P21's 2.4 Mpx cap** (998/1005); median
3.4 Mpx, max 59.0 Mpx (7680x7680), median downscale 0.85x. P21 asks for this
share to be reported per set; on this set it is very nearly everything, so the
token-count histogram is the informative artifact rather than the share.

### Q-2. Molmo2's pointing instruction and abstention -- MOSTLY RESOLVED 2026-09-03
Read from the Molmo2-8B card and its shipped processor code at revision
`e28fa2859`. Three of the four parts are settled:

* **Instruction.** The card's own worked example passes the literal text
  `"Point to the penguins."`, so P12's `"Point to the {expr}."` is Molmo2's
  native form rather than a paraphrase. Both prompt files are now `VERIFIED`.
* **Coordinate convention.** The card's decoder is explicit --
  `POINTS_REGEX = r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})"`, the comment "our
  points format assume coordinates are scaled by 1000", and
  `x, y = float(x)/1000*image_w`. So points are **0-1000 integers against the
  original image size**: SPEC's `relative_1000`.
  `configs/models/molmo2-8b.yaml` said `percent_float`, which was Molmo v1's
  convention and would have divided every coordinate by the wrong constant --
  silently, since both produce in-range points. Corrected.
* **Resolution policy (P19).** `max_crops: 8`, `crop_size: 378`,
  `patch_size: 14`, `pooling_size: [2, 2]`, `overlap_margins: [4, 4]`, read from
  `preprocessor_config.json` and recorded in the model config as P19 requires.

**Still open: the abstention form.** The card documents pointing but never shows
a no-such-object answer, and no abstention string appears in the shipped
processor or modeling code. So `none_patterns` stays `PIN_REQUIRED`. P12 already
names where this gets settled -- the `p12` dev slice of 100 gRefCOCO no-target
items under `--non-kill` -- so it is measured rather than guessed, and no design
change is needed. **Still blocks B8's primary-protocol Molmo2 rows.**

Worth noting against Q-1: Molmo2's native instruction offers no rejection
channel, so P12's secondary cue genuinely introduces the first one here. The
primary-versus-secondary contrast is real on the Molmo2 side and nearly absent
on the Qwen3-VL side.

### Q-3. Qwen3-VL's coordinate convention (SPEC Section 1) -- STILL OPEN, and now load-bearing
SPEC allows `relative_1000`, `absolute_resized`, `percent_float` or
`loc_tokens`. Qwen2-VL used 0-1000 relative; Qwen2.5-VL moved to absolute
coordinates of the resized input. Which applies to Qwen3-VL-8B-Instruct must be
read off the processor and confirmed on the dev slice, not assumed.
`PIN_REQUIRED` in `configs/models/qwen3vl-8b-instruct.yaml`. **Blocks B8.**

Checked 2026-09-03: neither the model card nor `config.json` /
`preprocessor_config.json` states it, so this needs an empirical probe on the
GPU -- deferred while the K1 edit bank has the device.

Q-1 raised the stakes. GroundingME's own scorer does *not* fix a convention: it
parses the box, builds four candidate readings (raw, normalised-or-0-999, MIMO,
Qwen) and **keeps whichever has the highest IoU with the ground-truth box**.
That is oracle-assisted decoding, and K2 cannot use it:

* the REMOVE condition has no ground-truth box by construction, so "the reading
  that maximises IoU with ground truth" is undefined there; and
* P14's same-box statistic is `IoU(box on REMOVE, box on ORIGINAL)`, which is
  meaningless unless both conditions are decoded in one frame.

So K2 must fix **one** convention per model and apply it to every condition.
A consequence to report rather than hide: K2's ORIGINAL-correct rate will not
match GroundingME's published accuracy for the same model, because the published
number is computed with the more generous best-of-four decode. P12 pins the
*prompt*, not the scorer, so this is consistent with running "under the
benchmark's protocol" -- but it has to be said out loud in the K2 table.

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
