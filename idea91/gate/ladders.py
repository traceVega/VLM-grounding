"""The sensitivity ladders and nulls of acceptance check 1.

A gate that passes because it is blind proves nothing.  Check 1 is what makes a
PASS mean something, and it runs entirely on the *1b pairs* -- the two
independent CONTROL_OBJ edits of the same image -- so no real REMOVE-vs-CONTROL
signal can contaminate a sensitivity measurement:

* **1a global ladder** (classifier capacity): edit #1 re-encoded whole at JPEG
  q75/q90/q92 against edit #2 at q95.  AUROC above 0.9 at q75 is *required*, and
  the q92 step must reach 0.7 in at least one gate row at the frozen gate
  resolution -- else the gate resolution is raised before the K1 freeze.
* **1a local ladder** (what the editor can leave): the same pairs with the JPEG
  re-encode composited *inside the dilated hole only* at q75/q50/q30.  The
  smallest detected step per hole-area bin is the gate's local floor, and a floor
  coarser than q75 in a bin means the gate is blind to editor-scale artefacts
  there -- not that the editor is clean.
* **1b null**: two independent CONTROL_OBJ edits, one labelled each way.  AUROC
  must be 0.5 +/- 0.05 in every gate row.
* **1c null**: paired crops outside both holes are pixel-identical, so AUROC is
  0.5 by construction; :func:`idea91.edits.composite.assert_paired_crop_identical`
  is the assert and this is the row that reports it.

The local ladder's reference class mirrors the global ladder's: the other edit
of the pair re-encoded in-hole at q95.  The design text names the ladder
qualities but not the reference for the local arm; recorded in
``notes/OPEN-QUESTIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from idea91.edits.composite import composite_in_mask
from idea91.masks import area_frac

GLOBAL_LADDER_QUALITIES = (75, 90, 92)  # check 1a, against q95
LOCAL_LADDER_QUALITIES = (75, 50, 30)  # check 1a, in-hole only
REFERENCE_QUALITY = 95

GLOBAL_REQUIRED_Q75_AUROC = 0.90  # "AUROC above 0.9 at q75 is required"
GLOBAL_REQUIRED_Q92_AUROC = 0.70  # "must reach AUROC at least 0.7 in at least one gate row"
LOCAL_DETECTION_AUROC = 0.70  # "the smallest detected local step (AUROC at least 0.7)"
NULL_TOLERANCE = 0.05  # check 1b: 0.5 +/- 0.05

#: Hole-area bins for the local floor.  P1 admits referents covering 0.5% to 15%
#: of the image; the design does not name the bin edges, so these are ours and
#: are recorded in ``notes/OPEN-QUESTIONS.md``.
HOLE_AREA_BIN_EDGES = (0.005, 0.01, 0.02, 0.05, 0.15)
HOLE_AREA_BINS = ("0.5-1%", "1-2%", "2-5%", "5-15%")


def hole_area_bin(hole: np.ndarray | float) -> str:
    """Bin a hole by its area fraction of the image."""
    frac = hole if isinstance(hole, float) else area_frac(hole)
    edges = HOLE_AREA_BIN_EDGES
    for i in range(len(HOLE_AREA_BINS)):
        if frac < edges[i + 1] or i == len(HOLE_AREA_BINS) - 1:
            return HOLE_AREA_BINS[i]
    return HOLE_AREA_BINS[-1]


def jpeg_reencode(image: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip through JPEG at ``quality`` (the ladder's damage operator)."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"JPEG encode failed at quality {quality}")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def local_jpeg(image: np.ndarray, hole: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode at ``quality`` but keep it inside the dilated hole (local ladder).

    Same compositing rule as the editor itself (P2), so the local ladder measures
    what the gate can see at editor scale rather than what a whole-image
    re-encode does.
    """
    return composite_in_mask(image, jpeg_reencode(image, quality), hole)


@dataclass
class LadderPair:
    """One image's contribution to a ladder: a damaged edit and its reference."""

    image_id: str
    damaged: np.ndarray
    reference: np.ndarray
    hole_area_bin: str


def global_ladder_pair(
    edit_a: np.ndarray,
    edit_b: np.ndarray,
    quality: int,
    *,
    image_id: str,
    hole_bin: str,
    reference_quality: int = REFERENCE_QUALITY,
) -> LadderPair:
    return LadderPair(
        image_id=image_id,
        damaged=jpeg_reencode(edit_a, quality),
        reference=jpeg_reencode(edit_b, reference_quality),
        hole_area_bin=hole_bin,
    )


def local_ladder_pair(
    edit_a: np.ndarray,
    hole_a: np.ndarray,
    edit_b: np.ndarray,
    hole_b: np.ndarray,
    quality: int,
    *,
    image_id: str,
    reference_quality: int = REFERENCE_QUALITY,
) -> LadderPair:
    return LadderPair(
        image_id=image_id,
        damaged=local_jpeg(edit_a, hole_a, quality),
        reference=local_jpeg(edit_b, hole_b, reference_quality),
        hole_area_bin=hole_area_bin(hole_a),
    )


# --- verdicts ----------------------------------------------------------------


@dataclass
class GlobalLadderVerdict:
    q75_auroc_by_row: dict[str, float]
    q92_auroc_by_row: dict[str, float]
    all_aurocs: dict[tuple[str, int], float] = field(default_factory=dict)

    @property
    def q75_ok(self) -> bool:
        """"AUROC above 0.9 at q75 is required" -- of the gate rows, the best one."""
        vals = [v for v in self.q75_auroc_by_row.values() if np.isfinite(v)]
        return bool(vals) and max(vals) > GLOBAL_REQUIRED_Q75_AUROC

    @property
    def q92_ok(self) -> bool:
        """"the q92 step must reach AUROC at least 0.7 in at least one gate row"."""
        vals = [v for v in self.q92_auroc_by_row.values() if np.isfinite(v)]
        return bool(vals) and max(vals) >= GLOBAL_REQUIRED_Q92_AUROC

    @property
    def passes(self) -> bool:
        return self.q75_ok and self.q92_ok

    def verdict_line(self) -> str:
        if self.passes:
            return "check 1a global ladder: PASS (gate is sensitive at the frozen resolution)"
        why = []
        if not self.q75_ok:
            why.append(f"q75 AUROC {max(self.q75_auroc_by_row.values(), default=float('nan')):.3f} <= 0.9")
        if not self.q92_ok:
            why.append(f"no gate row reaches 0.7 at q92 (best {max(self.q92_auroc_by_row.values(), default=float('nan')):.3f})")
        return (
            "check 1a global ladder: FAIL (" + "; ".join(why) + "). "
            "Raise the gate resolution before the K1 freeze (design check 1a)."
        )


def local_floor(aurocs_by_quality: dict[int, float]) -> int | None:
    """The smallest detected local step: the highest quality with AUROC >= 0.7.

    Qualities run 75 (least damage) -> 30 (most).  ``None`` means nothing on the
    ladder was detected, i.e. the gate is blind at editor scale in this bin.
    """
    detected = [
        q
        for q in sorted(aurocs_by_quality, reverse=True)
        if np.isfinite(aurocs_by_quality[q]) and aurocs_by_quality[q] >= LOCAL_DETECTION_AUROC
    ]
    return detected[0] if detected else None


def local_floor_line(bin_name: str, floor: int | None) -> str:
    """The line carried into the first table of every downstream use (check 1a)."""
    if floor is None:
        return (
            f"local floor [{bin_name}]: none of q75/q50/q30 detected -- the gate is blind to "
            "editor-scale artefacts in this bin; a PASS here says nothing about the editor"
        )
    if floor >= max(LOCAL_LADDER_QUALITIES):
        return f"local floor [{bin_name}]: q{floor} (the gate sees editor-scale damage)"
    return (
        f"local floor [{bin_name}]: q{floor}, coarser than q{max(LOCAL_LADDER_QUALITIES)} -- "
        "reads as the gate being blind in this bin, not as the editor being clean"
    )


@dataclass
class NullVerdict:
    """Check 1b: identical-distribution null, AUROC must be 0.5 +/- 0.05."""

    auroc_by_row: dict[str, float]
    tolerance: float = NULL_TOLERANCE

    @property
    def offenders(self) -> dict[str, float]:
        return {
            row: v
            for row, v in self.auroc_by_row.items()
            if not np.isfinite(v) or abs(v - 0.5) > self.tolerance
        }

    @property
    def passes(self) -> bool:
        return not self.offenders

    def verdict_line(self) -> str:
        if self.passes:
            return f"check 1b null: PASS (every gate row within 0.5 +/- {self.tolerance})"
        bad = ", ".join(f"{r}={v:.3f}" for r, v in self.offenders.items())
        return (
            f"check 1b null: FAIL ({bad}). Two edits from the same distribution are "
            "separable, so the gate's AUROC does not measure the class difference."
        )
