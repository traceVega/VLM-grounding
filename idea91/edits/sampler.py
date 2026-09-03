"""Control-edit samplers: CONTROL_OBJ and CONTROL_BG (design P3), and the
always-edited plan of P4.

P3 in full:

    CONTROL_OBJ (primary): removal, with the same operator and hole type, of a
    non-referent instance, preferring a labelled instance of another class (K1)
    or a SAM 3 instance of a non-referent noun phrase (K2), and falling back to a
    class-agnostic mask, recorded as ``control_source``; area between 0.5 and 2
    times the referent's; centre distance from the image centre within plus or
    minus 20% of the diagonal relative to the referent's; no overlap with the
    exclusion set after the candidate's own mask is removed from it.
    CONTROL_BG (secondary): the referent's hole shape at a position with no
    overlap with the exclusion set, inpainted the same way.  Exclusion set: the
    referent mask dilated 16 px, every SAM 3 instance of every parsed noun
    phrase of the expression (K2) or of every class label (K1), and every
    class-agnostic mask above 0.5% area.  Rejection sampling up to 100 tries;
    items with no valid CONTROL_OBJ are dropped and counted per set and bin.

The point of the matching is that neither "edited", nor "central smear", nor
"stuff versus thing" separates the K1 classes: both classes are one object
removed by the same operator, at matched area, matched centrality and matched
mask source.

Acceptance check 2 -- "no control hole overlaps the exclusion set of P3 with the
candidate's own mask removed from it, asserted over every pool and item" -- is
:func:`assert_no_exclusion_overlap`, and the sampler enforces it at sampling
time, so the assert is a second, independent pass over the built bank.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from idea91 import masks as M

HOLE_DILATION_PX = 5  # P2: the mask hole is the SAM 3 mask dilated by 5 px
EXCLUSION_DILATION_PX = 16  # P3: the referent mask dilated 16 px
CLASS_AGNOSTIC_MIN_AREA_FRAC = 0.005  # P1/P3: class-agnostic masks above 0.5%
AREA_RATIO_RANGE = (0.5, 2.0)  # P3
CENTRALITY_TOLERANCE = 0.20  # P3, as a fraction of the image diagonal
MAX_TRIES = 100  # P3
#: P3's overlap rule, reading B (OPEN-QUESTIONS Q-12, chosen 2026-09-03): the
#: referent is absolute, every other excluded instance tolerates a nick of at
#: most this fraction of its area.  The literal reading yielded CONTROL_OBJ on
#: 8% of real images because a dense segmentation leaves nothing untouched.
CONTROL_BITE_TOLERANCE = 0.10

#: ``control_source`` values, in the preference order of P3.
CONTROL_SOURCES = ("labelled_other_class", "noun_phrase_instance", "class_agnostic")
HOLE_TYPES = ("mask", "rect")


@dataclass(frozen=True)
class Instance:
    """One SAM 3 instance (or the referent) on one image."""

    instance_id: str
    mask: np.ndarray
    source: str  # 'referent' or one of CONTROL_SOURCES
    label: str | None = None  # class label (K1) or noun phrase (K2)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class SamplerStats:
    """Why candidates were rejected, so drop rates are diagnosable per set and bin."""

    considered: int = 0
    rejected: Counter = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.rejected[reason] += 1

    def as_dict(self) -> dict[str, int]:
        return {"considered": self.considered, **{f"rejected_{k}": v for k, v in self.rejected.items()}}


@dataclass(frozen=True)
class ControlChoice:
    """A sampled control edit, with the P3 matching quantities that licensed it."""

    control_instance_id: str | None
    control_source: str  # a CONTROL_SOURCES value, or 'background' for CONTROL_BG
    hole: np.ndarray  # the dilated hole actually edited
    mask: np.ndarray  # the undilated instance mask (== hole shape for CONTROL_BG)
    control_area_ratio: float
    control_centrality_delta: float


def hole_for(mask: np.ndarray, hole_type: str, dilation_px: int = HOLE_DILATION_PX) -> np.ndarray:
    """P2: mask hole = the mask dilated by 5 px; rectangular hole = its bbox."""
    if hole_type == "mask":
        return M.dilate(mask, dilation_px)
    if hole_type == "rect":
        return M.mask_from_box(M.bbox_xyxy(mask), mask.shape)
    raise ValueError(f"hole_type must be one of {HOLE_TYPES}, got {hole_type!r}")


def build_exclusion(
    referent: Instance,
    instances: list[Instance],
    *,
    referent_dilation_px: int = EXCLUSION_DILATION_PX,
    class_agnostic_min_area_frac: float = CLASS_AGNOSTIC_MIN_AREA_FRAC,
) -> np.ndarray:
    """The P3 exclusion set for one image."""
    excl = M.dilate(referent.mask, referent_dilation_px)
    for inst in instances:
        if inst.instance_id == referent.instance_id:
            continue
        if inst.source == "class_agnostic":
            if M.area_frac(inst.mask) <= class_agnostic_min_area_frac:
                continue
        excl |= inst.mask
    return excl


def _overlaps_exclusion(hole: np.ndarray, exclusion: np.ndarray, own: np.ndarray) -> int:
    """Pixels where the hole meets the exclusion set minus the candidate's own mask.

    The literal reading of P3.  Kept because acceptance check 1c-style asserts and
    the CONTROL_BG rule still use it; CONTROL_OBJ uses
    :func:`exclusion_violation` (Q-12, reading B).
    """
    return int(np.count_nonzero(hole & exclusion & ~own))


def exclusion_violation(
    hole: np.ndarray,
    candidate_mask: np.ndarray,
    referent: Instance,
    instances: list[Instance],
    *,
    candidate_id: str | None = None,
    tolerance: float = CONTROL_BITE_TOLERANCE,
    referent_dilation_px: int = EXCLUSION_DILATION_PX,
    class_agnostic_min_area_frac: float = CLASS_AGNOSTIC_MIN_AREA_FRAC,
) -> str | None:
    """P3's overlap rule under reading B (OPEN-QUESTIONS Q-12).

    The referent is absolute: a control hole may not touch its dilated mask at
    all, or the control would damage the very thing the REMOVE edit is supposed
    to be the only edit of.  Every *other* excluded instance tolerates a nick of
    at most ``tolerance`` of its area, measured outside the candidate itself,
    because a dense automatic segmentation puts every object in contact with its
    neighbours and the literal rule then yields a CONTROL_OBJ on 8% of images.

    The bite is measured against the other instance's *whole* area: a mask lying
    mostly inside the candidate is a part of the candidate and disappears with
    it, which is correct rather than damage.

    Returns ``None`` when the candidate is admissible, else a short reason.
    """
    if np.any(hole & M.dilate(referent.mask, referent_dilation_px)):
        return "touches the referent"
    for other in instances:
        if other.instance_id in (candidate_id, referent.instance_id):
            continue
        if other.source == "referent":
            continue
        if (
            other.source == "class_agnostic"
            and M.area_frac(other.mask) <= class_agnostic_min_area_frac
        ):
            continue
        area = int(other.mask.sum())
        if area == 0:
            continue
        bitten = int(np.count_nonzero(hole & other.mask & ~candidate_mask)) / area
        if bitten > tolerance:
            return f"bites {bitten:.0%} of {other.instance_id}"
    return None


def sample_control_obj(
    referent: Instance,
    instances: list[Instance],
    exclusion: np.ndarray,
    *,
    hole_type: str = "mask",
    rng: np.random.Generator | None = None,
    exclude_instance_ids: tuple[str, ...] = (),
    stats: SamplerStats | None = None,
    max_tries: int = MAX_TRIES,
) -> ControlChoice | None:
    """Sample one CONTROL_OBJ instance, or return None if the item must be dropped.

    ``exclude_instance_ids`` lets a caller draw the *second* independent
    CONTROL_OBJ that acceptance check 1b needs (two edits per image, one labelled
    each way, whose AUROC must be 0.5).
    """
    rng = rng or np.random.default_rng(0)
    stats = stats if stats is not None else SamplerStats()
    lo, hi = AREA_RATIO_RANGE

    tiers: dict[str, list[Instance]] = {s: [] for s in CONTROL_SOURCES}
    for inst in instances:
        if inst.instance_id == referent.instance_id or inst.instance_id in exclude_instance_ids:
            continue
        if inst.source in tiers:
            tiers[inst.source].append(inst)

    tries = 0
    for source in CONTROL_SOURCES:  # P3 preference order, fallback last
        pool = tiers[source]
        if not pool:
            continue
        for idx in rng.permutation(len(pool)):
            if tries >= max_tries:
                stats.reject("max_tries")
                return None
            tries += 1
            cand = pool[int(idx)]
            stats.considered += 1
            ratio = M.area_ratio(cand.mask, referent.mask)
            if not (lo <= ratio <= hi):
                stats.reject("area")
                continue
            delta = M.centrality_delta(cand.mask, referent.mask)
            if abs(delta) > CENTRALITY_TOLERANCE:
                stats.reject("centrality")
                continue
            hole = hole_for(cand.mask, hole_type)
            violation = exclusion_violation(
                hole, cand.mask, referent, instances, candidate_id=cand.instance_id
            )
            if violation is not None:
                stats.reject(
                    "referent_contact" if "referent" in violation else "exclusion_overlap"
                )
                continue
            return ControlChoice(
                control_instance_id=cand.instance_id,
                control_source=source,
                hole=hole,
                mask=cand.mask,
                control_area_ratio=float(ratio),
                control_centrality_delta=float(delta),
            )
    stats.reject("no_candidate")
    return None


def sample_control_bg(
    referent: Instance,
    exclusion: np.ndarray,
    *,
    hole_type: str = "mask",
    rng: np.random.Generator | None = None,
    max_tries: int = MAX_TRIES,
    stats: SamplerStats | None = None,
) -> ControlChoice | None:
    """P3 CONTROL_BG: the referent's hole shape moved somewhere with no overlap."""
    rng = rng or np.random.default_rng(0)
    stats = stats if stats is not None else SamplerStats()
    hole = hole_for(referent.mask, hole_type)
    ys, xs = np.nonzero(hole)
    if xs.size == 0:
        stats.reject("empty_referent")
        return None
    h, w = hole.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    # keep the whole shape inside the frame
    dx_range = (-x0, w - 1 - x1)
    dy_range = (-y0, h - 1 - y1)
    if dx_range[0] > dx_range[1] or dy_range[0] > dy_range[1]:
        stats.reject("no_room")
        return None

    for _ in range(max_tries):
        stats.considered += 1
        dx = int(rng.integers(dx_range[0], dx_range[1] + 1))
        dy = int(rng.integers(dy_range[0], dy_range[1] + 1))
        moved = M.translate(hole, dx, dy)
        if moved.sum() != hole.sum():  # clipped at the frame edge
            stats.reject("clipped")
            continue
        if np.any(moved & exclusion):
            stats.reject("exclusion_overlap")
            continue
        return ControlChoice(
            control_instance_id=None,
            control_source="background",
            hole=moved,
            mask=moved,
            control_area_ratio=1.0,
            control_centrality_delta=float(
                M.centrality(moved).frac_of_diagonal
                - M.centrality(referent.mask).frac_of_diagonal
            ),
        )
    stats.reject("max_tries")
    return None


@dataclass(frozen=True)
class EditPlan:
    """One edit to materialize: which operator, which hole, which source."""

    operator: str  # REMOVE, CONTROL_OBJ, CONTROL_OBJ_2, CONTROL_BG (+ RECT_ variants)
    hole: np.ndarray
    mask: np.ndarray
    hole_type: str
    control_instance_id: str | None = None
    control_source: str | None = None
    control_area_ratio: float | None = None
    control_centrality_delta: float | None = None


def plan_k1_image(
    referent: Instance,
    instances: list[Instance],
    *,
    hole_type: str = "mask",
    rng: np.random.Generator | None = None,
    stats: SamplerStats | None = None,
    with_second_control: bool = True,
    with_background: bool = True,
) -> list[EditPlan] | None:
    """P4 for the K1 pool: one hole per image, plus the check-1b and CONTROL_BG rows.

    Returns ``None`` when no valid CONTROL_OBJ exists, which is the P3 drop.
    K1 needs, per image: the REMOVE edit, the matched CONTROL_OBJ edit, a
    *second* independent CONTROL_OBJ for the identical-distribution null of check
    1b, and a CONTROL_BG row for the comparison rows of P7.
    """
    rng = rng or np.random.default_rng(0)
    stats = stats if stats is not None else SamplerStats()
    exclusion = build_exclusion(referent, instances)

    first = sample_control_obj(
        referent, instances, exclusion, hole_type=hole_type, rng=rng, stats=stats
    )
    if first is None:
        return None

    prefix = "" if hole_type == "mask" else "RECT_"
    plans = [
        EditPlan(
            operator=f"{prefix}REMOVE",
            hole=hole_for(referent.mask, hole_type),
            mask=referent.mask,
            hole_type=hole_type,
        ),
        EditPlan(
            operator=f"{prefix}CONTROL_OBJ",
            hole=first.hole,
            mask=first.mask,
            hole_type=hole_type,
            control_instance_id=first.control_instance_id,
            control_source=first.control_source,
            control_area_ratio=first.control_area_ratio,
            control_centrality_delta=first.control_centrality_delta,
        ),
    ]
    if with_second_control:
        second = sample_control_obj(
            referent,
            instances,
            exclusion,
            hole_type=hole_type,
            rng=rng,
            exclude_instance_ids=(first.control_instance_id,),
            stats=stats,
        )
        if second is not None:
            plans.append(
                EditPlan(
                    operator=f"{prefix}CONTROL_OBJ_2",
                    hole=second.hole,
                    mask=second.mask,
                    hole_type=hole_type,
                    control_instance_id=second.control_instance_id,
                    control_source=second.control_source,
                    control_area_ratio=second.control_area_ratio,
                    control_centrality_delta=second.control_centrality_delta,
                )
            )
    if with_background:
        bg = sample_control_bg(referent, exclusion, hole_type=hole_type, rng=rng, stats=stats)
        if bg is not None:
            plans.append(
                EditPlan(
                    operator=f"{prefix}CONTROL_BG",
                    hole=bg.hole,
                    mask=bg.mask,
                    hole_type=hole_type,
                    control_source=bg.control_source,
                    control_area_ratio=bg.control_area_ratio,
                    control_centrality_delta=bg.control_centrality_delta,
                )
            )
    return plans


def plan_k2_item(
    referent: Instance,
    instances: list[Instance],
    *,
    hole_type: str = "mask",
    rng: np.random.Generator | None = None,
    stats: SamplerStats | None = None,
) -> list[EditPlan] | None:
    """P4 for K2: every item gets ORIGINAL, REMOVE, CONTROL_OBJ and CONTROL_BG.

    ORIGINAL needs no edit, so this returns the three edits.
    """
    return plan_k1_image(
        referent,
        instances,
        hole_type=hole_type,
        rng=rng,
        stats=stats,
        with_second_control=False,
        with_background=True,
    )


def assert_no_exclusion_overlap(
    plans: list[EditPlan], referent: Instance, instances: list[Instance]
) -> None:
    """Acceptance check 2, as an independent pass over a built plan.

    Asserts the rule the sampler actually sampled under (P3 reading B): the
    referent is untouched absolutely, and no other excluded instance is bitten by
    more than :data:`CONTROL_BITE_TOLERANCE`.  A CONTROL_BG hole is background by
    construction and is held to the strict rule.
    """
    exclusion = build_exclusion(referent, instances)
    by_id = {i.instance_id: i for i in instances}
    for plan in plans:
        if "REMOVE" in plan.operator:
            continue  # the referent hole is meant to sit on the referent
        if plan.control_source == "background":
            n = _overlaps_exclusion(plan.hole, exclusion, np.zeros_like(exclusion))
            if n:
                raise AssertionError(
                    f"{plan.operator} background hole overlaps the P3 exclusion set in "
                    f"{n} pixels; acceptance check 2"
                )
            continue
        own = (
            by_id[plan.control_instance_id].mask
            if plan.control_instance_id in by_id
            else np.zeros_like(exclusion)
        )
        violation = exclusion_violation(
            plan.hole, own, referent, instances, candidate_id=plan.control_instance_id
        )
        if violation is not None:
            raise AssertionError(
                f"{plan.operator} hole violates the P3 exclusion rule: {violation} "
                f"(control_instance_id={plan.control_instance_id}); acceptance check 2"
            )
