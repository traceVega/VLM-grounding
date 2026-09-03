"""The edit verifier's decision procedure (design P10).

The three-way outcome is decided *before the question is asked*, from geometry
alone, so the verifier's answer can never redefine which rule applied:

1. If any other SAM 3 instance of the head noun overlaps the edit window, the
   item is undetermined by a same-noun neighbour and goes to the secondary rule.
2. Otherwise ask, on the edit window: "Is there a {head noun} in this image?
   Answer with one word, yes or no."  "no" is clean, "yes" is not clean, an
   unparseable answer counts as not clean.
3. Secondary rule: the same question on a tight crop -- R dilated by 50% of its
   size, at least 256 px on the longer side.  If a neighbour overlaps the tight
   crop too, the item *stays undetermined*: it leaves every verifier-based clean
   statistic and leaves kappa, and its share is reported per set and dimension.

Why the head noun and not the full expression (design deviation 4): relational
context usually lies outside the edit window, so asking the full expression on
the window would answer "no" whether or not remnants remain.  The head noun is
what the removal changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np

from idea91 import masks as M
from idea91.masks import Box

TIGHT_CROP_DILATION = 0.5  # P10: R dilated by 50% of its size
TIGHT_CROP_MIN_SIDE = 256  # P10: at least 256 px on the longer side
WINDOW_DISPLAY_MAX_PX = 1536  # P10: the edit window is shown at up to 1,536 px
VERIFIER_MAX_TOKENS = 4  # P10
VERIFIER_TEMPERATURE = 0.0  # P10


class Clean(str, Enum):
    CLEAN = "clean"
    NOT_CLEAN = "not_clean"
    UNDETERMINED = "undetermined"

    @property
    def is_clean(self) -> bool | None:
        if self is Clean.UNDETERMINED:
            return None
        return self is Clean.CLEAN


_FIRST_WORD = re.compile(r"[a-z]+")


def parse_yes_no(text: str | None) -> bool | None:
    """First-word match on yes or no (P10).  ``None`` means unparseable."""
    if not text:
        return None
    m = _FIRST_WORD.search(text.strip().lower())
    if not m:
        return None
    word = m.group(0)
    if word in ("yes", "yeah", "yep"):
        return True
    if word in ("no", "nope"):
        return False
    return None


def tight_crop_box(
    region: np.ndarray | Box,
    image_wh: tuple[int, int],
    *,
    dilation: float = TIGHT_CROP_DILATION,
    min_side: int = TIGHT_CROP_MIN_SIDE,
) -> Box:
    """P10's tight crop: R grown by 50% of its size, at least 256 px long side."""
    box = M.bbox_xyxy(region) if isinstance(region, np.ndarray) else region
    w, h = image_wh
    bw, bh = box[2] - box[0], box[3] - box[1]
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    side_w, side_h = bw * (1 + dilation), bh * (1 + dilation)
    if max(side_w, side_h) < min_side:
        scale = min_side / max(side_w, side_h)
        side_w, side_h = side_w * scale, side_h * scale

    def axis(centre: float, extent: float, limit: int) -> tuple[float, float]:
        # slide inside the frame rather than clip, so a referent near an edge
        # still gets a crop of the required size (only a small image shrinks it)
        span = min(extent, float(limit))
        lo = min(max(0.0, centre - span / 2.0), float(limit) - span)
        return lo, lo + span

    x0, x1 = axis(cx, side_w, w)
    y0, y1 = axis(cy, side_h, h)
    return (x0, y0, x1, y1)


def overlaps(box: Box, mask: np.ndarray) -> bool:
    """Does a mask have any pixel inside a box?"""
    x0 = max(0, int(np.floor(box[0])))
    y0 = max(0, int(np.floor(box[1])))
    x1 = min(mask.shape[1], int(np.ceil(box[2])))
    y1 = min(mask.shape[0], int(np.ceil(box[3])))
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(mask[y0:y1, x0:x1].any())


@dataclass(frozen=True)
class CleanDecision:
    outcome: Clean
    rule: str  # 'window', 'tight_crop', or 'neighbour_in_both'
    question: str | None  # the VERIFIER_QUESTIONS key that was asked, if any
    tight_crop: Box | None = None


def route(
    window_box: Box,
    removal_region: np.ndarray | Box,
    neighbour_masks: list[np.ndarray],
    image_wh: tuple[int, int],
) -> CleanDecision:
    """Decide *which* question P10 asks, before any answer exists.

    Returns a decision whose ``outcome`` is UNDETERMINED only when a same-noun
    neighbour overlaps both the window and the tight crop; otherwise the outcome
    is a placeholder and ``question`` names the question to ask.
    """
    if not any(overlaps(window_box, m) for m in neighbour_masks):
        return CleanDecision(Clean.UNDETERMINED, "window", "V1_headnoun_window")
    crop = tight_crop_box(removal_region, image_wh)
    if any(overlaps(crop, m) for m in neighbour_masks):
        return CleanDecision(Clean.UNDETERMINED, "neighbour_in_both", None, crop)
    return CleanDecision(Clean.UNDETERMINED, "tight_crop", "V1_headnoun_tight_crop", crop)


def decide(decision: CleanDecision, answer_text: str | None) -> Clean:
    """Apply P10's answer rule to the question :func:`route` selected.

    "no" (the head noun is gone) is clean; "yes" is not clean; unparseable
    counts as not clean.
    """
    if decision.question is None:
        return Clean.UNDETERMINED
    parsed = parse_yes_no(answer_text)
    if parsed is None:
        return Clean.NOT_CLEAN  # "unparseable counted as not clean"
    return Clean.NOT_CLEAN if parsed else Clean.CLEAN


def clean_for_remove(
    window_box: Box,
    removal_region: np.ndarray | Box,
    neighbour_masks: list[np.ndarray],
    image_wh: tuple[int, int],
    ask,
) -> tuple[Clean, CleanDecision]:
    """Route, ask, decide.  ``ask(question, crop_box) -> answer text``."""
    decision = route(window_box, removal_region, neighbour_masks, image_wh)
    if decision.question is None:
        return Clean.UNDETERMINED, decision
    crop = decision.tight_crop if decision.rule == "tight_crop" else window_box
    return decide(decision, ask(decision.question, crop)), decision


def control_is_valid(answer_text: str | None) -> bool:
    """P10 validity rule: the head noun must still be *present* in a control edit.

    Asked on the referent's window in the control image, where the referent is
    untouched; an unparseable answer fails validity, matching the conservative
    reading the clean rule uses.
    """
    return parse_yes_no(answer_text) is True


def removal_success_verifiable(control_source: str | None) -> bool:
    """P10: class-agnostic controls have no label to ask about.

    They are marked unverified (V2 only) and are never mixed into the
    verified-pairs row of the K1 table.
    """
    return control_source not in (None, "class_agnostic", "background")
