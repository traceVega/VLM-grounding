"""Model output to an ``outputs.parquet`` row (SPEC Section 4, design 4.6).

One parser per model family.  Each turns raw generated text into a
:class:`Parsed`: an ``output_type`` from SPEC's enum, a box or a point in
**original-image pixels**, and whether the parse succeeded.

Two things this module is deliberate about.

**Coordinates are converted here, once.**  Each model emits in its own frame --
Molmo2 in 0-1000 against the original image, Qwen3-VL in whatever Q-3 settles --
and everything downstream (P14's same-box IoU, P15's rates, the point-in-box
scorer) works in original-image pixels.  Converting anywhere else would leave
two frames in circulation, and the failure would be silent: a wrong constant
still yields in-range coordinates that look plausible on every row.  That is not
hypothetical -- the Molmo2 config carried ``percent_float``, Molmo v1's
convention, until the card was read (OPEN-QUESTIONS Q-2).

**The benchmark's own decoder is not copied.**  GroundingME parses a box, builds
four candidate readings and keeps whichever has the highest IoU with the
ground-truth box.  K2 cannot do that: the REMOVE condition has no ground-truth
box by construction, and P14 compares a REMOVE box against an ORIGINAL box, so
both have to be decoded in one frame chosen in advance.  One convention per
model, fixed in its YAML, applied to every condition.  The cost is that K2's
ORIGINAL-correct rate will not match GroundingME's published accuracy, which is
computed with the more generous best-of-four decode; that belongs in the K2
table rather than in a footnote (Q-3).

An unparseable answer is a row, never an exception: ``output_type='invalid'``
with ``parse_ok=False``, so the share of them is a reported number instead of a
hole in the data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

Box = tuple[float, float, float, float]
Point = tuple[float, float]

BOX = "box"
POINT = "point"
SET = "set"
NONE = "none"
INVALID = "invalid"
REFUSAL = "refusal"

#: SPEC Section 1's coordinate conventions.
RELATIVE_1000 = "relative_1000"
ABSOLUTE_RESIZED = "absolute_resized"
PERCENT_FLOAT = "percent_float"
LOC_TOKENS = "loc_tokens"


@dataclass(frozen=True)
class Parsed:
    """One model answer, normalised to original-image pixels."""

    output_type: str
    parse_ok: bool
    box_xyxy_px: Box | None = None
    point_xy_px: Point | None = None
    n_boxes: int = 0
    note: str = ""

    @property
    def abstained(self) -> bool:
        return self.output_type == NONE


def to_original_pixels(
    values: tuple[float, ...],
    convention: str,
    *,
    original_wh: tuple[int, int],
    sent_wh: tuple[int, int] | None = None,
) -> tuple[float, ...]:
    """Map a model's coordinates into original-image pixels.

    ``sent_wh`` is the size actually sent to the model, which differs from
    ``original_wh` whenever P21's cap downscaled the image -- on GroundingME
    that is 99.3% of items, so ``absolute_resized`` is the case that matters
    most and the one most easily got wrong.
    """
    width, height = original_wh
    xs = (0, 2) if len(values) == 4 else (0,)
    ys = (1, 3) if len(values) == 4 else (1,)

    if convention == RELATIVE_1000:
        scale_x, scale_y = width / 1000.0, height / 1000.0
    elif convention == PERCENT_FLOAT:
        scale_x, scale_y = width / 100.0, height / 100.0
    elif convention == ABSOLUTE_RESIZED:
        if sent_wh is None:
            raise ValueError(
                "absolute_resized needs the size actually sent: the model's pixels are "
                "in the resized frame, and P21's cap resizes almost every image"
            )
        scale_x, scale_y = width / float(sent_wh[0]), height / float(sent_wh[1])
    elif convention == LOC_TOKENS:
        raise NotImplementedError("loc_tokens has no model using it here yet")
    else:
        raise ValueError(f"unknown coordinate convention {convention!r}")

    out = list(values)
    for i in xs:
        out[i] = values[i] * scale_x
    for i in ys:
        out[i] = values[i] * scale_y
    return tuple(out)


def clamp_box(box: Box, original_wh: tuple[int, int]) -> Box:
    """Clip to the frame.  A box hanging off the edge is a real answer, not a
    parse failure, so it is clipped rather than rejected."""
    width, height = original_wh
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0.0, min(x0, width)), max(0.0, min(x1, width))))
    y0, y1 = sorted((max(0.0, min(y0, height)), max(0.0, min(y1, height))))
    return (x0, y0, x1, y1)


def looks_like_refusal(text: str) -> bool:
    """A safety refusal is its own output_type (P17), never a failed parse."""
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "i can't help",
            "i cannot help",
            "i'm unable to",
            "i am unable to",
            "i can't assist",
            "i cannot assist",
            "as an ai",
        )
    )


def matches_none_pattern(text: str, patterns: tuple[str, ...] | list[str]) -> bool:
    """Whether an answer is one of the model's abstentions (P12's none_patterns).

    Matched on word boundaries: "none" must not fire on "nonetheless", and a
    model that says "there is no such object" must not be missed because the
    pattern was written as a bare substring.
    """
    lowered = text.lower()
    for pattern in patterns:
        if not pattern or pattern == "PIN_REQUIRED":
            continue
        if re.search(rf"(?<![a-z]){re.escape(pattern.lower())}(?![a-z])", lowered):
            return True
    return False


# --- Qwen3-VL, under GroundingME's own protocol -------------------------------

#: The evaluator's own extraction: the first JSON object mentioning bbox_2d.
_BBOX_JSON = re.compile(r'\{[^{}]*"bbox_2d"[^{}]*\}', re.DOTALL)
#: Its fallback: the last four numbers anywhere in the text.
_FOUR_NUMBERS = re.compile(
    r"(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)"
)


def parse_qwen3vl(
    text: str,
    *,
    convention: str,
    original_wh: tuple[int, int],
    sent_wh: tuple[int, int] | None = None,
    none_patterns: tuple[str, ...] = (),
) -> Parsed:
    """P12's primary protocol: ``{"bbox_2d": [x1,y1,x2,y2]}`` or ``null``.

    The null is GroundingME's own rejection channel, so an abstention here is
    ``output_type='none'`` and not a parse failure -- which is what makes P16's
    abstention leg measurable under the primary protocol at all (Q-1).
    """
    if looks_like_refusal(text):
        return Parsed(REFUSAL, parse_ok=True, note="refusal")

    match = _BBOX_JSON.search(text)
    if match:
        try:
            value = json.loads(match.group(0))["bbox_2d"]
        except (ValueError, KeyError, TypeError):
            value = ...  # fall through to the number scan
        else:
            if value is None:
                return Parsed(NONE, parse_ok=True, note="bbox_2d: null")
            if isinstance(value, list) and len(value) == 4:
                return _box_result(tuple(float(v) for v in value), convention,
                                   original_wh, sent_wh)
            return Parsed(INVALID, parse_ok=False, note=f"bbox_2d was {value!r}")

    # The secondary protocol adds "output none", so an abstention can arrive as
    # a bare word with no JSON at all.
    if none_patterns and matches_none_pattern(text, none_patterns):
        return Parsed(NONE, parse_ok=True, note="none pattern")

    numbers = _FOUR_NUMBERS.findall(text)
    if numbers:
        return _box_result(tuple(float(v) for v in numbers[-1]), convention,
                           original_wh, sent_wh, note="recovered from loose numbers")
    return Parsed(INVALID, parse_ok=False, note="no bbox_2d and no four numbers")


def _box_result(
    values: tuple[float, ...],
    convention: str,
    original_wh: tuple[int, int],
    sent_wh: tuple[int, int] | None,
    note: str = "",
) -> Parsed:
    mapped = to_original_pixels(values, convention, original_wh=original_wh, sent_wh=sent_wh)
    box = clamp_box((mapped[0], mapped[1], mapped[2], mapped[3]), original_wh)
    if box[2] <= box[0] or box[3] <= box[1]:
        # A degenerate box is a real (bad) answer; P15 needs it counted, not lost.
        return Parsed(INVALID, parse_ok=False, box_xyxy_px=box, n_boxes=1,
                      note=f"degenerate box {box}")
    return Parsed(BOX, parse_ok=True, box_xyxy_px=box, n_boxes=1, note=note)


# --- Molmo2 -------------------------------------------------------------------

#: The card's own regexes, at revision e28fa2859.
_MOLMO_COORDS = re.compile(r'<(?:points|tracks)[^>]*?coords="([0-9\t:;, .]+)"\s*/?>')
_MOLMO_TRIPLE = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def parse_molmo2(
    text: str,
    *,
    original_wh: tuple[int, int],
    convention: str = RELATIVE_1000,
    none_patterns: tuple[str, ...] = (),
    sent_wh: tuple[int, int] | None = None,
) -> Parsed:
    """Molmo2's pointing output.

    Points arrive as ``<points ... coords="idx x y ...">`` with the coordinates
    scaled by 1000 against the **original** image -- the card's decoder says so
    and divides by 1000 accordingly (Q-2).  ``sent_wh`` is accepted and unused
    for that reason: Molmo2's frame does not depend on what P21 sent.

    P19 scores point-in-box, so several points collapse to ``output_type='set'``
    with the first kept as the scored point; an empty point list is an
    abstention, which is one of the two forms P12 allows.
    """
    if looks_like_refusal(text):
        return Parsed(REFUSAL, parse_ok=True, note="refusal")

    points: list[Point] = []
    offered = 0  # triples the model emitted, before the in-frame filter
    tagged = False
    for coords in _MOLMO_COORDS.finditer(text):
        tagged = True
        for triple in _MOLMO_TRIPLE.finditer(coords.group(1)):
            offered += 1
            _, raw_x, raw_y = triple.groups()
            x, y = to_original_pixels(
                (float(raw_x), float(raw_y)), convention, original_wh=original_wh,
                sent_wh=sent_wh,
            )
            if 0 <= x <= original_wh[0] and 0 <= y <= original_wh[1]:
                points.append((x, y))

    if points:
        kind = POINT if len(points) == 1 else SET
        return Parsed(kind, parse_ok=True, point_xy_px=points[0], n_boxes=len(points))

    if none_patterns and matches_none_pattern(text, none_patterns):
        return Parsed(NONE, parse_ok=True, note="none pattern")
    if tagged and offered == 0:
        # The tag was emitted with nothing in it: P12's "empty point list".
        return Parsed(NONE, parse_ok=True, note="empty point list")
    if tagged:
        # Points were offered and every one fell outside the frame. That is a
        # bad answer, not an abstention, and the difference is not cosmetic:
        # P16 (a) fires on the `none` rate over clean REMOVE and (b) on the
        # abstention rate over CONTROL_OBJ, so counting a wild point as an
        # abstention would push both rules towards firing on a model that never
        # declined anything.
        return Parsed(
            INVALID, parse_ok=False, n_boxes=offered,
            note=f"{offered} point(s), all outside the frame",
        )
    return Parsed(INVALID, parse_ok=False, note="no points tag and no none pattern")


# --- scoring ------------------------------------------------------------------


def iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return float(intersection / union) if union > 0 else 0.0


def point_in_box(point: Point, box: Box) -> bool:
    """P19's Molmo2 scorer.  Inclusive on the edges: a point on the boundary is
    inside the object as far as the annotation is concerned."""
    x, y = point
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


PARSERS = {"qwen3vl_box": parse_qwen3vl, "molmo2_point": parse_molmo2}
