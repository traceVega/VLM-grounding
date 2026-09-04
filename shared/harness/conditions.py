"""P13's conditions: what image and what expression each one sends (design 4.6).

    ORIGINAL     the untouched image, the item's own expression
    REMOVE       the referent removed
    CONTROL_OBJ  a matched non-referent object removed
    CONTROL_BG   a background patch removed
    T_NULL       the untouched image, the expression replaced by "the object"
    T_HEAD       the untouched image, the head noun replaced by an absent category
    T_ATTR       the untouched image, the discriminating attribute swapped

The four image conditions differ from ORIGINAL only in pixels; the three text
conditions differ only in words.  Keeping that split honest is the whole point
of the module: a control that varies in two ways at once measures neither.

That is also why :func:`t_null_expression` renders the *same* template as the
real condition.  P13 says "the grounding instruction with the expression
replaced by 'the object'", and a second prompt file holding a copy of the
instruction would be free to drift from it -- at which point T_NULL differs from
ORIGINAL in the instruction as well as the expression, and its box rate stops
being a baseline for anything.

T_HEAD needs a category the image does not contain, which is a GPU question
(P13: "a category from a fixed list of 50 common categories whose SAM 3 concept
prompt returns no instance in the image, chosen with seed 0").  The choosing
lives with the instances code; this module takes the chosen category and does
the rewriting, which is where the subtleties are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ORIGINAL = "ORIGINAL"
REMOVE = "REMOVE"
CONTROL_OBJ = "CONTROL_OBJ"
CONTROL_BG = "CONTROL_BG"
T_NULL = "T_NULL"
T_HEAD = "T_HEAD"
T_ATTR = "T_ATTR"

#: Conditions that change the image and leave the words alone.
IMAGE_CONDITIONS = (REMOVE, CONTROL_OBJ, CONTROL_BG)
#: Conditions that change the words and leave the image alone.
TEXT_CONDITIONS = (T_NULL, T_HEAD, T_ATTR)

#: P13's replacement expression for T_NULL, verbatim.
NULL_EXPRESSION = "the object"

#: P13's "fixed list of 50 common categories".  Frozen at B0b with P8 to P21.
#: Chosen to be concrete, picturable and easy for a segmenter to be sure about,
#: so that "SAM 3 finds none of these in the image" is a meaningful filter; and
#: to span indoor, outdoor, animal and vehicle, so a scene of any kind still has
#: candidates left after the absent-category filter.
T_HEAD_CATEGORIES: tuple[str, ...] = (
    "accordion", "anchor", "avocado", "banjo", "barbell",
    "birdcage", "blender", "bulldozer", "cactus", "camel",
    "candelabra", "canoe", "chandelier", "chess board", "combine harvester",
    "crocodile", "dartboard", "dumbbell", "elephant", "fire hydrant",
    "flamingo", "globe", "gramophone", "hammock", "harp",
    "hot air balloon", "igloo", "jellyfish", "kayak", "lighthouse",
    "llama", "microscope", "octopus", "ostrich", "parachute",
    "peacock", "penguin", "pineapple", "polar bear", "pyramid",
    "rocking horse", "sewing machine", "snowmobile", "stethoscope", "sundial",
    "telescope", "tractor", "trombone", "unicycle", "windmill",
)


@dataclass(frozen=True)
class Resolved:
    """What actually gets sent for one (item, condition)."""

    condition: str
    expr: str
    #: which edit to compose, or ``None`` for the untouched image
    operator: str | None
    note: str = ""

    @property
    def uses_edited_image(self) -> bool:
        return self.operator is not None


def t_null_expression() -> str:
    return NULL_EXPRESSION


def _noun_pattern(noun: str) -> re.Pattern[str]:
    r"""Match ``noun`` as a whole word, case-insensitively, plural tolerated.

    ``\b`` alone would match "car" inside "carpet"; the trailing ``(?:s|es)?``
    catches the plural the expression may well use, and the lookahead stops it
    swallowing a longer word that merely starts the same way.
    """
    escaped = re.escape(noun.strip())
    return re.compile(rf"\b{escaped}(?:es|s)?\b", re.IGNORECASE)


def swap_head_noun(expr: str, head_noun: str, replacement: str) -> tuple[str, int]:
    """P13's T_HEAD rewrite.  Returns the new expression and how many hits.

    Every occurrence is replaced, not just the first.  GroundingME's expressions
    are paragraph-shaped -- median 221 characters, and 68% open with "The object
    is ..." -- so the head noun frequently recurs, and leaving later mentions in
    place would produce an expression naming *both* nouns.  That is not the
    control P13 describes: the point is an expression whose head noun is absent
    from the image, and one surviving mention of the real noun would let the
    model ground on it.

    The count comes back so a caller can tell a rewrite that fired from one that
    did nothing -- zero hits means the head noun does not appear in its own
    expression, which is a data problem worth surfacing, not a silent no-op.
    """
    if not head_noun.strip():
        return expr, 0
    pattern = _noun_pattern(head_noun)
    hits = len(pattern.findall(expr))
    if hits == 0:
        return expr, 0

    def _replace(match: re.Match[str]) -> str:
        matched = match.group(0)
        # Keep the original's case shape so the sentence still reads: an
        # expression starting "The car ..." must not become "The tractor ..."
        # only if it was capitalised, and "Car" -> "Tractor".
        if matched[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return pattern.sub(_replace, expr), hits


def resolve(
    condition: str,
    *,
    expr: str,
    head_noun: str | None = None,
    absent_category: str | None = None,
    attr_swapped_expr: str | None = None,
    hole_type: str = "mask",
) -> Resolved:
    """One (item, condition) to the expression and the edit it needs."""
    if condition == ORIGINAL:
        return Resolved(ORIGINAL, expr, None)

    if condition in IMAGE_CONDITIONS:
        prefix = "" if hole_type == "mask" else "RECT_"
        return Resolved(condition, expr, f"{prefix}{condition}")

    if condition == T_NULL:
        return Resolved(T_NULL, t_null_expression(), None)

    if condition == T_HEAD:
        if not head_noun or not absent_category:
            raise ValueError(
                "T_HEAD needs the item's head noun and a category SAM 3 found no "
                "instance of in this image (P13); neither is guessable here"
            )
        rewritten, hits = swap_head_noun(expr, head_noun, absent_category)
        if hits == 0:
            raise ValueError(
                f"T_HEAD found no occurrence of head noun {head_noun!r} in the "
                f"expression, so the control would be identical to ORIGINAL: {expr[:80]!r}"
            )
        return Resolved(T_HEAD, rewritten, None, note=f"{head_noun}->{absent_category} x{hits}")

    if condition == T_ATTR:
        if not attr_swapped_expr:
            raise ValueError(
                "T_ATTR needs the expression rewritten by Qwen3.5-9B text-only (P13); "
                "it is reported, not scored, and is never synthesised here"
            )
        return Resolved(T_ATTR, attr_swapped_expr, None)

    raise ValueError(f"unknown condition {condition!r}")


# --- the T_NULL baseline ------------------------------------------------------


def prior_box(
    image_wh: tuple[int, int], median_box_area_frac: float, aspect: float = 1.0
) -> tuple[float, float, float, float]:
    """P15's "centre-and-median-size prior box", the baseline T_NULL is read against.

    A model given "the object" and no other information can do no better than
    guess the middle of the image at the typical object size.  Reporting T_NULL's
    pass rate next to this says whether the text-side control actually removed
    information or merely moved the box somewhere that happened to score.
    """
    width, height = image_wh
    area = median_box_area_frac * width * height
    box_w = (area * aspect) ** 0.5
    box_h = area / box_w if box_w else 0.0
    cx, cy = width / 2.0, height / 2.0
    return (cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0)
