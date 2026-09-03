"""The gate's inputs (design P5) and the row registry the P7 verdict reads.

P5, in full:

    Location-agnostic rows, no hole coordinates given to any classifier: (i)
    ResNet-18 on the full edited image at the smaller of native size and the 2.4
    Mpx cap; (ii) ViT-S/16 on native-resolution 512 px tiles covering the image,
    each tile inheriting the image label, aggregated per image by the maximum
    tile score; (iii) both classifiers on the full image at 1,024 px longer side.
    Reported rows: both classifiers at 512 px longer side; a shown crop centred
    on the hole (1.5 times the hole's longer side, resized to 256).  Paired crops
    outside both holes: the null of check 1c only.

"The gate must see what the policy sees": the 2.4 Mpx cap is the P21 setting the
models under test run at, so a gate that only looks at 512 px would clear an
editor whose artefacts a policy can still read.

No transform here takes a hole coordinate.  :func:`shown_crop` does -- and it is
registered as a *reported* row, never a gate row, for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from idea91.masks import Box, longer_side

MPX_CAP = 2_457_600  # P21: max_pixels for Qwen3-VL, the resolution a policy sees
TILE_PX = 512
SHOWN_CROP_FACTOR = 1.5
SHOWN_CROP_PX = 256

RESNET18 = "resnet18"
VIT_S16 = "vit_s16"
DINOV2_B = "dinov2_b"
FORENSIC = "forensic"


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (size[0] * size[1]) < (w * h) else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interp)


def full_at_cap(image: np.ndarray, cap: int = MPX_CAP) -> np.ndarray:
    """Row (i): native size, or the 2.4 Mpx cap, whichever is smaller.

    Downscale only: an image below the cap is passed through untouched, which is
    what "the smaller of native size and the cap" means.
    """
    h, w = image.shape[:2]
    if w * h <= cap:
        return image
    scale = float(np.sqrt(cap / float(w * h)))
    return _resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))


def full_at_longer_side(image: np.ndarray, px: int) -> np.ndarray:
    """Rows (iii) and the reported 512 px row.  Downscale only."""
    h, w = image.shape[:2]
    if max(h, w) <= px:
        return image
    scale = px / float(max(h, w))
    return _resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))


def tile_positions(length: int, tile: int) -> list[int]:
    """Start offsets covering ``length`` with ``tile``-wide tiles.

    The last tile is pulled back to the edge rather than padded, so every tile
    carries real pixels and the max-aggregation is not fed a border artefact.
    """
    if length <= tile:
        return [0]
    steps = int(np.ceil(length / tile))
    last = length - tile
    return sorted({min(i * tile, last) for i in range(steps)} | {last})


def tiles(
    image: np.ndarray,
    tile: int = TILE_PX,
    *,
    max_tiles: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Row (ii): native-resolution tiles covering the image.

    ``max_tiles`` subsamples for training (a 7,680 px image is 180 tiles); the
    evaluation pass leaves it None so the per-image max is over full coverage.
    """
    h, w = image.shape[:2]
    out = [
        image[y : y + min(tile, h), x : x + min(tile, w)]
        for y in tile_positions(h, tile)
        for x in tile_positions(w, tile)
    ]
    if max_tiles is not None and len(out) > max_tiles:
        rng = rng or np.random.default_rng(0)
        pick = rng.choice(len(out), size=max_tiles, replace=False)
        out = [out[int(i)] for i in sorted(pick)]
    return out


def shown_crop(
    image: np.ndarray,
    hole_box: Box,
    factor: float = SHOWN_CROP_FACTOR,
    out_px: int = SHOWN_CROP_PX,
) -> np.ndarray:
    """Reported row: a crop centred on the hole, 1.5x its longer side, at 256 px.

    This one *is* told where the hole is, which is why P5 keeps it out of the
    gate rows: it answers "can a classifier see the edit when shown it", not
    "can a policy find the edit on its own".
    """
    h, w = image.shape[:2]
    side = max(8.0, factor * longer_side(hole_box))
    cx = (hole_box[0] + hole_box[2]) / 2.0
    cy = (hole_box[1] + hole_box[3]) / 2.0
    x0 = int(np.clip(round(cx - side / 2), 0, max(0, w - 1)))
    y0 = int(np.clip(round(cy - side / 2), 0, max(0, h - 1)))
    x1 = int(np.clip(round(cx + side / 2), x0 + 1, w))
    y1 = int(np.clip(round(cy + side / 2), y0 + 1, h))
    return _resize(image[y0:y1, x0:x1], (out_px, out_px))


def sample_paired_crop_box(
    image_wh: tuple[int, int],
    hole_boxes: list[Box],
    *,
    side: int = SHOWN_CROP_PX,
    rng: np.random.Generator | None = None,
    max_tries: int = 200,
) -> Box | None:
    """A crop box that touches none of ``hole_boxes`` (the check 1c null row)."""
    from idea91.edits.composite import boxes_overlap

    rng = rng or np.random.default_rng(0)
    w, h = image_wh
    if w <= side or h <= side:
        return None
    for _ in range(max_tries):
        x0 = int(rng.integers(0, w - side))
        y0 = int(rng.integers(0, h - side))
        box = (float(x0), float(y0), float(x0 + side), float(y0 + side))
        if not any(boxes_overlap(box, hb) for hb in hole_boxes):
            return box
    return None


# --- the row registry --------------------------------------------------------


@dataclass(frozen=True)
class GateRow:
    """One (input, classifier) pair, and whether it can fail the P7 gate."""

    name: str
    classifier: str
    input_kind: str  # full_cap | full_1024 | full_512 | tiles_512 | shown_crop | paired_crop
    gating: bool
    aggregate: str = "image"  # 'image' or 'max_tile'
    note: str = ""


#: P7: "the maximum over the gate rows (i), (ii) and (iii) of the AUROC of
#: ResNet-18 and of ViT-S/16 ... is at most 0.6, on both hole types."
GATE_ROWS: tuple[GateRow, ...] = (
    GateRow("i_resnet18_full_cap", RESNET18, "full_cap", True, note="P5 row (i)"),
    GateRow("ii_vit_s16_tiles_native", VIT_S16, "tiles_512", True, "max_tile", "P5 row (ii)"),
    GateRow("iii_resnet18_1024", RESNET18, "full_1024", True, note="P5 row (iii)"),
    GateRow("iii_vit_s16_1024", VIT_S16, "full_1024", True, note="P5 row (iii)"),
)

#: Reported, never gating.
REPORTED_ROWS: tuple[GateRow, ...] = (
    GateRow("rep_resnet18_512", RESNET18, "full_512", False),
    GateRow("rep_vit_s16_512", VIT_S16, "full_512", False),
    GateRow("rep_resnet18_shown_crop", RESNET18, "shown_crop", False),
    GateRow("rep_vit_s16_shown_crop", VIT_S16, "shown_crop", False),
    GateRow("null_resnet18_paired_crop", RESNET18, "paired_crop", False, note="check 1c null"),
)

#: P6 strong adversaries: reported without a hard gate, one seed each.  A
#: DINOv2-B AUROC above 0.7 on any gate row is carried into the first table of
#: every downstream use (P7).
ADVERSARY_ROWS: tuple[GateRow, ...] = (
    GateRow("adv_dinov2b_1024", DINOV2_B, "full_1024", False),
    GateRow("adv_dinov2b_tiles_native", DINOV2_B, "tiles_512", False, "max_tile"),
    GateRow("adv_forensic_full_cap", FORENSIC, "full_cap", False),
)

ALL_ROWS: tuple[GateRow, ...] = GATE_ROWS + REPORTED_ROWS + ADVERSARY_ROWS
ROWS_BY_NAME = {r.name: r for r in ALL_ROWS}


def render(image: np.ndarray, row: GateRow, hole_box: Box | None = None, **kwargs):
    """Apply a row's input transform.  Returns an image, or a list for tile rows."""
    if row.input_kind == "full_cap":
        return full_at_cap(image)
    if row.input_kind == "full_1024":
        return full_at_longer_side(image, 1024)
    if row.input_kind == "full_512":
        return full_at_longer_side(image, 512)
    if row.input_kind == "tiles_512":
        return tiles(image, TILE_PX, **kwargs)
    if row.input_kind == "shown_crop":
        if hole_box is None:
            raise ValueError(f"{row.name} needs hole_box")
        return shown_crop(image, hole_box)
    if row.input_kind == "paired_crop":
        box = kwargs.get("crop_box")
        if box is None:
            raise ValueError(f"{row.name} needs crop_box")
        from idea91.edits.composite import crop

        return crop(image, box)
    raise ValueError(f"unknown input kind {row.input_kind!r}")
