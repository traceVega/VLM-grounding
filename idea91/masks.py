"""Mask and box primitives for the edit stack (design P2, P3, P9).

Masks are stored as COCO RLE so a 7,680 px GroundingME image costs kilobytes in
``edits/index.parquet`` instead of megabytes.  Everything here is pure geometry:
no model, no I/O, so the sampler's assertions (acceptance check 2) and the
compositing assert (check 1c) are unit-testable without SAM 3 or LaMa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
from pycocotools import mask as mask_utils

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in original pixels


# --- RLE ---------------------------------------------------------------------


def encode_rle(mask: np.ndarray) -> str:
    """Binary mask -> a JSON string holding COCO RLE (portable inside parquet)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(mask))
    return json.dumps({"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")})


def decode_rle(rle: str) -> np.ndarray:
    d = json.loads(rle)
    return mask_utils.decode(
        {"size": d["size"], "counts": d["counts"].encode("ascii")}
    ).astype(bool)


# --- shape facts -------------------------------------------------------------


def area_frac(mask: np.ndarray) -> float:
    return float(mask.sum()) / float(mask.size)


def bbox_xyxy(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty mask has no bounding box")
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty mask has no centroid")
    return (float(xs.mean()), float(ys.mean()))


def longer_side(box: Box) -> float:
    return max(box[2] - box[0], box[3] - box[1])


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def box_to_mask_iou(box: Box, mask: np.ndarray) -> float:
    """P9: the gate on whether a ground-truth box gets a mask hole or a rect hole."""
    return box_iou(box, bbox_xyxy(mask)) if mask.any() else 0.0


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate by ``px`` with a disc, the P2/P3 dilation (5 px holes, 16 px exclusion)."""
    if px <= 0:
        return mask.copy()
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def mask_from_box(box: Box, shape: tuple[int, int]) -> np.ndarray:
    """The rectangular hole of P2, used when box-to-mask IoU is below 0.5 (P8)."""
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    x0 = int(np.clip(np.floor(box[0]), 0, w))
    y0 = int(np.clip(np.floor(box[1]), 0, h))
    x1 = int(np.clip(np.ceil(box[2]), 0, w))
    y1 = int(np.clip(np.ceil(box[3]), 0, h))
    m[y0:y1, x0:x1] = True
    return m


def translate(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a mask, dropping whatever leaves the frame (CONTROL_BG placement)."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx0 >= sx1 or sy0 >= sy1:
        return out
    out[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = mask[sy0:sy1, sx0:sx1]
    return out


# --- P3 matching quantities --------------------------------------------------


@dataclass(frozen=True)
class Centrality:
    """Distance of a mask's centroid from the image centre, in pixels and in diagonals."""

    px: float
    frac_of_diagonal: float


def centrality(mask: np.ndarray) -> Centrality:
    h, w = mask.shape
    cx, cy = centroid(mask)
    d = float(np.hypot(cx - w / 2.0, cy - h / 2.0))
    diag = float(np.hypot(w, h))
    return Centrality(px=d, frac_of_diagonal=d / diag if diag else 0.0)


def area_ratio(candidate: np.ndarray, referent: np.ndarray) -> float:
    ref = float(referent.sum())
    return float(candidate.sum()) / ref if ref else float("inf")


def centrality_delta(candidate: np.ndarray, referent: np.ndarray) -> float:
    """Signed difference in centre distance, as a fraction of the image diagonal.

    P3 matches |delta| <= 0.20.
    """
    return centrality(candidate).frac_of_diagonal - centrality(referent).frac_of_diagonal


def size_bin_of(box: Box) -> str:
    """SPEC Section 3 ``size_bin`` by longer side of the first ground-truth box.

    Thresholds follow the usual referring-expression convention (IDEA-11 D3):
    tiny < 32 px, small < 96, medium < 224, large < 448, xl otherwise.
    """
    s = longer_side(box)
    if s < 32:
        return "tiny"
    if s < 96:
        return "small"
    if s < 224:
        return "medium"
    if s < 448:
        return "large"
    return "xl"
