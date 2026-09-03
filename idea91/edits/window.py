"""The window operator of design P2.

    "Editing runs on a window of 4 times the hole's longer side, at least 768 px
    and at most 2,048 px on the longer side, clipped to the image, resized to at
    most 2,048 px for LaMa and pasted back at native scale [...] The window PNG
    and its offset are stored; the full edited image is composed at load time."

Why a window at all: GroundingME images run to 7,680 px, and big-LaMa on a
7,680 px image is neither fast nor good.  Editing a bounded window around the
hole keeps the inpainter at a resolution it was trained near, and keeps the
stored edit small -- a 2,048 px PNG instead of a second copy of a 60 Mpx image.

The window is a square of the clamped side before clipping; clipping to the
image edge is what makes a stored window non-square.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from idea91.masks import Box, longer_side

WINDOW_FACTOR = 4.0
WINDOW_MIN_SIDE = 768
WINDOW_MAX_SIDE = 2048
LAMA_MAX_SIDE = 2048
LAMA_PAD_MULTIPLE = 8


@dataclass(frozen=True)
class Window:
    """A pixel-aligned crop of the original image, and the offset to paste it back."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def box(self) -> Box:
        return (float(self.x0), float(self.y0), float(self.x1), float(self.y1))

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0 : self.y1, self.x0 : self.x1]

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def window_box(
    hole_box: Box,
    image_wh: tuple[int, int],
    factor: float = WINDOW_FACTOR,
    min_side: int = WINDOW_MIN_SIDE,
    max_side: int = WINDOW_MAX_SIDE,
) -> Window:
    """The P2 window around ``hole_box`` (x0, y0, x1, y1 in original pixels)."""
    w, h = image_wh
    side = float(np.clip(factor * longer_side(hole_box), min_side, max_side))
    cx = (hole_box[0] + hole_box[2]) / 2.0
    cy = (hole_box[1] + hole_box[3]) / 2.0

    def axis(centre: float, extent: float, limit: int) -> tuple[int, int]:
        span = int(min(round(extent), limit))
        lo = int(round(centre - span / 2.0))
        lo = max(0, min(lo, limit - span))  # slide inside the frame, never shrink
        return lo, lo + span

    x0, x1 = axis(cx, side, w)
    y0, y1 = axis(cy, side, h)
    win = Window(x0, y0, x1, y1)
    if not contains(win, hole_box):
        # A hole wider than max_side cannot be covered by a max_side window; the
        # caller must fall back to editing the whole image rather than a window
        # that cuts the hole.  P1/P8 cap referent area (15% and 30%), so this is
        # a guard, not a routine path.
        raise ValueError(
            f"hole {hole_box} does not fit in window {win.as_list()} of an "
            f"{w}x{h} image; raise max_side or edit whole-image"
        )
    return win


def contains(window: Window, box: Box) -> bool:
    return (
        box[0] >= window.x0 - 1e-6
        and box[1] >= window.y0 - 1e-6
        and box[2] <= window.x1 + 1e-6
        and box[3] <= window.y1 + 1e-6
    )


def window_for_mask(mask: np.ndarray, **kwargs) -> Window:
    from idea91.masks import bbox_xyxy

    h, w = mask.shape
    return window_box(bbox_xyxy(mask), (w, h), **kwargs)


@dataclass(frozen=True)
class LamaInput:
    """A window resized and padded for big-LaMa, with what it takes to invert."""

    image: np.ndarray  # HxWx3 uint8, padded
    mask: np.ndarray  # HxW uint8 {0,255}, padded
    native_size: tuple[int, int]  # (w, h) of the window before resizing
    resized_size: tuple[int, int]  # (w, h) after the max-side resize, before padding


def prepare_for_lama(
    window_image: np.ndarray,
    hole_mask: np.ndarray,
    max_side: int = LAMA_MAX_SIDE,
    pad_multiple: int = LAMA_PAD_MULTIPLE,
) -> LamaInput:
    """Resize to at most ``max_side`` and pad to a multiple of 8 (LaMa's stride)."""
    h, w = window_image.shape[:2]
    if hole_mask.shape[:2] != (h, w):
        raise ValueError(f"mask {hole_mask.shape[:2]} does not match window {(h, w)}")
    scale = min(1.0, max_side / float(max(h, w)))
    rw, rh = (int(round(w * scale)), int(round(h * scale))) if scale < 1.0 else (w, h)
    img = (
        cv2.resize(window_image, (rw, rh), interpolation=cv2.INTER_AREA)
        if (rw, rh) != (w, h)
        else window_image.copy()
    )
    m = hole_mask.astype(np.uint8) * 255
    if (rw, rh) != (w, h):
        # nearest keeps the mask binary; a dilated hole loses nothing from it
        m = cv2.resize(m, (rw, rh), interpolation=cv2.INTER_NEAREST)
    ph = (-rh) % pad_multiple
    pw = (-rw) % pad_multiple
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        m = cv2.copyMakeBorder(m, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    return LamaInput(image=img, mask=m, native_size=(w, h), resized_size=(rw, rh))


def restore_from_lama(filled: np.ndarray, prepared: LamaInput) -> np.ndarray:
    """Undo the pad and the resize: back to the window's native pixel size."""
    rw, rh = prepared.resized_size
    out = filled[:rh, :rw]
    w, h = prepared.native_size
    if (rw, rh) != (w, h):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(out)
