"""In-mask compositing and the compositing assert (design P2, acceptance check 1c).

P2: "compositing is in-mask (LaMa output inside the dilated mask, original pixels
elsewhere; asserted by check 1c)".  This is the property the whole K1 gate rests
on: because nothing outside the hole changes, a crop taken at identical
coordinates outside both holes is *byte-identical* between the REMOVE and the
CONTROL_OBJ version of the same image, so a hidden-location classifier reading
such a crop is at AUROC 0.5 by construction -- which is why the design dropped
hidden random crops as the gate and made the gate location-agnostic instead
(deviation 1 in the design header).

Nothing here writes a lossy format: a stored window is a PNG, and the composed
full image is handed to the model as an array or a PNG.  A JPEG anywhere in this
path would break check 1c by re-quantising pixels outside the hole.
"""

from __future__ import annotations

import numpy as np

from idea91.edits.window import Window
from idea91.masks import Box


class CompositingError(AssertionError):
    """Raised when an edit changed pixels outside its hole."""


def composite_in_mask(
    original: np.ndarray, filled: np.ndarray, hole: np.ndarray
) -> np.ndarray:
    """``filled`` inside ``hole``, ``original`` everywhere else.

    All three are window-sized; ``hole`` is the dilated mask (P2).
    """
    if original.shape != filled.shape:
        raise ValueError(f"shape mismatch: original {original.shape}, filled {filled.shape}")
    if hole.shape != original.shape[:2]:
        raise ValueError(f"hole {hole.shape} does not match window {original.shape[:2]}")
    out = original.copy()
    m = hole.astype(bool)
    out[m] = filled[m]
    return out


def assert_in_mask(original: np.ndarray, edited: np.ndarray, hole: np.ndarray) -> None:
    """Check 1c, per edit: no pixel outside the dilated hole changed."""
    if original.shape != edited.shape:
        raise CompositingError(
            f"shape mismatch: original {original.shape}, edited {edited.shape}"
        )
    outside = ~hole.astype(bool)
    if not outside.any():
        return
    differing = np.any(original != edited, axis=-1) if original.ndim == 3 else original != edited
    n = int(np.count_nonzero(differing & outside))
    if n:
        ys, xs = np.nonzero(differing & outside)
        raise CompositingError(
            f"{n} pixels changed outside the hole (first at x={xs[0]}, y={ys[0]}); "
            "compositing must be in-mask (design P2, acceptance check 1c)"
        )


def compose_full(original: np.ndarray, window_image: np.ndarray, window: Window) -> np.ndarray:
    """Rebuild the full edited image from the original plus the stored window.

    P2 stores the window PNG and its offset rather than a second copy of a
    60 Mpx image; this is the "composed at load time" step.
    """
    if window_image.shape[:2] != (window.height, window.width):
        raise ValueError(
            f"window image {window_image.shape[:2]} does not match "
            f"{(window.height, window.width)} from the stored offset"
        )
    out = original.copy()
    out[window.y0 : window.y1, window.x0 : window.x1] = window_image
    return out


def crop(image: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    return image[y0:y1, x0:x1]


def boxes_overlap(a: Box, b: Box) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def assert_paired_crop_identical(
    remove_image: np.ndarray,
    control_image: np.ndarray,
    crop_box: Box,
    remove_hole_box: Box,
    control_hole_box: Box,
) -> np.ndarray:
    """Acceptance check 1c: a crop outside both holes is pixel-identical.

    Returns the (shared) crop so a caller can feed it to the paired-crop null
    row of P5, where the AUROC must come out at 0.5.
    """
    if boxes_overlap(crop_box, remove_hole_box) or boxes_overlap(crop_box, control_hole_box):
        raise ValueError(
            f"crop {crop_box} touches a hole ({remove_hole_box} / {control_hole_box}); "
            "the paired-crop null is defined on crops outside both holes"
        )
    a = crop(remove_image, crop_box)
    b = crop(control_image, crop_box)
    if a.shape != b.shape:
        raise CompositingError(f"paired crops differ in shape: {a.shape} vs {b.shape}")
    if not np.array_equal(a, b):
        n = int(np.count_nonzero(np.any(a != b, axis=-1) if a.ndim == 3 else a != b))
        raise CompositingError(
            f"paired crop at {crop_box} differs in {n} pixels between REMOVE and "
            "CONTROL_OBJ; in-mask compositing was violated (acceptance check 1c)"
        )
    return a


def high_pass_residual_energy(image: np.ndarray, hole: np.ndarray) -> tuple[float, float]:
    """Design 4.4 frequency check: mean high-pass energy inside vs outside the hole.

    Reported once per editor.  A Laplacian is the cheap high-pass; the point is
    the ratio between the two regions, not the absolute value.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    hp = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3) ** 2
    m = hole.astype(bool)
    inside = float(hp[m].mean()) if m.any() else float("nan")
    outside = float(hp[~m].mean()) if (~m).any() else float("nan")
    return inside, outside
