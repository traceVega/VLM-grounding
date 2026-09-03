"""Materialize an :class:`~idea91.edits.sampler.EditPlan` into a stored window
plus an ``edits/index.parquet`` row (design 4.3, P2, Section 5).

    edits/<pool>/<image_id>/<instance_id>/<operator>.png

The PNG is the *edited window at native window scale*, already composited
in-mask, so pasting it whole back into the original changes only hole pixels --
which is what makes acceptance check 1c hold for the composed full images.
Nothing here writes a lossy format.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from idea91 import masks as M
from idea91.edits import composite as C
from idea91.edits.inpaint import Inpainter
from idea91.edits.sampler import EditPlan
from idea91.edits.window import Window, prepare_for_lama, restore_from_lama, window_box


@dataclass(frozen=True)
class BuiltEdit:
    """What :func:`materialize` produced: the window on disk and its index row."""

    window_image: np.ndarray
    window: Window
    index_row: dict

    @property
    def window_path(self) -> str:
        return self.index_row["window_path"]


def read_image(path: str | Path) -> np.ndarray:
    """RGB uint8.  ``cv2.imread`` handles the 7,680 px GroundingME files."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_png(path: Path, image: np.ndarray) -> str:
    """Write lossless PNG, return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise OSError(f"failed to write {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(
    image: np.ndarray,
    plan: EditPlan,
    inpainter: Inpainter,
    *,
    image_id: str,
    instance_id: str,
    set_or_pool: str,
    edits_root: Path,
    pair_id: str | None = None,
    box_to_mask_iou: float | None = None,
    box_inpaint_flag: bool = False,
    freeze_version: str = "unfrozen",
) -> BuiltEdit:
    """Run one edit: window, inpaint, in-mask composite, assert, store, index."""
    h, w = image.shape[:2]
    hole_box = M.bbox_xyxy(plan.hole)
    window = window_box(hole_box, (w, h))

    win_img = np.ascontiguousarray(window.crop(image))
    win_hole = np.ascontiguousarray(window.crop(plan.hole))

    prepared = prepare_for_lama(win_img, win_hole)
    filled = inpainter.fill(prepared.image, prepared.mask)
    filled_native = restore_from_lama(filled, prepared)
    edited_win = C.composite_in_mask(win_img, filled_native, win_hole)
    C.assert_in_mask(win_img, edited_win, win_hole)  # P2 / acceptance check 1c

    rel = Path(set_or_pool) / image_id / instance_id / f"{plan.operator}.png"
    sha = write_png(edits_root / rel, edited_win)

    row = {
        "image_id": image_id,
        "set_or_pool": set_or_pool,
        "instance_id": instance_id,
        "pair_id": pair_id,
        "operator": plan.operator,
        "editor": inpainter.name,
        "editor_weights_sha256": inpainter.weights_sha256,
        "editor_settings_hash": inpainter.settings_hash,
        "mask_rle": M.encode_rle(plan.mask),
        "hole_type": plan.hole_type,
        "mask_area_frac": M.area_frac(plan.mask),
        "box_to_mask_iou": box_to_mask_iou,
        "box_inpaint_flag": box_inpaint_flag,
        "control_instance_id": plan.control_instance_id,
        "control_source": plan.control_source,
        "control_area_ratio": plan.control_area_ratio,
        "control_centrality_delta": plan.control_centrality_delta,
        "window_xyxy_px": window.as_list(),
        "window_path": rel.as_posix(),
        "window_sha256": sha,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "freeze_version": freeze_version,
    }
    return BuiltEdit(window_image=edited_win, window=window, index_row=row)


def load_edited(original: np.ndarray, index_row: dict, edits_root: Path) -> np.ndarray:
    """P2's "composed at load time": original + stored window -> full edited image."""
    x0, y0, x1, y1 = index_row["window_xyxy_px"]
    window = Window(int(x0), int(y0), int(x1), int(y1))
    win_img = read_image(edits_root / index_row["window_path"])
    return C.compose_full(original, win_img, window)


def hole_from_row(index_row: dict) -> np.ndarray:
    """The dilated hole of a stored edit, for the gate's crop rows and the asserts."""
    from idea91.edits.sampler import hole_for

    return hole_for(M.decode_rle(index_row["mask_rle"]), index_row["hole_type"])
