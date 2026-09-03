"""Relations over one pair: P14 same-box, the P15 flags, REDUNDANT and UNRESOLVED.

Design 4.7: "functions over ``outputs.parquet`` joined on ``pair_id``,
``condition`` and ``abstain_protocol`` that compute P14 and P15 per item, the
clean and validity flags from ``verifier.parquet`` under the P10 rules and from
``human_labels.csv``, and the REDUNDANT and UNRESOLVED flags.  Sufficiency,
necessity-plus and the score are not implemented here."

Two rules that are easy to get subtly wrong, so they are spelled out:

* A box that moves under the *control* edit makes the item UNRESOLVED (P15), and
  an UNRESOLVED item is excluded from same-box.  Same-box then means "the model
  re-predicted the same box although the referent is gone", not "the model is
  unstable".
* An item the verifier could not determine (P10's same-noun neighbour case) is
  excluded from every verifier-based clean statistic *and* from kappa, but stays
  in the human-clean column if it carries human labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from idea91 import masks as M
from idea91.masks import Box

#: A localization output, whatever its shape.  Molmo2 points; Qwen3-VL boxes.
LOCALIZATION_TYPES = ("box", "point", "set")
SAME_BOX_IOU = 0.5  # P14
SAME_REGION_IOU = 0.3  # P14's "IoU(box on REMOVE, R) at least 0.3"
ORIGINAL_CORRECT_IOU = 0.5  # P14 / SPEC is_failure
SHIFT_IOU = 0.5  # P15: a box "shifted" when IoU with the ORIGINAL box is below 0.5


@dataclass
class Prediction:
    """One model output on one condition, already parsed into original pixels."""

    output_type: str
    box: Box | None = None
    point: tuple[float, float] | None = None
    parse_ok: bool = True

    @property
    def is_none(self) -> bool:
        return self.output_type == "none"

    @property
    def localized(self) -> bool:
        return self.output_type in LOCALIZATION_TYPES and (
            self.box is not None or self.point is not None
        )


@dataclass
class PairInput:
    """Everything the relations need for one (pair, model, protocol)."""

    pair_id: str
    model_id: str
    abstain_protocol: str
    predictions: dict[str, Prediction]  # condition -> Prediction
    gt_box: Box | None = None
    removal_region: np.ndarray | None = None  # R (P9), a mask over the original image
    removal_region_box: Box | None = None  # fallback when R is a rectangle
    prior_box: Box | None = None  # P13's centre-and-median-size prior
    n_head_noun_instances: int = 1  # P15 REDUNDANT
    verifier_clean: bool | None = None  # P10, None = undetermined
    human_clean: bool | None = None  # P11 majority label
    control_obj_valid: bool | None = None  # P10 validity rule
    control_bg_valid: bool | None = None
    item_id: str | None = None
    set_name: str | None = None
    size_bin: str | None = None
    image_id: str | None = None
    extra: dict = field(default_factory=dict)

    def get(self, condition: str) -> Prediction | None:
        return self.predictions.get(condition)


# --- geometry helpers --------------------------------------------------------


def iou_boxes(a: Box | None, b: Box | None) -> float | None:
    if a is None or b is None:
        return None
    return M.box_iou(a, b)


def iou_box_region(box: Box | None, region: np.ndarray | None, region_box: Box | None) -> float | None:
    """IoU of a predicted box with the removal region R.

    R is a mask when the removal used a SAM 3 mask hole and a rectangle when P8's
    ``box_inpaint`` flag is set, so both forms are accepted.
    """
    if box is None:
        return None
    if region is not None:
        box_mask = M.mask_from_box(box, region.shape)
        inter = int(np.count_nonzero(box_mask & region))
        union = int(np.count_nonzero(box_mask | region))
        return float(inter / union) if union else 0.0
    return M.box_iou(box, region_box) if region_box is not None else None


def point_in_region(
    point: tuple[float, float] | None, region: np.ndarray | None, region_box: Box | None
) -> bool | None:
    """P14 for a pointing model: does the REMOVE point lie inside R?"""
    if point is None:
        return None
    x, y = point
    if region is not None:
        h, w = region.shape
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            return False
        return bool(region[yi, xi])
    if region_box is not None:
        return bool(region_box[0] <= x <= region_box[2] and region_box[1] <= y <= region_box[3])
    return None


def moved_from(pred: Prediction | None, reference: Box | None, threshold: float = SHIFT_IOU) -> bool | None:
    """True when the prediction is no longer on ``reference`` (a shift, or a none)."""
    if pred is None:
        return None
    if pred.is_none:
        return True
    if pred.box is None or reference is None:
        return None
    return M.box_iou(pred.box, reference) < threshold


# --- the relations -----------------------------------------------------------


def relate(inp: PairInput) -> dict:
    """One row of ``relations.parquet`` (design Section 5)."""
    orig = inp.get("ORIGINAL")
    remove = inp.get("REMOVE")
    ctl_obj = inp.get("CONTROL_OBJ")
    ctl_bg = inp.get("CONTROL_BG")

    orig_box = orig.box if orig else None
    orig_correct = None
    if orig is not None:
        if orig.localized and orig.box is not None and inp.gt_box is not None:
            orig_correct = M.box_iou(orig.box, inp.gt_box) >= ORIGINAL_CORRECT_IOU
        elif orig.localized and orig.point is not None and inp.gt_box is not None:
            x, y = orig.point
            orig_correct = bool(
                inp.gt_box[0] <= x <= inp.gt_box[2] and inp.gt_box[1] <= y <= inp.gt_box[3]
            )
        else:
            orig_correct = False

    # --- P15 UNRESOLVED: instability under the control edit ------------------
    unresolved = None
    if orig is not None and ctl_obj is not None:
        if orig.output_type != ctl_obj.output_type:
            unresolved = True
        elif orig.box is not None and ctl_obj.box is not None:
            unresolved = M.box_iou(ctl_obj.box, orig.box) < SHIFT_IOU
        elif orig.point is not None and ctl_obj.point is not None:
            # a pointing model has no box to compare; treat a point that left the
            # ORIGINAL point's neighbourhood as a change of output, handled by the
            # output_type check above, so nothing more is decidable here
            unresolved = False
        else:
            unresolved = True

    # --- P15 rates on REMOVE -------------------------------------------------
    none_on_remove = remove.is_none if remove is not None else None
    box_on_remove = remove.localized if remove is not None else None

    same_box_50 = None
    same_box_r30 = None
    if remove is not None and orig_correct:
        if remove.box is not None and orig_box is not None:
            same_box_50 = M.box_iou(remove.box, orig_box) >= SAME_BOX_IOU
            r_iou = iou_box_region(remove.box, inp.removal_region, inp.removal_region_box)
            same_box_r30 = None if r_iou is None else r_iou >= SAME_REGION_IOU
        elif remove.point is not None:
            # P14 for Molmo2: the REMOVE point lies inside R
            same_box_r30 = point_in_region(
                remove.point, inp.removal_region, inp.removal_region_box
            )
            if orig is not None and orig.point is not None and inp.removal_region_box is not None:
                same_box_50 = same_box_r30
        elif remove.is_none:
            same_box_50 = False
            same_box_r30 = False

    # --- P15 control-edit behaviour, on ORIGINAL-correct items ---------------
    shift_obj = shift_bg = None
    if orig_correct:
        shift_obj = moved_from(ctl_obj, orig_box)
        shift_bg = moved_from(ctl_bg, orig_box)
    abstain_obj = ctl_obj.is_none if ctl_obj is not None else None

    # --- P13 text-side controls ---------------------------------------------
    t_null = inp.get("T_NULL")
    t_head = inp.get("T_HEAD")
    t_attr = inp.get("T_ATTR")
    t_null_pass = moved_from(t_null, orig_box)
    t_head_pass = moved_from(t_head, orig_box)
    t_attr_moved = moved_from(t_attr, orig_box)
    t_null_prior_pass = None
    if t_null is not None and inp.prior_box is not None:
        t_null_prior_pass = (
            True if t_null.is_none else moved_from(t_null, inp.prior_box)
        )

    return {
        "pair_id": inp.pair_id,
        "model_id": inp.model_id,
        "abstain_protocol": inp.abstain_protocol,
        "orig_correct": orig_correct,
        "remove_clean_verifier": inp.verifier_clean,
        "remove_clean_human": inp.human_clean,
        "control_obj_valid": inp.control_obj_valid,
        "control_bg_valid": inp.control_bg_valid,
        "none_on_remove": none_on_remove,
        "box_on_remove": box_on_remove,
        "same_box_50": same_box_50,
        "same_box_r30": same_box_r30,
        "shift_on_control_obj": shift_obj,
        "shift_on_control_bg": shift_bg,
        "abstain_on_control_obj": abstain_obj,
        "t_null_pass": t_null_pass,
        "t_null_prior_pass": t_null_prior_pass,
        "t_head_pass": t_head_pass,
        "t_attr_moved": t_attr_moved,
        "redundant": inp.n_head_noun_instances > 1,
        "unresolved": unresolved,
        "item_id": inp.item_id,
        "set": inp.set_name,
        "size_bin": inp.size_bin,
        "image_id": inp.image_id,
        "undetermined": inp.verifier_clean is None,
    }


def prior_box(image_wh: tuple[int, int], median_box_frac: float) -> Box:
    """P13's baseline: a centred box of the set's median size.

    ``median_box_frac`` is the median ratio of box longer side to image longer
    side over the set, computed once and passed in, so the baseline is a property
    of the set rather than of the item.
    """
    w, h = image_wh
    side = median_box_frac * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
