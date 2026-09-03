"""Assemble the instances one image needs (design 4.3, P1, P3, P8, P9).

K1 (P1): prompt every class label of the image, pick one referent whose mask
covers 0.5% to 15% of the image with seed 0, keep every other labelled instance,
and add class-agnostic masks above 0.5% so the exclusion set covers unlabelled
objects too.

K2 (P8, P9): the referent mask from a box prompt on the edit window; the
box-to-mask IoU decides mask hole versus rectangular hole (``box_inpaint``);
every non-referent noun phrase of the expression becomes an instance, and every
*other* instance of the head noun becomes a neighbour -- the geometry P10 routes
on and the count P15's REDUNDANT flag reads.

Instance ``source`` carries two values beyond the three ``control_source``
values of P3: ``same_class_other_instance`` (K1) and ``head_noun_neighbour``
(K2).  Both belong in the exclusion set but are never control candidates, and
neither ever reaches the index's ``control_source`` column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from idea91 import masks as M
from idea91.edits.sampler import Instance
from idea91.instances.backend import RawInstance, Segmenter, deduplicate, filter_by_area
from idea91.instances.nounphrase import NounPhraseParse
from idea91.masks import Box

REFERENT_AREA_BAND_K1 = (0.005, 0.15)  # P1
REFERENT_MAX_AREA_K2 = 0.30  # P8: "referent area at most 30% of the image"
BOX_TO_MASK_IOU_MIN = 0.50  # P8/P9
CLASS_AGNOSTIC_MIN_AREA = 0.005  # P1/P3


@dataclass
class Scene:
    """Everything the sampler and the verifier need for one image."""

    image_id: str
    referent: Instance
    instances: list[Instance]
    backend: dict[str, object]
    head_noun_neighbours: list[Instance] = field(default_factory=list)
    box_to_mask_iou: float | None = None
    box_inpaint_flag: bool = False
    hole_type: str = "mask"
    parse: NounPhraseParse | None = None
    drop_reason: str | None = None

    @property
    def n_head_noun_instances(self) -> int:
        """P15 REDUNDANT: more than one head-noun instance in the image."""
        return 1 + len(self.head_noun_neighbours)

    @property
    def kill_grade(self) -> bool:
        """False when a contingency backend or parser made this scene."""
        if not self.backend.get("pinned_backend", False):
            return False
        return self.parse.kill_grade if self.parse is not None else True


def _to_instances(raws: list[RawInstance], source: str, prefix: str, label_from_prompt: bool = True):
    return [
        Instance(
            instance_id=f"{prefix}{i:03d}",
            mask=r.mask,
            source=source,
            label=(r.prompt if label_from_prompt else None),
        )
        for i, r in enumerate(raws)
    ]


def build_k1_scene(
    image: np.ndarray,
    image_id: str,
    class_labels: list[str],
    segmenter: Segmenter,
    *,
    seed: int = 0,
    area_band: tuple[float, float] = REFERENT_AREA_BAND_K1,
    gt_boxes: dict[str, list[Box]] | None = None,
    with_generic: bool = True,
) -> Scene:
    """P1's scene.  ``gt_boxes`` enables the design's SAM 2 contingency path.

    With a concept-capable segmenter every class label is prompted.  Without one
    (SAM 2), boxes must be supplied and each is box-prompted instead -- the
    design's "SAM 2 masks prompted by the OpenImages ground-truth boxes,
    labelled as such"; ``Scene.kill_grade`` is then False.
    """
    raws: list[RawInstance] = []
    if segmenter.supports_concept:
        for label in class_labels:
            raws += segmenter.concept(image, label)
    else:
        if not gt_boxes:
            return Scene(
                image_id=image_id,
                referent=None,  # type: ignore[arg-type]
                instances=[],
                backend=segmenter.describe(),
                drop_reason="no_concept_mode_and_no_boxes",
            )
        for label, boxes in gt_boxes.items():
            for box in boxes:
                raws.append(
                    RawInstance(
                        mask=segmenter.from_box(image, box),
                        prompt=label,
                        origin="box",
                        meta={"from_gt_box": True},
                    )
                )
    raws = deduplicate(raws)

    candidates = filter_by_area(raws, *area_band)
    if not candidates:
        return Scene(
            image_id=image_id,
            referent=None,  # type: ignore[arg-type]
            instances=[],
            backend=segmenter.describe(),
            drop_reason="no_instance_in_the_p1_area_band",
        )
    rng = np.random.default_rng(seed)
    chosen = candidates[int(rng.integers(0, len(candidates)))]
    referent_label = chosen.prompt

    instances: list[Instance] = []
    referent = Instance("ref", chosen.mask, "referent", referent_label)
    instances.append(referent)
    for i, raw in enumerate(raws):
        if raw is chosen:
            continue
        same_class = raw.prompt == referent_label
        instances.append(
            Instance(
                instance_id=f"lab{i:03d}",
                mask=raw.mask,
                source="same_class_other_instance" if same_class else "labelled_other_class",
                label=raw.prompt,
            )
        )
    if with_generic and segmenter.supports_generic:
        generic = segmenter.generic(image, CLASS_AGNOSTIC_MIN_AREA)
        instances += _to_instances(generic, "class_agnostic", "gen", label_from_prompt=False)

    return Scene(
        image_id=image_id,
        referent=referent,
        instances=instances,
        backend=segmenter.describe(),
    )


def build_k2_scene(
    image: np.ndarray,
    image_id: str,
    parse: NounPhraseParse,
    gt_box: Box,
    segmenter: Segmenter,
    *,
    with_generic: bool = True,
    max_referent_area: float = REFERENT_MAX_AREA_K2,
) -> Scene:
    """P8/P9's scene: the referent from the ground-truth box, plus the phrases."""
    referent_mask = segmenter.from_box(image, gt_box)
    if not referent_mask.any():
        return Scene(
            image_id=image_id,
            referent=None,  # type: ignore[arg-type]
            instances=[],
            backend=segmenter.describe(),
            parse=parse,
            drop_reason="empty_referent_mask",
        )
    iou = M.box_to_mask_iou(gt_box, referent_mask)
    if M.area_frac(referent_mask) > max_referent_area:
        return Scene(
            image_id=image_id,
            referent=None,  # type: ignore[arg-type]
            instances=[],
            backend=segmenter.describe(),
            parse=parse,
            box_to_mask_iou=iou,
            drop_reason="referent_area_above_p8_cap",
        )

    # P9: the SAM mask when box-to-mask IoU is at least 0.5, else the box
    if iou >= BOX_TO_MASK_IOU_MIN:
        hole_type, box_inpaint = "mask", False
        region = referent_mask
    else:
        hole_type, box_inpaint = "rect", True
        region = M.mask_from_box(gt_box, referent_mask.shape)

    referent = Instance("ref", region, "referent", parse.head_noun)
    instances = [referent]
    neighbours: list[Instance] = []

    if segmenter.supports_concept:
        head_raws = deduplicate(segmenter.concept(image, parse.head_noun))
        for i, raw in enumerate(head_raws):
            if M.mask_iou(raw.mask, region) >= 0.5:
                continue  # this is the referent itself
            inst = Instance(f"head{i:03d}", raw.mask, "head_noun_neighbour", parse.head_noun)
            neighbours.append(inst)
            instances.append(inst)
        for j, phrase in enumerate(parse.other_phrases()):
            for i, raw in enumerate(deduplicate(segmenter.concept(image, phrase))):
                instances.append(
                    Instance(f"np{j:02d}_{i:03d}", raw.mask, "noun_phrase_instance", phrase)
                )

    if with_generic and segmenter.supports_generic:
        instances += _to_instances(
            segmenter.generic(image, CLASS_AGNOSTIC_MIN_AREA),
            "class_agnostic",
            "gen",
            label_from_prompt=False,
        )

    return Scene(
        image_id=image_id,
        referent=referent,
        instances=instances,
        head_noun_neighbours=neighbours,
        backend=segmenter.describe(),
        box_to_mask_iou=iou,
        box_inpaint_flag=box_inpaint,
        hole_type=hole_type,
        parse=parse,
    )


def scene_rows(scene: Scene, set_or_pool: str) -> list[dict]:
    """Rows for ``instances.parquet`` (see notes/DEVIATIONS.md D-12)."""
    if scene.referent is None:
        return []
    rows = []
    for inst in scene.instances:
        centre = M.centroid(inst.mask)
        rows.append(
            {
                "image_id": scene.image_id,
                "set_or_pool": set_or_pool,
                "instance_id": inst.instance_id,
                "source": inst.source,
                "label": inst.label,
                "mask_rle": M.encode_rle(inst.mask),
                "area_frac": M.area_frac(inst.mask),
                "bbox_xyxy_px": list(M.bbox_xyxy(inst.mask)),
                "centre_xy_px": [centre[0], centre[1]],
                "box_to_mask_iou": scene.box_to_mask_iou if inst.source == "referent" else None,
                "box_inpaint_flag": scene.box_inpaint_flag if inst.source == "referent" else False,
                "segmenter": str(scene.backend.get("segmenter", "")),
                "segmenter_revision": str(scene.backend.get("segmenter_revision", "")),
                "kill_grade": scene.kill_grade,
                "head_noun": scene.parse.head_noun if scene.parse else None,
                "n_head_noun_instances": scene.n_head_noun_instances,
            }
        )
    return rows
