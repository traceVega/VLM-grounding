"""Class-agnostic masks by point-grid prompting (design P1, P3, O6).

P1 adds "class-agnostic masks above 0.5% area ... so the exclusion set covers
unlabelled objects", and P3 puts them in the exclusion set so a control hole
never clips something Open Images did not label.  Design O6 leaves open whether
SAM 3 provides them; it does not -- ``Sam3Processor`` takes text or box prompts
only -- so O6 resolves to its named fallback, SAM 2's automatic generator.

This is that generator, written against ``transformers``' ``Sam2Model`` rather
than the ``sam2`` PyPI package, whose 1.1.0 sdist declares no dependencies at all
and does not match the upstream project.  The algorithm is SAM's own: a regular
grid of point prompts, three candidate masks per point, then filtering by the
model's predicted IoU, by a stability score, and by area, then de-duplication.

The ordering matters for memory.  A 32x32 grid gives 3,072 candidate masks, and
holding those at a 1,024 px original size would be ~9 GB; so every filter runs on
the 256 px decoder output and only the survivors are upscaled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from idea91.instances.backend import RawInstance

SAM2_HF_PATH = "facebook/sam2.1-hiera-large"
SAM2_REVISION = "665f8e2ad61c"

POINTS_PER_SIDE = 32
POINT_BATCH = 64
PRED_IOU_THRESH = 0.80
STABILITY_SCORE_THRESH = 0.90
STABILITY_OFFSET = 1.0
NMS_IOU_THRESH = 0.70
MIN_AREA_FRAC = 0.005  # P1/P3

#: Upper bound on a class-agnostic mask, as a fraction of the image.
#: P1 sets a floor ("above 0.5% area") and no ceiling, but SAM readily returns
#: the background itself as a high-confidence "object" -- measured at 81% of a
#: test scene.  Putting that in the P3 exclusion set would leave nowhere for a
#: CONTROL_BG hole to go and would drop most CONTROL_OBJ candidates, so masks
#: covering more than half the image are treated as scene, not object.
#: Recorded in notes/DEVIATIONS.md D-23 and OPEN-QUESTIONS Q-11.
MAX_AREA_FRAC = 0.50


def point_grid(points_per_side: int, width: int, height: int) -> np.ndarray:
    """A regular grid of prompt points in original-image pixels."""
    step = 1.0 / points_per_side
    offsets = (np.arange(points_per_side) + 0.5) * step
    xs = offsets * width
    ys = offsets * height
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)
    return grid.reshape(-1, 2)


def stability_score(logits, threshold: float = 0.0, offset: float = STABILITY_OFFSET):
    """SAM's stability score: agreement between two mask thresholds.

    A mask whose area barely changes when the cut moves is a stable object; one
    that balloons is usually a boundary artefact.
    """
    import torch

    high = (logits > (threshold + offset)).flatten(-2).sum(-1)
    low = (logits > (threshold - offset)).flatten(-2).sum(-1)
    return torch.where(low > 0, high / low.clamp(min=1), torch.zeros_like(high, dtype=torch.float32))


def mask_boxes(masks: np.ndarray) -> np.ndarray:
    """xyxy boxes for a stack of binary masks; empty masks get a degenerate box."""
    out = np.zeros((len(masks), 4), dtype=np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if xs.size:
            out[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy box NMS; returns kept indices, highest score first."""
    order = np.argsort(-scores)
    keep: list[int] = []
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
        iou = inter / np.maximum(1e-6, areas[i] + areas[rest] - inter)
        order = rest[iou <= threshold]
    return keep


def mask_nms(masks: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy NMS on *mask* IoU rather than box IoU.

    Box NMS is the detection convention, but it fails here: a candidate mask
    often carries a few stray pixels far from the object, and those pixels blow
    up its bounding box without changing the mask.  Measured on a scene with
    three objects, two candidates had mask IoU 1.000 and box IoU 0.348, so box
    NMS kept both.  Masks are compared as flat bitsets, which is why this runs on
    the 256 px decoder output and not on full-resolution masks.
    """
    flat = masks.reshape(len(masks), -1)
    areas = flat.sum(1).astype(np.int64)
    order = list(np.argsort(-scores))
    keep: list[int] = []
    while order:
        i = int(order.pop(0))
        keep.append(i)
        if not order:
            break
        rest = np.array(order, dtype=int)
        inter = (flat[rest] & flat[i]).sum(1).astype(np.int64)
        union = areas[rest] + areas[i] - inter
        iou = inter / np.maximum(1, union)
        order = [int(r) for r, keep_it in zip(rest, iou <= threshold) if keep_it]
    return keep


@dataclass
class AutoMaskSettings:
    points_per_side: int = POINTS_PER_SIDE
    point_batch: int = POINT_BATCH
    pred_iou_thresh: float = PRED_IOU_THRESH
    stability_score_thresh: float = STABILITY_SCORE_THRESH
    stability_offset: float = STABILITY_OFFSET
    nms_iou_thresh: float = NMS_IOU_THRESH
    min_area_frac: float = MIN_AREA_FRAC
    max_area_frac: float = MAX_AREA_FRAC


class Sam2AutomaticMasks:
    """SAM 2 automatic mask generation over a point grid."""

    def __init__(
        self,
        hf_path: str = SAM2_HF_PATH,
        revision: str = SAM2_REVISION,
        device: str = "cuda",
        settings: AutoMaskSettings | None = None,
        processor=None,
        model=None,
    ) -> None:
        self.hf_path = hf_path
        self.revision = revision
        self.device = device
        self.settings = settings or AutoMaskSettings()
        self._processor = processor
        self._model = model

    def _load(self):
        if self._processor is None or self._model is None:
            import torch
            from transformers import Sam2Model, Sam2Processor

            self._processor = self._processor or Sam2Processor.from_pretrained(
                self.hf_path, revision=self.revision
            )
            self._model = self._model or (
                Sam2Model.from_pretrained(self.hf_path, revision=self.revision, dtype=torch.float32)
                .to(self.device)
                .eval()
            )
        return self._processor, self._model

    def generate(self, image: np.ndarray, min_area_frac: float | None = None) -> list[RawInstance]:
        import torch

        s = self.settings
        floor = s.min_area_frac if min_area_frac is None else min_area_frac
        processor, model = self._load()
        height, width = image.shape[:2]

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            embeddings = model.get_image_embeddings(pixel_values)

        grid = point_grid(s.points_per_side, width, height)
        kept_logits: list[torch.Tensor] = []
        kept_scores: list[torch.Tensor] = []

        for start in range(0, len(grid), s.point_batch):
            chunk = grid[start : start + s.point_batch]
            # [image][object][point][xy]: one object per point, one point each
            points = [[[[float(x), float(y)]] for x, y in chunk]]
            labels = [[[1] for _ in chunk]]
            prompt = processor(
                images=image,
                input_points=points,
                input_labels=labels,
                return_tensors="pt",
            )
            with torch.inference_mode():
                out = model(
                    image_embeddings=embeddings,
                    input_points=prompt["input_points"].to(self.device),
                    input_labels=prompt["input_labels"].to(self.device),
                    multimask_output=True,
                )
            # (1, n_points, 3, h, w) logits and (1, n_points, 3) scores
            logits = out.pred_masks.flatten(0, 2) if out.pred_masks.dim() == 5 else out.pred_masks
            scores = out.iou_scores.flatten()
            logits = logits.reshape(-1, *logits.shape[-2:])

            good = scores >= s.pred_iou_thresh
            if good.any():
                stability = stability_score(logits[good], 0.0, s.stability_offset)
                stable = stability >= s.stability_score_thresh
                if stable.any():
                    kept_logits.append(logits[good][stable].float().cpu())
                    kept_scores.append(scores[good][stable].float().cpu())

        if not kept_logits:
            return []

        logits = torch.cat(kept_logits)
        scores = torch.cat(kept_scores).numpy()

        # De-duplicate on the 256 px decoder output, before any upscaling: the
        # comparison is cheap there, and only the survivors need full resolution.
        low = (logits > 0).numpy()
        low_area = low.reshape(len(low), -1).mean(1)
        plausible = (low_area > floor * 0.5) & (low_area < s.max_area_frac)
        if not plausible.any():
            return []
        index = np.flatnonzero(plausible)
        keep = mask_nms(low[index], scores[index], s.nms_iou_thresh)
        index = index[keep]

        with torch.inference_mode():
            masks = processor.post_process_masks(
                [logits[index][:, None]], original_sizes=[(height, width)], binarize=True
            )[0]
        masks = masks.squeeze(1).cpu().numpy().astype(bool)
        scores = scores[index]

        areas = masks.reshape(len(masks), -1).sum(1) / float(height * width)
        good = (areas > floor) & (areas < s.max_area_frac)
        return [
            RawInstance(mask=masks[i], score=float(scores[i]), origin="generic")
            for i in np.flatnonzero(good)
        ]
