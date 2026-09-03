"""Segmentation backends behind one interface (design 4.3, O6).

The instance stack needs three different things from a segmenter, and no single
released model does all three the same way:

* **concept prompts** -- "every instance of {class label}" (K1) or "of {noun
  phrase}" (K2).  SAM 3 has this; SAM 2 does not.
* **box prompts** -- the referent mask from the ground-truth box on the edit
  window (P8, P9).  Both have this.
* **generic / class-agnostic masks** above 0.5% of the image, which fill the
  exclusion set so it covers unlabelled objects (P1, P3).  Design O6 is open on
  whether SAM 3 has a generic mode; SAM 2's automatic mask generator is the
  named fallback.

:class:`Segmenter` declares which of the three a backend supports, so
``idea91/instances/build.py`` can refuse an unsupported path with the design's
own contingency named rather than silently producing a thinner exclusion set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from idea91 import masks as M
from idea91.masks import Box


@dataclass
class RawInstance:
    """One mask a segmenter returned, before the sampler's roles are assigned."""

    mask: np.ndarray
    score: float = 1.0
    prompt: str | None = None  # the concept text, or None for generic/box masks
    origin: str = "concept"  # concept | box | generic
    meta: dict = field(default_factory=dict)

    @property
    def area_frac(self) -> float:
        return M.area_frac(self.mask)

    @property
    def bbox(self) -> Box:
        return M.bbox_xyxy(self.mask)


class SegmenterUnsupported(NotImplementedError):
    """A backend was asked for a capability it does not have."""


class Segmenter(ABC):
    """The instance stack's view of a segmentation model."""

    name: str = "abstract"
    revision: str = "PIN_REQUIRED"
    supports_concept: bool = False
    supports_generic: bool = False
    #: False when this backend is the design's contingency rather than its pin,
    #: so every instance it produces is labelled as such in the store.
    is_pinned_backend: bool = False

    @abstractmethod
    def from_box(self, image: np.ndarray, box: Box) -> np.ndarray:
        """Mask for a box prompt (P8/P9: the referent mask on the edit window)."""

    def concept(self, image: np.ndarray, phrase: str) -> list[RawInstance]:
        raise SegmenterUnsupported(
            f"{self.name} has no concept-prompt mode. K1 can run on ground-truth boxes "
            "instead (design Section 8: 'K1 can start with SAM 2 masks prompted by the "
            "OpenImages ground-truth boxes, labelled as such'); K2 waits for SAM 3."
        )

    def generic(self, image: np.ndarray, min_area_frac: float = 0.005) -> list[RawInstance]:
        raise SegmenterUnsupported(
            f"{self.name} has no generic-object mode (design O6). Use SAM 2's automatic "
            "mask generator for the class-agnostic masks of P1/P3."
        )

    def describe(self) -> dict[str, object]:
        """What the instance store records about the backend that made a mask."""
        return {
            "segmenter": self.name,
            "segmenter_revision": self.revision,
            "pinned_backend": self.is_pinned_backend,
        }


def filter_by_area(
    instances: list[RawInstance], min_frac: float = 0.0, max_frac: float = 1.0
) -> list[RawInstance]:
    """P1's 0.5% to 15% band, and P3's 0.5% floor for class-agnostic masks."""
    return [i for i in instances if min_frac <= i.area_frac <= max_frac]


def drop_empty(instances: list[RawInstance]) -> list[RawInstance]:
    """Discard masks with no pixels.

    A concept prompt or an automatic mask can binarize to nothing at original
    resolution, and such a mask has no centroid, no bounding box and no meaning.
    Filtered here rather than guarded downstream, so no empty instance can reach
    the sampler or the store.
    """
    return [i for i in instances if i.mask.any()]


def deduplicate(instances: list[RawInstance], iou_threshold: float = 0.9) -> list[RawInstance]:
    """Drop near-duplicate masks (several prompts can return the same object).

    Kept greedily by score, so the highest-scoring mask of a duplicate group
    survives with its own prompt attached.
    """
    kept: list[RawInstance] = []
    for cand in sorted(instances, key=lambda i: -i.score):
        if not any(M.mask_iou(cand.mask, k.mask) >= iou_threshold for k in kept):
            kept.append(cand)
    return kept
