"""SAM 3 and SAM 2 adapters (design 4.3, Section 8, O6).

SAM 3 is the pin: concept prompts for the class labels (K1) and the noun phrases
(K2), box prompts for the referent, and -- if it has a generic mode, which design
O6 leaves open until pin time -- the class-agnostic masks.

SAM 2 is the design's own contingency, and it is a *partial* one: it can prompt
by box and it has an automatic mask generator, but it has no concept mode.  So
SAM 2 can carry K1 on the OpenImages ground-truth boxes ("labelled as such") and
can fill the class-agnostic masks under O6, while K2's noun-phrase instances
wait for SAM 3.  ``is_pinned_backend`` is False on the SAM 2 adapter and travels
into every instance row, so a bank built on the contingency can never be read as
a SAM 3 bank.

Neither class imports its package at module import time: the checkpoints are not
on this machine yet, and the rest of the stack must stay importable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from idea91.masks import Box
from idea91.instances.backend import RawInstance, Segmenter, SegmenterUnsupported


class Sam3Segmenter(Segmenter):
    """The pinned backend.  Requires the gated SAM 3 checkpoint (design O7)."""

    name = "sam3"
    supports_concept = True
    is_pinned_backend = True

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        config: str | None = None,
        device: str = "cuda",
        revision: str = "PIN_REQUIRED",
        has_generic_mode: bool | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint or os.environ.get("VLMG_SAM3_CKPT", "")).expanduser()
        self.config = config
        self.device = device
        self.revision = revision
        # design O6, open until pin time; set explicitly once the release is read
        self.supports_generic = bool(has_generic_mode)
        self._model = None

    def _load(self):
        if self._model is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(
                    "SAM 3 checkpoint not found. It is gated (design O7): request access, "
                    "record the repository commit and the file sha256 in shared/env/PINS.md, "
                    "and set VLMG_SAM3_CKPT. Until then use Sam2Segmenter, which the design "
                    "names as the contingency and which labels its output as such."
                )
            try:
                from sam3.build_sam import build_sam3  # type: ignore
                from sam3.sam3_image_predictor import SAM3ImagePredictor  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on the release
                raise ImportError(
                    "the SAM 3 package is not installed; pin the repository commit in "
                    "shared/env/PINS.md and install it into ~/vlmg-env"
                ) from exc
            self._model = SAM3ImagePredictor(build_sam3(self.config, str(self.checkpoint)))
        return self._model

    def from_box(self, image: np.ndarray, box: Box) -> np.ndarray:  # pragma: no cover
        model = self._load()
        model.set_image(image)
        masks, scores, _ = model.predict(box=np.array(box)[None, :], multimask_output=False)
        return np.asarray(masks[0]).astype(bool)

    def concept(self, image: np.ndarray, phrase: str) -> list[RawInstance]:  # pragma: no cover
        model = self._load()
        model.set_image(image)
        masks, scores = model.predict_concept(phrase)
        return [
            RawInstance(mask=np.asarray(m).astype(bool), score=float(s), prompt=phrase,
                        origin="concept")
            for m, s in zip(masks, scores)
        ]

    def generic(self, image, min_area_frac: float = 0.005) -> list[RawInstance]:  # pragma: no cover
        if not self.supports_generic:
            raise SegmenterUnsupported(
                "SAM 3's generic-object mode is design O6, still open. Use "
                "Sam2Segmenter.generic (its automatic mask generator) for the "
                "class-agnostic masks of P1/P3."
            )
        model = self._load()
        model.set_image(image)
        masks = model.generate()
        out = [RawInstance(mask=np.asarray(m).astype(bool), origin="generic") for m in masks]
        return [i for i in out if i.area_frac > min_area_frac]


class Sam2Segmenter(Segmenter):
    """The contingency: box prompts and automatic masks, no concept mode.

    ``sam2.1_hiera_large.pt`` is already on this host, so the O6 class-agnostic
    pass and a labelled-as-such K1 run can proceed without SAM 3.
    """

    name = "sam2.1_hiera_large"
    supports_concept = False
    supports_generic = True
    is_pinned_backend = False

    DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        config: str = DEFAULT_CONFIG,
        device: str = "cuda",
        revision: str = "PIN_REQUIRED",
        points_per_side: int = 32,
        min_area_frac: float = 0.005,
    ) -> None:
        self.checkpoint = Path(
            checkpoint or os.environ.get("VLMG_SAM2_CKPT", Path.home() / "sam2_ckpts" / "sam2.1_hiera_large.pt")
        ).expanduser()
        self.config = config
        self.device = device
        self.revision = revision
        self.points_per_side = points_per_side
        self.min_area_frac = min_area_frac
        self._predictor = None
        self._generator = None

    def _build(self):  # pragma: no cover - needs the checkpoint
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM 2 checkpoint not found at {self.checkpoint}; set VLMG_SAM2_CKPT"
            )
        try:
            from sam2.build_sam import build_sam2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "the sam2 package is not installed in ~/vlmg-env; pin its commit in "
                "shared/env/PINS.md and install it"
            ) from exc
        return build_sam2(self.config, str(self.checkpoint), device=self.device)

    def from_box(self, image: np.ndarray, box: Box) -> np.ndarray:  # pragma: no cover
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        if self._predictor is None:
            self._predictor = SAM2ImagePredictor(self._build())
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            box=np.array(box)[None, :], multimask_output=False
        )
        return np.asarray(masks[0]).astype(bool)

    def generic(self, image: np.ndarray, min_area_frac: float | None = None) -> list[RawInstance]:  # pragma: no cover
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore

        if self._generator is None:
            self._generator = SAM2AutomaticMaskGenerator(
                self._build(), points_per_side=self.points_per_side
            )
        floor = self.min_area_frac if min_area_frac is None else min_area_frac
        out = [
            RawInstance(
                mask=np.asarray(r["segmentation"]).astype(bool),
                score=float(r.get("predicted_iou", 1.0)),
                origin="generic",
            )
            for r in self._generator.generate(image)
        ]
        return [i for i in out if i.area_frac > floor]


def get_segmenter(name: str = "sam3", **kwargs) -> Segmenter:
    if name in ("sam3", "sam-3"):
        return Sam3Segmenter(**kwargs)
    if name in ("sam2", "sam-2", "sam2.1"):
        return Sam2Segmenter(**kwargs)
    raise ValueError(f"unknown segmenter {name!r}")
