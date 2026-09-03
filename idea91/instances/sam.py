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

SAM 3 is loaded through ``transformers`` (``Sam3Model`` / ``Sam3Processor``,
native since transformers 5), not through a separate ``sam3`` package: the
processor takes ``text=`` for a concept prompt and ``input_boxes=`` for a box
prompt, which is exactly the two capabilities the instance stack needs.

Neither class imports its package at module import time: the weights are gated
and may not be on the machine, and the rest of the stack must stay importable.
Both accept injected ``processor``/``model`` objects so the tensor plumbing --
thresholds, target sizes, the numpy conversion -- is unit-testable without
weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from idea91.masks import Box
from idea91.instances.backend import RawInstance, Segmenter, SegmenterUnsupported


#: The transformers-loadable SAM 3 release.  ``facebook/sam3.1`` exists and is
#: newer, but ships only ``sam3.1_multiplex.pt`` -- no ``model.safetensors`` --
#: so ``Sam3Model.from_pretrained`` cannot read it.  See notes/DEVIATIONS.md D-16.
SAM3_HF_PATH = "facebook/sam3"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"

SAM3_SCORE_THRESHOLD = 0.3  # transformers' post-process default
SAM3_MASK_THRESHOLD = 0.5


class Sam3Segmenter(Segmenter):
    """The pinned backend: ``Sam3Model`` / ``Sam3Processor`` from transformers.

    The repository is gated (``gated=manual``), so the weights need an accepted
    licence and a token on the machine; design O7 also wants the licence text
    read before anything derived from it is released.
    """

    name = "sam3"
    supports_concept = True
    is_pinned_backend = True

    def __init__(
        self,
        hf_path: str = SAM3_HF_PATH,
        revision: str = SAM3_REVISION,
        device: str = "cuda",
        *,
        score_threshold: float = SAM3_SCORE_THRESHOLD,
        mask_threshold: float = SAM3_MASK_THRESHOLD,
        has_generic_mode: bool = False,  # design O6, open
        processor=None,
        model=None,
    ) -> None:
        self.hf_path = hf_path
        self.revision = revision
        self.device = device
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.supports_generic = bool(has_generic_mode)
        self._processor = processor
        self._model = model

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "hf_path": self.hf_path}

    def _load(self):
        if self._processor is None or self._model is None:
            try:
                import torch
                from transformers import Sam3Model, Sam3Processor
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "SAM 3 needs transformers >= 5 (Sam3Model / Sam3Processor) in ~/vlmg-env"
                ) from exc
            try:
                self._processor = self._processor or Sam3Processor.from_pretrained(
                    self.hf_path, revision=self.revision
                )
                self._model = self._model or Sam3Model.from_pretrained(
                    self.hf_path, revision=self.revision, dtype=torch.bfloat16
                ).to(self.device).eval()
            except Exception as exc:  # gated repo, no token, or no network
                raise RuntimeError(
                    f"cannot load {self.hf_path}@{self.revision[:12]}: {exc}. The repository is "
                    "gated: accept the licence on Hugging Face and put a token on this machine "
                    "(`hf auth login`, or HF_TOKEN). Until then use Sam2Segmenter, which the "
                    "design names as the contingency and which labels its output as such."
                ) from exc
        return self._processor, self._model

    def _predict(self, image: np.ndarray, **prompt) -> list[dict]:
        """One forward pass, post-processed to original-resolution instances."""
        import torch

        processor, model = self._load()
        height, width = image.shape[:2]
        inputs = processor(images=image, return_tensors="pt", **prompt)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        return processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(height, width)],
        )

    @staticmethod
    def _to_numpy(mask) -> np.ndarray:
        arr = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
        return arr.astype(bool)

    def concept(self, image: np.ndarray, phrase: str) -> list[RawInstance]:
        """Every instance of ``phrase``: the class labels (K1), noun phrases (K2)."""
        result = self._predict(image, text=phrase)[0]
        return [
            RawInstance(
                mask=self._to_numpy(mask),
                score=float(score),
                prompt=phrase,
                origin="concept",
            )
            for mask, score in zip(result["masks"], result["scores"])
        ]

    def from_box(self, image: np.ndarray, box: Box) -> np.ndarray:
        """P8/P9: the referent mask from the ground-truth box.

        The highest-scoring instance is taken; an empty result returns an empty
        mask, which ``build_k2_scene`` drops with ``empty_referent_mask`` rather
        than editing a hole that is not there.
        """
        result = self._predict(
            image,
            input_boxes=[[[float(v) for v in box]]],
            input_boxes_labels=[[[1]]],
        )[0]
        masks, scores = result["masks"], result["scores"]
        if len(masks) == 0:
            return np.zeros(image.shape[:2], dtype=bool)
        best = int(np.argmax([float(s) for s in scores]))
        return self._to_numpy(masks[best])

    def generic(self, image, min_area_frac: float = 0.005) -> list[RawInstance]:
        if not self.supports_generic:
            raise SegmenterUnsupported(
                "SAM 3's generic-object mode is design O6, still open. Use "
                "Sam2Segmenter.generic (its automatic mask generator) for the "
                "class-agnostic masks of P1/P3."
            )
        raise SegmenterUnsupported(
            "has_generic_mode=True was set, but transformers' Sam3Processor exposes no "
            "generate-everything call: it takes text or box prompts only. If SAM 3 grows "
            "one, wire it here and close design O6; until then Sam2Segmenter.generic is "
            "the class-agnostic pass."
        )


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
