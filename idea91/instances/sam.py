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
from idea91.instances.automask import (
    SAM2_HF_PATH,
    SAM2_REVISION,
    AutoMaskSettings,
    Sam2AutomaticMasks,
)
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
        cache_vision_embeds: bool = True,
        processor=None,
        model=None,
    ) -> None:
        self.hf_path = hf_path
        self.revision = revision
        self.device = device
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.supports_generic = bool(has_generic_mode)
        self.cache_vision_embeds = cache_vision_embeds
        self._processor = processor
        self._model = model
        self._embed_key = None
        self._embeds = None
        self.encodes = 0   # image encodes actually run
        self.prompts = 0   # prompts served

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

    @staticmethod
    def _model_dtype(model):
        try:
            return next(model.parameters()).dtype
        except (AttributeError, StopIteration, TypeError):
            return None  # an injected fake in the tests

    def _vision_embeds(self, image: np.ndarray):
        """Encode the image once and reuse it across every prompt on it.

        P1 prompts *every* class label of an image (4.12 on average, up to 20),
        and K2 prompts the head noun plus each context phrase.  Re-encoding per
        prompt is the dominant cost; ``Sam3Model.forward`` documents
        ``vision_embeds`` as reusable, so the encoder runs once per image.

        The cache key is a content hash, not ``id(image)``: array ids are reused
        after garbage collection, and serving one image's embeddings for another
        would be a silent correctness bug.
        """
        import hashlib

        import torch

        contiguous = np.ascontiguousarray(image)
        key = (contiguous.shape, hashlib.sha1(contiguous).hexdigest())
        if self._embed_key != key:
            processor, model = self._load()
            pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
            with torch.inference_mode():
                self._embeds = model.get_vision_features(
                    pixel_values=pixel_values.to(self.device, self._model_dtype(model))
                )
            self._embed_key = key
            self.encodes += 1
        return self._embeds

    def _predict(self, image: np.ndarray, **prompt) -> list[dict]:
        """One forward pass, post-processed to original-resolution instances."""
        import torch

        processor, model = self._load()
        height, width = image.shape[:2]
        inputs = processor(images=image, return_tensors="pt", **prompt)
        dtype = self._model_dtype(model)

        def to_device(value):
            if not hasattr(value, "to"):
                return value
            # box coordinates arrive as float32 and meet bf16 weights in the
            # geometry encoder; text prompts are integer ids and never hit this
            if dtype is not None and getattr(value, "is_floating_point", None) and value.is_floating_point():
                return value.to(self.device, dtype)
            return value.to(self.device)

        inputs = {k: to_device(v) for k, v in inputs.items()}
        if self.cache_vision_embeds and dtype is not None:
            # forward takes pixel_values or vision_embeds, never both
            inputs.pop("pixel_values", None)
            inputs["vision_embeds"] = self._vision_embeds(image)
        self.prompts += 1
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
            # boxes nest [image][box][xyxy]; labels nest only [image][box].  The
            # type hint on Sam3Processor.__call__ says three levels for labels,
            # but its validator rejects that at runtime -- verified on the real
            # processor, not read off the signature.
            input_boxes=[[[float(v) for v in box]]],
            input_boxes_labels=[[1]],
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
    """The contingency backend, and the class-agnostic pass O6 resolves to.

    Two distinct jobs, and only the first is a contingency:

    * **box prompts** stand in for SAM 3 when it is unavailable -- design Section
      8's "K1 can start with SAM 2 masks prompted by the OpenImages ground-truth
      boxes, labelled as such".  ``is_pinned_backend`` is False, so every scene
      built this way carries ``kill_grade=False``.
    * **automatic masks** are not a contingency at all: SAM 3 has no
      generate-everything call, so P1/P3's class-agnostic masks come from here in
      the normal course of things (``idea91/instances/automask.py``).

    Loaded through ``transformers`` (``Sam2Model`` / ``Sam2Processor``), not the
    ``sam2`` PyPI package, whose 1.1.0 sdist declares no dependencies and does not
    match the upstream project.
    """

    name = "sam2.1_hiera_large"
    supports_concept = False
    supports_generic = True
    is_pinned_backend = False

    def __init__(
        self,
        hf_path: str = SAM2_HF_PATH,
        revision: str = SAM2_REVISION,
        device: str = "cuda",
        min_area_frac: float = 0.005,
        settings: "AutoMaskSettings | None" = None,
        processor=None,
        model=None,
        **legacy,
    ) -> None:
        # ``checkpoint=`` was the original-format path; accepted and ignored so
        # older call sites fail loudly on behaviour, not on a TypeError
        self.legacy_checkpoint = legacy.pop("checkpoint", None)
        if legacy:
            raise TypeError(f"unexpected arguments: {sorted(legacy)}")
        self.hf_path = hf_path
        self.revision = revision
        self.device = device
        self.min_area_frac = min_area_frac
        self.settings = settings
        self._processor = processor
        self._model = model
        self._auto = None

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "hf_path": self.hf_path}

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

    def from_box(self, image: np.ndarray, box: Box) -> np.ndarray:
        import torch

        processor, model = self._load()
        height, width = image.shape[:2]
        inputs = processor(
            images=image,
            input_boxes=[[[float(v) for v in box]]],
            return_tensors="pt",
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs, multimask_output=False)
            masks = processor.post_process_masks(
                out.pred_masks, original_sizes=[(height, width)], binarize=True
            )[0]
        arr = masks.squeeze().cpu().numpy()
        if arr.ndim == 3:  # several candidates: take the highest-scoring
            best = int(out.iou_scores.flatten().argmax())
            arr = arr[best]
        return arr.astype(bool)

    def generic(self, image: np.ndarray, min_area_frac: float | None = None) -> list[RawInstance]:
        """P1/P3's class-agnostic masks above 0.5% area."""
        if self._auto is None:
            self._auto = Sam2AutomaticMasks(
                hf_path=self.hf_path,
                revision=self.revision,
                device=self.device,
                settings=self.settings,
                processor=self._processor,
                model=self._model,
            )
        floor = self.min_area_frac if min_area_frac is None else min_area_frac
        return self._auto.generate(image, min_area_frac=floor)


class CompositeSegmenter(Segmenter):
    """SAM 3 for concept and box prompts, SAM 2 for the class-agnostic masks.

    This is how design O6 actually resolves. SAM 3 is the pinned backend and
    answers every prompt the design names by text or box; it has no
    generate-everything call, so P1/P3's class-agnostic masks -- the ones that
    make the exclusion set cover *unlabelled* objects -- come from SAM 2's
    automatic generator.

    ``is_pinned_backend`` stays True: every mask that decides a referent or a
    control instance comes from SAM 3, and SAM 2 only contributes obstacles to
    the exclusion set, which can only ever make the sampler more conservative.
    The two backends are named separately in ``describe()`` so an index row still
    says exactly what produced it.
    """

    name = "sam3+sam2"
    supports_concept = True
    supports_generic = True
    is_pinned_backend = True

    def __init__(self, concept: Segmenter | None = None, generic: Segmenter | None = None,
                 device: str = "cuda") -> None:
        self.concept_backend = concept or Sam3Segmenter(device=device)
        self.generic_backend = generic or Sam2Segmenter(device=device)
        self.revision = self.concept_backend.revision

    def describe(self) -> dict[str, object]:
        return {
            "segmenter": f"{self.concept_backend.name}+{self.generic_backend.name}",
            "segmenter_revision": (
                f"{self.concept_backend.revision}+{self.generic_backend.revision}"
            ),
            "pinned_backend": self.concept_backend.is_pinned_backend,
            "hf_path": getattr(self.concept_backend, "hf_path", ""),
        }

    def concept(self, image: np.ndarray, phrase: str) -> list[RawInstance]:
        return self.concept_backend.concept(image, phrase)

    def from_box(self, image: np.ndarray, box: Box) -> np.ndarray:
        return self.concept_backend.from_box(image, box)

    def generic(self, image: np.ndarray, min_area_frac: float = 0.005) -> list[RawInstance]:
        return self.generic_backend.generic(image, min_area_frac)


def get_segmenter(name: str = "sam3", **kwargs) -> Segmenter:
    if name in ("sam3", "sam-3"):
        return Sam3Segmenter(**kwargs)
    if name in ("sam2", "sam-2", "sam2.1"):
        return Sam2Segmenter(**kwargs)
    if name in ("composite", "sam3+sam2", "default"):
        return CompositeSegmenter(**kwargs)
    raise ValueError(f"unknown segmenter {name!r}")
