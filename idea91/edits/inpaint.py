"""Inpainters behind one interface (design 4.3, P2).

The pinned editor is big-LaMa; :class:`LamaInpainter` loads the TorchScript
``big-lama.pt`` and is the only backend whose output may enter a kill bank.
:class:`TeleaInpainter` is OpenCV's diffusion fill: no weights, no download, and
useful for exactly two things -- exercising the pipeline end to end before the
LaMa weights land, and giving the K1 gate a *known-dirty* editor to prove the
classifiers can separate something (its smear is coarse enough that a gate
blind to it would be a broken gate, not a clean editor).

Every backend reports ``name``, ``weights_sha256`` and ``settings_hash``, which
go into ``edits/index.parquet`` so a bank can never be traced to the wrong
editor.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shared.harness.schema import sha256_file


class Inpainter(ABC):
    """Fill ``mask`` in ``image``.  Both are window-sized; the mask is the dilated hole."""

    name: str = "abstract"
    kill_grade: bool = False  #: may its output enter a kill bank?

    @abstractmethod
    def fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 in, RGB uint8 HxWx3 out.  Only in-hole pixels matter:
        the caller composites in-mask (P2), so anything outside is discarded."""

    @property
    def weights_sha256(self) -> str:
        return ""

    @property
    def settings(self) -> dict[str, Any]:
        return {}

    @property
    def settings_hash(self) -> str:
        blob = json.dumps({"name": self.name, **self.settings}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class TeleaInpainter(Inpainter):
    """OpenCV's Telea fill.  Not kill-grade: it is a diffusion smear, not a model."""

    name = "opencv_telea"
    kill_grade = False

    def __init__(self, radius: int = 7) -> None:
        self.radius = radius

    @property
    def settings(self) -> dict[str, Any]:
        return {"radius": self.radius}

    def fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out = cv2.inpaint(bgr, m, self.radius, cv2.INPAINT_TELEA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


class LamaInpainter(Inpainter):
    """big-LaMa via TorchScript (``big-lama.pt``), the pinned editor of P2.

    Weights are pinned by sha256 in ``shared/env/PINS.md``; the file is not in
    the repository.  Set ``VLMG_LAMA_PT`` or pass ``weights=``.
    """

    name = "big_lama"
    kill_grade = True

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "cuda",
        pad_multiple: int = 8,
    ) -> None:
        import os

        self.weights_path = Path(weights or os.environ.get("VLMG_LAMA_PT", "")).expanduser()
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                "big-LaMa TorchScript weights not found. Download big-lama.pt, record its "
                "sha256 in shared/env/PINS.md, and set VLMG_LAMA_PT to its path."
            )
        self.device = device
        self.pad_multiple = pad_multiple
        self._model = None
        self._sha = ""

    @property
    def weights_sha256(self) -> str:
        if not self._sha:
            self._sha = sha256_file(self.weights_path)
        return self._sha

    @property
    def settings(self) -> dict[str, Any]:
        return {"pad_multiple": self.pad_multiple, "device": self.device}

    def _load(self):
        if self._model is None:
            import torch

            self._model = torch.jit.load(str(self.weights_path), map_location=self.device)
            self._model.eval()
        return self._model

    def fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        model = self._load()
        h, w = image.shape[:2]
        ph, pw = (-h) % self.pad_multiple, (-w) % self.pad_multiple
        img = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect") if (ph or pw) else image
        m = np.pad(mask.astype(np.uint8), ((0, ph), (0, pw))) if (ph or pw) else mask.astype(np.uint8)

        with torch.inference_mode():
            t_img = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)[None].to(self.device)
            t_m = torch.from_numpy((m > 0).astype(np.float32))[None, None].to(self.device)
            out = model(t_img, t_m)
            out = out[0].permute(1, 2, 0).mul_(255.0).clamp_(0, 255).to("cpu", torch.uint8).numpy()
        return np.ascontiguousarray(out[:h, :w])


def get_inpainter(name: str = "big_lama", **kwargs) -> Inpainter:
    if name in ("big_lama", "lama"):
        return LamaInpainter(**kwargs)
    if name in ("opencv_telea", "telea"):
        return TeleaInpainter(**kwargs)
    raise ValueError(f"unknown inpainter {name!r}")
