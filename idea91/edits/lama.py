"""Load and run big-LaMa from its original checkpoint (design P2, PINS).

The released artefact is `big-lama.zip`: a hydra `config.yaml` plus a PyTorch
Lightning `models/best.ckpt`.  Neither of those libraries is installed here, and
neither is needed:

* the checkpoint is unpickled behind a meta-path finder that stubs the training
  packages the pickle references (`pytorch_lightning`, `omegaconf`, ...), since
  the only thing wanted from the file is `state_dict['generator.*']`;
* the config's `${a.b.c}` interpolations are resolved by a small resolver rather
  than by omegaconf.

The generator itself is the vendored `FFCResNetGenerator` (see
`vendor/NOTICE.md`).  Inference follows upstream's `default.py` exactly:

    masked = image * (1 - mask)
    predicted = generator(cat([masked, mask], dim=1))     # 4 channels in, 3 out
    inpainted = mask * predicted + (1 - mask) * image

so the model never sees the pixels under the hole, and everything outside the
hole is the original by construction -- which is what makes the in-mask
compositing assert of check 1c hold before it is even applied.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VENDOR_ROOT = Path(__file__).parent / "vendor"

#: Packages the Lightning pickle references but whose behaviour is irrelevant:
#: the state_dict is plain tensors.
_STUB_PREFIXES = ("pytorch_lightning", "omegaconf", "hydra")

_INTERPOLATION = re.compile(r"^\$\{([A-Za-z0-9_.]+)\}$")


class _StubAny:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __iter__(self):
        return iter(())


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []

        def __getattr__(name: str):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _StubAny

        module.__getattr__ = __getattr__
        return module

    def exec_module(self, module) -> None:
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUB_PREFIXES:
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


def _install_stubs() -> None:
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())


def _ensure_vendor_on_path() -> None:
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_interpolations(config: dict) -> dict:
    """Resolve omegaconf-style ``${a.b.c}`` references inside one config.

    big-LaMa's generator block uses three of them; resolving iteratively handles
    a reference that points at another reference.
    """

    def lookup(dotted: str):
        node: Any = config
        for part in dotted.split("."):
            node = node[part]
        return node

    def walk(node, depth=0):
        if isinstance(node, dict):
            return {k: walk(v, depth) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, depth) for v in node]
        if isinstance(node, str) and (m := _INTERPOLATION.match(node.strip())):
            if depth > 10:
                raise ValueError(f"interpolation cycle at {node}")
            return walk(lookup(m.group(1)), depth + 1)
        return node

    return walk(config)


@dataclass
class LamaCheckpoint:
    """A loaded big-LaMa: the generator config and its weights."""

    generator_kwargs: dict
    state_dict: dict
    config: dict

    @property
    def n_params(self) -> int:
        return sum(int(v.numel()) for v in self.state_dict.values())


def read_checkpoint(model_dir: str | Path) -> LamaCheckpoint:
    """Read ``config.yaml`` and ``models/best.ckpt`` from an unpacked big-lama."""
    import torch
    import yaml

    model_dir = Path(model_dir).expanduser()
    config_path = model_dir / "config.yaml"
    ckpt_path = model_dir / "models" / "best.ckpt"
    for path in (config_path, ckpt_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Unpack big-lama.zip (the URL is in the upstream README, "
                "recorded in idea91/edits/vendor/NOTICE.md) and point VLMG_LAMA_DIR at the "
                "resulting big-lama directory."
            )

    config = resolve_interpolations(yaml.safe_load(config_path.read_text()))
    generator = dict(config["generator"])
    kind = generator.pop("kind", None)
    if kind != "ffc_resnet":
        raise ValueError(f"only the ffc_resnet generator is wired up, config says {kind!r}")

    _install_stubs()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prefix = "generator."
    state = {
        k[len(prefix) :]: v for k, v in raw["state_dict"].items() if k.startswith(prefix)
    }
    if not state:
        raise ValueError(f"{ckpt_path} has no generator.* tensors")
    return LamaCheckpoint(generator_kwargs=generator, state_dict=state, config=config)


def build_generator(checkpoint: LamaCheckpoint, device: str = "cuda"):
    """Instantiate the vendored FFCResNetGenerator and load the weights."""
    _ensure_vendor_on_path()
    from saicinpainting.training.modules.ffc import FFCResNetGenerator  # type: ignore

    model = FFCResNetGenerator(**checkpoint.generator_kwargs)
    missing, unexpected = model.load_state_dict(checkpoint.state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"big-LaMa weights do not match the generator: {len(missing)} missing, "
            f"{len(unexpected)} unexpected (first missing: {missing[:3]})"
        )
    return model.to(device).eval()


def infer(model, image: np.ndarray, mask: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Upstream's inference path, on one window.

    ``image`` is HxWx3 uint8, ``mask`` is HxW with non-zero inside the hole.
    Returns the *inpainted* image: predicted pixels inside the hole, original
    pixels everywhere else.
    """
    import torch

    with torch.inference_mode():
        img = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
        m = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))[None]
        img = img[None].to(device)
        m = m[None].to(device)

        masked = img * (1 - m)
        predicted = model(torch.cat([masked, m], dim=1))
        inpainted = m * predicted + (1 - m) * img

        out = inpainted[0].permute(1, 2, 0).mul_(255.0).clamp_(0, 255)
        return out.to("cpu", torch.uint8).numpy()
