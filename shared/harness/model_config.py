"""Model configuration files (SPEC v1.2 Section 1).

The field list is SPEC's, exactly.  Values that cannot be filled in until the
model is pulled -- a commit SHA, a coordinate convention that must be read off the
model card, a benchmark's null-box literal -- carry the sentinel
``PIN_REQUIRED``.  :func:`load` refuses to serve a config with any sentinel left
to a kill run, so an unpinned revision cannot reach a kill table by accident.

SPEC Section 2 also forbids override values in the YAML: the file carries the
resolution *policy* (and, for crop-based processors, the pinned crop settings),
and the run supplies ``min_pixels``/``max_pixels`` as run-level overrides that
the manifest records.  :func:`load` enforces that too.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from shared import paths

SENTINEL = "PIN_REQUIRED"

COORDINATE_CONVENTIONS = ("relative_1000", "absolute_resized", "percent_float", "loc_tokens")
BACKENDS = ("hf", "vllm")
OUTPUT_TYPES = ("box", "point", "set", "none")

REQUIRED_FIELDS = (
    "model_id",
    "hf_path",
    "revision",
    "dtype",
    "attn_impl",
    "backend_default",
    "resolution_policy",
    "prompt_templates",
    "coordinate_convention",
    "parser",
    "abstain_protocol",
    "output_type_support",
    "tokens",
)

#: Values a *run* supplies, never the YAML (SPEC Section 2).
FORBIDDEN_IN_YAML = ("min_pixels", "max_pixels")


class ModelConfigError(RuntimeError):
    pass


@dataclass
class ModelConfig:
    data: dict[str, Any]
    path: Path
    unpinned: list[str] = field(default_factory=list)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def config_hash(self) -> str:
        """The ``config_hash`` a manifest carries (SPEC Section 2)."""
        return hashlib.sha256(
            json.dumps(self.data, sort_keys=True, default=str).encode()
        ).hexdigest()

    @property
    def pinned(self) -> bool:
        return not self.unpinned


def _find_sentinels(node: Any, trail: str = "") -> list[str]:
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in _find_sentinels(v, f"{trail}.{k}" if trail else k)]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _find_sentinels(v, f"{trail}[{i}]")]
    return [trail] if node == SENTINEL else []


def _check(data: dict[str, Any], path: Path) -> list[str]:
    problems = [f"missing field {f}" for f in REQUIRED_FIELDS if f not in data]
    if problems:
        return problems
    if data["dtype"] != "bfloat16":
        problems.append(f"dtype must be bfloat16 (SPEC Section 1), got {data['dtype']!r}")
    if data["backend_default"] not in BACKENDS:
        problems.append(f"backend_default must be one of {BACKENDS}")
    conv = data["coordinate_convention"]
    if conv != SENTINEL and conv not in COORDINATE_CONVENTIONS:
        problems.append(f"coordinate_convention must be one of {COORDINATE_CONVENTIONS}")
    bad_types = set(data["output_type_support"]) - set(OUTPUT_TYPES)
    if bad_types:
        problems.append(f"output_type_support has values outside {OUTPUT_TYPES}: {sorted(bad_types)}")
    for key in FORBIDDEN_IN_YAML:
        if key in json.dumps(data.get("resolution_policy", {})) and "fields" not in str(
            data.get("resolution_policy", {})
        ):
            problems.append(
                f"{key} is a run-level override and must not carry a value in the YAML "
                "(SPEC Section 2)"
            )
    protocols = data.get("abstain_protocol", {})
    if "primary" not in protocols:
        problems.append("abstain_protocol needs a 'primary' entry (P12)")
    for name, spec in protocols.items():
        if not isinstance(spec, dict) or "none_patterns" not in spec:
            problems.append(f"abstain_protocol.{name} needs none_patterns (SPEC Section 1)")
    return problems


def load(model_id: str, *, non_kill: bool = False, root: Path | None = None) -> ModelConfig:
    root = root or paths.MODELS_ROOT
    path = root / f"{model_id}.yaml"
    if not path.is_file():
        raise ModelConfigError(f"no model config at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    problems = _check(data, path)
    if problems:
        raise ModelConfigError(f"{path}:\n  " + "\n  ".join(problems))
    if data["model_id"] != model_id:
        raise ModelConfigError(f"{path}: model_id {data['model_id']!r} != file name {model_id!r}")
    unpinned = _find_sentinels(data)
    if unpinned and not non_kill:
        raise ModelConfigError(
            f"{path}: still unpinned -> {unpinned}. A kill run needs a commit SHA and the "
            "model's real conventions (SPEC Section 1). Use --non-kill for smoke, parity "
            "and dev-slice runs."
        )
    return ModelConfig(data=data, path=path, unpinned=unpinned)


def list_models(root: Path | None = None) -> list[str]:
    root = root or paths.MODELS_ROOT
    return sorted(p.stem for p in root.glob("*.yaml"))
