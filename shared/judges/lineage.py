"""The judge registry and the lineage rule (design 4.5, E91-0).

    Write the judge-lineage rule into the service config: judge A Qwen3.5-9B for
    rewards and the crop judge, judge B Gemma4-12B held out, the GroundLive
    generator of a different lineage from the judge, the edit verifier different
    from the reward judge.

Enforced here at load time rather than left as a convention, so a later change
that quietly points two roles at the same family fails immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).with_name("judges.yaml")
SENTINEL = "PIN_REQUIRED"


class LineageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Judge:
    key: str
    name: str
    hf_path: str
    revision: str
    lineage_family: str
    roles: tuple[str, ...]
    vram_gb: float
    max_model_len: int = 4096
    notes: str = ""

    @property
    def pinned(self) -> bool:
        return self.revision != SENTINEL


@lru_cache(maxsize=4)
def _registry(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else REGISTRY_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    judges = {
        key: Judge(
            key=key,
            name=spec["name"],
            hf_path=spec["hf_path"],
            revision=spec.get("revision", SENTINEL),
            lineage_family=spec["lineage_family"],
            roles=tuple(spec.get("roles", ())),
            vram_gb=float(spec.get("vram_gb", 0)),
            max_model_len=int(spec.get("max_model_len", 4096)),
            notes=spec.get("notes", ""),
        )
        for key, spec in data["judges"].items()
    }
    assignments: dict[str, str] = data["assignments"]
    for role, key in assignments.items():
        if key not in judges:
            raise LineageError(f"role {role!r} is assigned to unknown judge {key!r}")
        if role not in judges[key].roles:
            raise LineageError(
                f"role {role!r} is assigned to {key!r}, which does not list it in roles"
            )
    check_constraints(judges, assignments, data.get("lineage_constraints", []))
    return {"judges": judges, "assignments": assignments, "raw": data}


def check_constraints(
    judges: dict[str, Judge],
    assignments: dict[str, str],
    constraints: list[list[str]],
) -> None:
    """Two roles in a constraint pair must not share a lineage family."""
    for pair in constraints:
        a, b = pair
        if a not in assignments or b not in assignments:
            continue  # a role not used in this idea (e.g. the GroundLive generator)
        fam_a = judges[assignments[a]].lineage_family
        fam_b = judges[assignments[b]].lineage_family
        if fam_a == fam_b:
            raise LineageError(
                f"lineage rule violated: {a} and {b} both resolve to the "
                f"{fam_a!r} family ({assignments[a]}, {assignments[b]}). "
                "A model must not grade what its own family produced (design 4.5)."
            )


def for_role(role: str, path: str | None = None) -> Judge:
    reg = _registry(path)
    if role not in reg["assignments"]:
        raise LineageError(f"no judge assigned to role {role!r}")
    return reg["judges"][reg["assignments"][role]]


def serving_args(role: str, headroom_gb: float = 2.0) -> dict[str, Any]:
    """What ``serve.sh`` needs, with ``gpu_memory_utilization`` measured, not guessed."""
    judge = for_role(role)
    try:
        from shared.env.probe import gpu_memory_utilization

        util = gpu_memory_utilization(judge.vram_gb, headroom_gb=headroom_gb)
    except Exception:  # no GPU here (CI, or a laptop): fall back to the design's figure
        util = 0.88
    return {
        "role": role,
        "judge": judge.key,
        "name": judge.name,
        "hf_path": judge.hf_path,
        "revision": judge.revision,
        "max_model_len": judge.max_model_len,
        "gpu_memory_utilization": util,
    }
