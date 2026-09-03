"""Run manifests and the immutable result directory (SPEC v1.2 Section 2).

``results/<run_id>/`` holds ``outputs.parquet`` and ``manifest.json``.  The
manifest's ``file_sha256`` map is written last and covers every other file in
the directory, which is what makes the directory immutable: :func:`read` recomputes
the hashes and refuses a directory that changed after the manifest was written.

Two rules from SPEC that analysis code depends on:

* ``--non-kill`` marks acceptance, parity, smoke and dev-slice runs, and
  :func:`require_kill_runs` refuses to join those ids into a kill table.
* ``validity_run_ids`` lists the acceptance and parity runs a kill run relies on
  (SPEC Section 5), so a kill table can name the parity run that licensed its
  backend.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.harness.schema import sha256_file

MANIFEST_NAME = "manifest.json"

#: Every field of SPEC Section 2, in the order the prose lists them.
REQUIRED_FIELDS = (
    "run_id",
    "non_kill",
    "prereg_version",
    "code_sha",
    "config_hash",
    "prompt_hashes",
    "model_id",
    "model_revision",
    "dataset_revisions",
    "overrides",
    "effective_resolution",
    "backend",
    "server_args",
    "rejected_requests",
    "seeds",
    "package_versions",
    "gpu_name",
    "idle_vram_mb",
    "started_at",
    "ended_at",
    "validity_run_ids",
    "file_sha256",
)


class ManifestError(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Manifest:
    run_id: str
    non_kill: bool
    prereg_version: str
    model_id: str
    model_revision: str
    backend: str
    code_sha: str = ""
    config_hash: str = ""
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    dataset_revisions: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    effective_resolution: dict[str, Any] = field(default_factory=dict)
    server_args: dict[str, Any] = field(default_factory=dict)
    rejected_requests: int = 0
    seeds: dict[str, int] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)
    gpu_name: str = ""
    idle_vram_mb: int = 0
    started_at: str = field(default_factory=utcnow)
    ended_at: str = ""
    validity_run_ids: list[str] = field(default_factory=list)
    file_sha256: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def write(run_dir: str | Path, manifest: Manifest, *, allow_overwrite: bool = False) -> Path:
    """Hash every file in ``run_dir``, then write ``manifest.json`` last.

    The directory is immutable afterwards: a second write without
    ``allow_overwrite`` raises.
    """
    run_dir = Path(run_dir)
    target = run_dir / MANIFEST_NAME
    if target.exists() and not allow_overwrite:
        raise ManifestError(
            f"{target} exists; a result directory is immutable after the manifest is written "
            "(SPEC Section 2). Use a new run id."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.ended_at:
        manifest.ended_at = utcnow()
    manifest.file_sha256 = {
        p.relative_to(run_dir).as_posix(): sha256_file(p)
        for p in sorted(run_dir.rglob("*"))
        if p.is_file() and p.name != MANIFEST_NAME
    }
    missing = [f for f in REQUIRED_FIELDS if f not in manifest.to_json()]
    if missing:
        raise ManifestError(f"manifest is missing SPEC Section 2 fields: {missing}")
    target.write_text(json.dumps(manifest.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    return target


def read(run_dir: str | Path, *, verify: bool = True) -> Manifest:
    """Read a manifest and (by default) re-verify the directory's file hashes."""
    run_dir = Path(run_dir)
    target = run_dir / MANIFEST_NAME
    if not target.exists():
        raise ManifestError(f"no manifest at {target}: the run did not finish")
    data = json.loads(target.read_text(encoding="utf-8"))
    unknown = set(data) - set(REQUIRED_FIELDS)
    if unknown:
        raise ManifestError(f"{target}: fields outside SPEC Section 2: {sorted(unknown)}")
    manifest = Manifest(**data)
    if verify:
        for name, digest in manifest.file_sha256.items():
            path = run_dir / name
            if not path.exists():
                raise ManifestError(f"{run_dir}: {name} is in the manifest but missing on disk")
            if sha256_file(path) != digest:
                raise ManifestError(
                    f"{run_dir}: {name} changed after the manifest was written "
                    "(result directories are immutable, SPEC Section 2)"
                )
        on_disk = {
            p.relative_to(run_dir).as_posix()
            for p in run_dir.rglob("*")
            if p.is_file() and p.name != MANIFEST_NAME
        }
        added = on_disk - set(manifest.file_sha256)
        if added:
            raise ManifestError(f"{run_dir}: files added after the manifest: {sorted(added)}")
    return manifest


def require_kill_runs(run_dirs: list[str | Path]) -> list[Manifest]:
    """SPEC Section 2: analysis refuses to join non-kill run ids into kill tables."""
    manifests = [read(d) for d in run_dirs]
    non_kill = [m.run_id for m in manifests if m.non_kill]
    if non_kill:
        raise ManifestError(
            f"non-kill run ids cannot enter a kill table (SPEC Section 2): {non_kill}"
        )
    return manifests


def package_versions(packages: tuple[str, ...] = ()) -> dict[str, str]:
    """Installed versions of the packages a manifest should pin."""
    from importlib.metadata import PackageNotFoundError, version

    default = (
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "timm",
        "vllm",
        "numpy",
        "pyarrow",
        "pandas",
        "pillow",
        "opencv-python-headless",
        "scikit-image",
        "scipy",
    )
    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in packages or default:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out
