"""Machine facts that every manifest carries (SPEC Section 2, design 4.1).

Design 4.1: "Idle VRAM taken by the display is measured at startup and the
memory envelope is checked against the remaining amount."  On this host the
display holds ~3.4 GB of the 5090's 32 GB, so a 25 GB judge needs
``gpu_memory_utilization`` derived from what is actually free, not from a
constant.

Run ``python -m shared.env.probe`` to print the block that goes into
``notes/`` at setup time.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from shared import paths
from shared.harness.manifest import package_versions


@dataclass
class GpuInfo:
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    driver: str
    compute_capability: str = ""

    @property
    def idle_vram_mb(self) -> int:
        """VRAM already taken when nothing of ours is loaded (the display)."""
        return self.used_mb


def query_gpu(index: int = 0) -> GpuInfo | None:
    """``nvidia-smi`` without importing torch, so this runs before the GPU stack."""
    fields = "name,memory.total,memory.used,memory.free,driver_version,compute_cap"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if index >= len(out):
        return None
    parts = [p.strip() for p in out[index].split(",")]
    cc = parts[5] if len(parts) > 5 else ""
    return GpuInfo(
        name=parts[0],
        total_mb=int(float(parts[1])),
        used_mb=int(float(parts[2])),
        free_mb=int(float(parts[3])),
        driver=parts[4],
        compute_capability=cc,
    )


def gpu_memory_utilization(weights_gb: float, headroom_gb: float = 2.0, index: int = 0) -> float:
    """vLLM's ``gpu_memory_utilization`` for a model of ``weights_gb``.

    The fraction is of *total* VRAM and must leave the display's idle usage
    alone, so it is ``(free - headroom) / total`` capped at 0.95.  Design 4.1
    quotes ~0.88 for Gemma4-12B with a display attached; this computes it.
    """
    gpu = query_gpu(index)
    if gpu is None:
        raise RuntimeError("no GPU visible to nvidia-smi")
    usable_mb = gpu.free_mb - headroom_gb * 1024
    if usable_mb < weights_gb * 1024:
        raise RuntimeError(
            f"{gpu.name}: {gpu.free_mb} MB free, {gpu.used_mb} MB taken by the display; "
            f"{weights_gb:.1f} GB of weights plus {headroom_gb:.1f} GB headroom does not fit"
        )
    return round(min(0.95, (gpu.used_mb + usable_mb) / gpu.total_mb), 3)


def code_sha(repo: Path | None = None) -> str:
    """git HEAD if this is a repository, else a hash over the tracked sources.

    The design's freeze points (B0a, B0b) are git tags, so a real repository is
    the intended state; the fallback keeps manifests honest until then and says
    so with the ``tree:`` prefix.
    """
    repo = repo or paths.REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    import hashlib

    h = hashlib.sha256()
    for path in sorted(repo.rglob("*.py")):
        if any(part in {".venv", "__pycache__", ".git"} for part in path.parts):
            continue
        h.update(path.relative_to(repo).as_posix().encode())
        h.update(path.read_bytes())
    return "tree:" + h.hexdigest()[:40]


def probe() -> dict:
    gpu = query_gpu()
    return {
        "gpu": asdict(gpu) if gpu else None,
        "code_sha": code_sha(),
        "package_versions": package_versions(),
        "paths": paths.describe(),
    }


def main() -> None:
    print(json.dumps(probe(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
