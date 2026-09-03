"""Where the bytes live.

Design 4.1 (inherited from IDEA-11): under WSL2 the data, results and model
caches sit on the ext4 volume, never under ``/mnt/c`` -- and, for the same
throughput reason, not under ``/mnt/d`` either.  Only the source tree lives in
the Windows project folder.  Every root is overridable by environment variable
so the same code runs on a RunPod volume.

    VLMG_DATA_ROOT     raw and prepared datasets      default ~/vlmg-data
    VLMG_EDITS_ROOT    the edit bank and its index    default $VLMG_DATA_ROOT/edits
    VLMG_RESULTS_ROOT  results/<run_id>/              default ~/vlmg-results
    VLMG_TABLES_ROOT   tables/*.md                    default <repo>/tables
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _root(var: str, default: Path) -> Path:
    return Path(os.environ.get(var) or default).expanduser()


DATA_ROOT = _root("VLMG_DATA_ROOT", Path.home() / "vlmg-data")
EDITS_ROOT = _root("VLMG_EDITS_ROOT", DATA_ROOT / "edits")
RESULTS_ROOT = _root("VLMG_RESULTS_ROOT", Path.home() / "vlmg-results")
TABLES_ROOT = _root("VLMG_TABLES_ROOT", REPO_ROOT / "tables")

CONFIGS_ROOT = REPO_ROOT / "configs"
PROMPTS_ROOT = CONFIGS_ROOT / "prompts"
MODELS_ROOT = CONFIGS_ROOT / "models"

RAW = DATA_ROOT / "raw"
PREPARED = DATA_ROOT / "prepared"


def run_dir(run_id: str) -> Path:
    return RESULTS_ROOT / run_id


def items_path(set_name: str) -> Path:
    return PREPARED / set_name / "items.parquet"


def ensure(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def describe() -> dict[str, str]:
    return {
        "repo": str(REPO_ROOT),
        "data": str(DATA_ROOT),
        "edits": str(EDITS_ROOT),
        "results": str(RESULTS_ROOT),
        "tables": str(TABLES_ROOT),
    }
