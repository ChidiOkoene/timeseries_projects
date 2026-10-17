"""Repo-relative data and artifact locations."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("TIMESERIES_DATA_DIR", REPO_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("TIMESERIES_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
EXAMPLES_DIR = REPO_ROOT / "examples"
# Legacy local chart / YOLO trees (often still under codes/ on disk)
CODES_DIR = REPO_ROOT / "codes"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def ensure_runtime_dirs() -> None:
    """Create local data/artifact dirs if missing (does not download data)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
