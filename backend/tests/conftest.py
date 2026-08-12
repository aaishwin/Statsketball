"""Shared pytest fixtures for the Statsketball backend test suite."""

import os
import sys
from pathlib import Path

# faiss-cpu and sklearn/hdbscan both bundle OpenMP; allow duplicate load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BACKEND_DIR: Path = Path(__file__).resolve().parent.parent

# Make `app` importable exactly the way run_api.py does.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

GOLDEN_DIR: Path = Path(__file__).resolve().parent / "golden"
DATA_DIR: Path = BACKEND_DIR.parent / "data" / "nba-aba-baa-stats" / "versions" / "56"
FAISS_OUTPUT_DIR: Path = BACKEND_DIR / "faiss_output"
