"""Shared helpers: config loading and paths."""
import os

# On Windows, Intel MKL's threaded layer crashes (0xc06d007f) when NumPy BLAS
# runs in the same process as PyBullet + Python multiprocessing. Forcing the
# sequential layer avoids it. Must be set before MKL initializes, so we do it
# here — common is imported at the top of every entry point.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(ROOT, "configs", "default.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for d in (MODELS_DIR, DATA_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)
