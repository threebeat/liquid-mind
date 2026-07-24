"""Phase 1: stable specialist representations (experiment family
specialists_phase1_v1).

Two frozen echo-state specialists (lidar, body) with trainable 16-d message
readouts, local and relational predictors, and a mandatory baseline suite.
All artifacts live under results/specialists_phase1_v1/ and
models/specialists_phase1_v1/; nothing under Gate D paths is touched.
"""
import os

from common import MODELS_DIR, RESULTS_DIR

EXPERIMENT = "specialists_phase1_v1"
P1_RESULTS = os.path.join(RESULTS_DIR, EXPERIMENT)
P1_MODELS = os.path.join(MODELS_DIR, EXPERIMENT)

# Nominal decision cadence (240 Hz physics / 8 substeps = 30 Hz).
NOMINAL_DT = 8.0 / 240.0

# Prediction horizons in seconds -> decision counts at the nominal cadence.
HORIZONS_S = (0.25, 0.5, 1.0, 2.0)
PRIMARY_HORIZON_S = 0.5
HORIZON_STEPS = {h: int(round(h / NOMINAL_DT)) for h in HORIZONS_S}  # 8/15/30/60
MAX_HORIZON_STEPS = max(HORIZON_STEPS.values())

CONTEXT_LEN = 15  # decisions of context per window

# Disjoint environment-seed ranges per stage/split. The legacy 80-episode
# buffer (data/experience.npz) was collected with seeds 10_000..10_079 and is
# reserved for the smoke stage only.
SEED_RANGES = {
    "smoke": {"legacy": (10_000, 10_079)},
    "pilot": {"train": (20_000, 20_299),      # 300 episodes
              "val": (21_000, 21_074),        # 75
              "test": (22_000, 22_099)},      # 100 (pilot-test; barred from
                                              # confirmatory use)
    "confirmatory": {"train": (30_000, 30_499),   # 500
                     "val": (31_000, 31_099),     # 100
                     "test": (32_000, 32_199)},   # 200
}


def ensure_p1_dirs():
    for d in (P1_RESULTS, P1_MODELS,
              os.path.join(P1_RESULTS, "smoke"),
              os.path.join(P1_RESULTS, "pilot"),
              os.path.join(P1_RESULTS, "confirmatory")):
        os.makedirs(d, exist_ok=True)
