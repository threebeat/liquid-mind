import os
import sys

# On Windows, Intel MKL's threaded layer crashes (0xc06d007f) when NumPy BLAS
# runs in the same process as PyBullet. common.py sets this too, but pytest
# may import numpy before any test module imports common — so set it here,
# in the earliest hook point pytest offers.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

# make the repo root importable regardless of where pytest is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
