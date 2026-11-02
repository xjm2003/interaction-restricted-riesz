"""Set bounded numerical threads and use archived local adapter modules."""
from pathlib import Path
import os, sys
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
