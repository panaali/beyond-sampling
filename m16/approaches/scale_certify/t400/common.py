"""T426 shim: T400's verify.py imports `common` for the weights; this serves a T421 grid model (env T426_TAG)."""
import os

import numpy as np

_d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data", "t426", "weights",
                          os.environ["T426_TAG"] + ".npz"))
Wi, Wh, Wo = (_d[k].astype(np.float64) for k in ("Wi", "Wh", "Wo"))
