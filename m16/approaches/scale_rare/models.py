"""T424 model access: the T421 registry (claude/t421-scale-study m16/approaches/scale_study/), read from a checkout
given by env T421_DIR (re-read on every call, so new grid points appear as they land). sha256 of each npz is checked
against the registry row."""
import hashlib
import json
import os

import numpy as np


def _dir():
    return os.environ["T421_DIR"]


def registry():
    return json.load(open(os.path.join(_dir(), "registry.json")))["models"]


def load(row):
    fn = os.path.join(_dir(), row["npz"]); d = np.load(fn)
    Wi, Wh, Wo = (np.asarray(d[k], np.float64).copy() for k in ("Wi", "Wh", "Wo"))
    assert Wh.shape == (row["H"], row["H"]) and Wo.shape == (row["T"], row["H"])
    h = hashlib.sha256(np.concatenate([Wi, Wh.ravel(), Wo.ravel()]).tobytes()).hexdigest()
    assert h == row["sha256_npz_weights"], row["tag"]
    return Wi, Wh, Wo


def by_tag(tag):
    return next(r for r in registry() if r["tag"] == tag)
