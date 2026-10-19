"""T424 pre-registered threshold rule (fixed before any estimate or truth; README 'Pre-registration').
Model-free part: the exact law of g23 = v(2) - v(3) for T iid N(0,1) (depends on T only):
  P_T(g23 > t0) = int C3 phi(v) Phi(v)^(T-3) (1 - Phi(v + t0))^2 dv,  C3 = T(T-1)(T-2)/2.
Model-level factor: the registry's plain-MC error rate e = 1 - acc (a common quantity, no rare-event measurement).
Predicted p(t0) = P_T(g23 > t0) x c x e with c = 0.015 fixed from T366's (16,10) DEV3 conditional rates (log-centre
1.5e-3 at e ~ .1). Two thresholds per model: the t0 (on a .05 grid, rounded to the nearest) with predicted p = 1e-7 and
1e-8, the log-centres of [1e-9, 1e-6] with a decade of margin each way.
usage: python thresholds.py -> prints the table for every registry model"""
import os
import sys

import numpy as np
from scipy.special import log_ndtr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
C_FACTOR = 0.015
TARGETS = (1e-7, 1e-8)
STEP = 0.05
GRID = np.linspace(-9.0, 7.0, 1_600_001)


def log_pgap(T, t0):
    lf = np.log(T * (T - 1) * (T - 2) / 2) - GRID ** 2 / 2 - 0.5 * np.log(2 * np.pi) + (T - 3) * log_ndtr(GRID) + 2 * log_ndtr(-(GRID + t0))
    m = lf.max(); f = np.exp(lf - m)
    return float(m + np.log(((f[1:] + f[:-1]) / 2 * np.diff(GRID)).sum()))


def p_gap(T, t0):
    return float(np.exp(log_pgap(T, t0)))


_TAB = {}


def choose(T, acc):
    grid = np.round(np.arange(0.5, 5.0 + 1e-9, STEP), 2)
    if T not in _TAB:
        _TAB[T] = np.array([log_pgap(T, t) for t in grid])
    out = []
    for p in TARGETS:
        target = np.log(p / (C_FACTOR * (1 - acc)))
        out.append(float(grid[np.abs(_TAB[T] - target).argmin()]))
    return out


if __name__ == "__main__":
    import models as M
    for r in M.registry():
        print(r["tag"], round(r["acc"], 4), choose(r["T"], r["acc"]))
