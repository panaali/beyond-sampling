"""Leaderboard entry: deterministic psychometric-observer estimator (v4, nv=ng=1, q=2, nq=12, clipped noise).
No random model inputs (DEDUCTIVE). Cost is fixed (~215k forward-pass equivalents) and ignores `budget`.
Model-free precomputation (tie-level quantiles, quadrature grid, truncated-normal bin moments) is free."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _m in ("core", "quad", "v1", "v3", "v4", "est"):
    sys.modules.pop(_m, None)            # avoid clashes with same-named modules of other approaches
sys.path.insert(0, _here)
sys.path.insert(1, os.path.dirname(os.path.dirname(_here)))
import est as _est  # noqa: E402

DEDUCTIVE = True
_f = _est.make(nv=1, ng=1, q=2, nq=12, noise=1.0, ver=4)


def est(model, budget=None, rng=None):
    acc, cost, *_ = _f(model)
    return float(acc), float(cost)
