"""Other estimators run on a new panel (DEV2), each with its own cost accounting:
  lb_est     an LB3 leaderboard estimator by name at a budget (cone_enum, propagation, branch_bound); cost in the
             estimator's FPE (6000 FLOPs) converted to 5948.
  v5_est     probe_v5 (T274a, ext/probe_v5.py): v5a = est, v5b = est_spread(rivals=2, n_low=3), v5q = est_full.
  tcone_bias T274b deductive root-cone baseline: DIAGNOSTIC of its approximation bias (paired with the network on
             n random inputs, as T274b measured it; 'perfect integration'); est = truth + paired bias; cost 180x11 FPE
             (integration uncounted: F is a lower bound)."""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cf  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


_cache = {}


def lb_est(model, name, budget, return_parts=False):
    import leaderboard as L
    if name not in _cache:
        _cache[name] = L.load_estimator(name)[0]
    e, c = _cache[name](model, budget, np.random.default_rng(0))
    c = c * 6000 / cf.FPE
    return (e, c, {}) if return_parts else (e, c)


def v5_est(model, which, budget, return_parts=False):
    if "v5" not in _cache:
        _cache["v5"] = _load("probe_v5_ext", os.path.join(HERE, "ext", "probe_v5.py"))
    V = _cache["v5"]
    f = {"v5a": V.est, "v5q": V.est_full, "v5b": lambda m, b, r: V.est_spread(m, b, r, rivals=2, n_low=3)}[which]
    e, c = f(model, budget, None)
    c = c * 6000 / cf.FPE
    return (e, c, {}) if return_parts else (e, c)


def tcone_bias(model, m, n=200_000, return_parts=False, name="", truth=None):
    if "fb" not in _cache:
        _cache["fb"] = _load("frozen_bias", os.path.join(HERE, "ext", "t274b_frozen_bias.py"))
        _cache["cb"] = _load("t274b_cone_bias", os.path.join(HERE, "ext", "t274b_cone_bias.py"))
    FB, CB = _cache["fb"], _cache["cb"]
    Wi, Wh, Wo = model; rng = np.random.default_rng(286_900 + sum(map(ord, name))); d = 0
    for s in range(0, n, 100_000):
        x = rng.standard_normal((min(100_000, n - s), 10)); y = np.argsort(-x, 1)[:, 1]
        h = np.zeros((len(x), 16))
        for t in range(10):
            h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
        ok = (h @ Wo.T).argmax(1) == y
        G = FB.gates(model, CB.canonical_cone(x))
        d += int(((FB.approx_pred(model, x, G, m) == y).astype(int) - ok).sum())
    e = truth + d / n
    return (e, 180 * 11.0, dict(paired_bias=d / n)) if return_parts else (e, 180 * 11.0)
