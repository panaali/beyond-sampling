"""Estimators with ARC's interface est(model, budget, rng) -> (estimate, evals). All sampling-based, unbiased.

- est_strat:      m16/stratified.py as published (baseline).
- est_strat_swap: same strata; in strata where the pilot says so, each draw x is scored as the average of
                  g(x) and g(s x), s = swap of the positions holding the two values with the smaller relative
                  top gap (the reflection t -> -t along the tie line (e_i - e_j)/sqrt2). s is a
                  measure-preserving involution that leaves the sorted values (hence the stratum) unchanged,
                  so E[g(s x) | stratum] = E[g(x) | stratum]: unbiased for any weights. Cost 2 evals.
                  Neyman allocation with cost: n_h ~ P_h sigma_h / sqrt(c_h) (pilot on original model only).
- est_tieline:    exact piecewise-linear integration along the tie line, restricted to the window on which
                  the stratum and min-gap pair are unchanged (conditional MC; tielines.Z_stratum), in strata
                  1..LINE_MAX, plain elsewhere. Cost = 2 x pieces (forward + tangent).
- est_swap_rqmc:  scrambled Sobol over all 10 coordinates, swap-average applied when min rel gap < C_SW.
"""
import json, os, sys
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
from stratified import stratum, draw, N_STRATA, EDGES, estimate as strat_estimate   # noqa: E402
from tielines import forward, Z_stratum                                            # noqa: E402

_D = json.load(open(os.path.join(HERE, "..", "..", "stratified_design.json")))
P, ALLOC0 = np.array(_D["P"]), np.array(_D["alloc"])


def g_of(model, x):
    return (forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]).astype(float)


def swap_minpair(x):
    N = len(x); o = np.argsort(-x, 1); xs = np.take_along_axis(x, o, 1); r = np.arange(N)
    g23 = (xs[:, 1] - xs[:, 2]) / np.maximum(np.abs(xs[:, 1]), 1e-12); g12 = (xs[:, 0] - xs[:, 1]) / np.abs(xs[:, 0])
    a = np.where(g23 < g12, 1, 0); b = a + 1
    y = x.copy(); y[r, o[r, a]] = x[r, o[r, b]]; y[r, o[r, b]] = x[r, o[r, a]]
    return y, np.minimum(g12, g23), xs[:, 1] <= 0


def est_sampling(model, budget, rng):
    x = rng.standard_normal((budget, 10)); return g_of(model, x).mean(), budget


def est_strat(model, budget, rng):
    return strat_estimate(lambda x: forward(model, x), budget, rng, P, ALLOC0)


def _alloc(sig, cost):
    w = P * sig / np.sqrt(cost); n_per_budget = w / (w * cost).sum()   # n_h = budget * n_per_budget
    return n_per_budget


# pilot on the ORIGINAL model (check4.py, seed 4, 400k draws): per-stratum sd of plain and swap terms
SIG_PLAIN = np.array([0.420, 0.480, 0.381, 0.271, 0.170, 0.103, 0.076])
SIG_SWAP = np.array([0.316, 0.251, 0.251, 0.192, 0.123, 0.073, 0.057])
USE_SWAP = SIG_SWAP * np.sqrt(2) < SIG_PLAIN
SW_SIG = np.where(USE_SWAP, SIG_SWAP, SIG_PLAIN); SW_COST = np.where(USE_SWAP, 2.0, 1.0)
SW_N = _alloc(SW_SIG, SW_COST)


def est_strat_swap(model, budget, rng):
    n = np.maximum(2, np.round(SW_N * budget).astype(int)); acc, used = 0.0, 0
    for h in range(N_STRATA):
        x = draw(h, n[h], rng); v = g_of(model, x); used += n[h]
        if USE_SWAP[h]:
            v = (v + g_of(model, swap_minpair(x)[0])) / 2; used += n[h]
        acc += P[h] * v.mean()
    return acc, used


# tie-line variant: pilot (check5.py) sd of Z and mean cost (2 x pieces) per stratum 1..4
SIG_LINE = {1: 0.206, 2: 0.228, 3: 0.171, 4: 0.119}; COST_LINE = {1: 6.6, 2: 6.8, 3: 5.7, 4: 6.2}
LN_SIG = SIG_PLAIN.copy(); LN_COST = np.ones(N_STRATA); LINE = np.zeros(N_STRATA, bool)
for h in SIG_LINE:
    if SIG_LINE[h] * np.sqrt(COST_LINE[h]) < SIG_PLAIN[h]:
        LN_SIG[h], LN_COST[h], LINE[h] = SIG_LINE[h], COST_LINE[h], True
LN_N = _alloc(LN_SIG, LN_COST)


def est_tieline(model, budget, rng, force=(1, 2)):
    """force: strata where the exact tie line is used regardless of the pilot (pilot says none pays off)."""
    use = LINE.copy(); use[list(force)] = True
    sig = np.where(use, [SIG_LINE.get(h, 1) for h in range(N_STRATA)], SIG_PLAIN)
    cost = np.where(use, [COST_LINE.get(h, 1) for h in range(N_STRATA)], 1.0)
    n = np.maximum(2, np.round(_alloc(sig, cost) * budget).astype(int)); acc, used = 0.0, 0
    for h in range(N_STRATA):
        x = draw(h, n[h], rng)
        if use[h]:
            Z, pieces = Z_stratum(model, x, EDGES); acc += P[h] * Z.mean(); used += 2 * pieces.sum()
        else:
            acc += P[h] * g_of(model, x).mean(); used += n[h]
    return acc, int(used)


C_SW = 0.05


def est_swap_rqmc(model, budget, rng, c=C_SW):
    """Expected cost per point 1 + P(min rel gap < c, not blind) ~ 1.17 at c=0.05; n = power of 2 <= budget/1.17
    is NOT used (it would waste budget); scrambled Sobol with arbitrary n is still unbiased."""
    n = int(budget / 1.173)
    u = qmc.Sobol(10, scramble=True, seed=rng).random(n)
    x = ndtri(np.clip(u, 1e-15, 1 - 1e-15))
    v = g_of(model, x); y, gmin, blind = swap_minpair(x); sel = (gmin < c) & ~blind
    v[sel] = (v[sel] + g_of(model, y[sel])) / 2
    return v.mean(), n + int(sel.sum())


def est_rqmc(model, budget, rng):
    x = ndtri(np.clip(qmc.Sobol(10, scramble=True, seed=rng).random(budget), 1e-15, 1 - 1e-15))
    return g_of(model, x).mean(), budget


def est_joint(model, budget, rng):
    from joint import est_joint as f
    return f(model, budget, rng)
