"""Leaderboard entry: stratified sampling + tie-swap antithetics (see README.md). Sampling-based, unbiased.

est(model, budget, rng) -> (estimate, cost). Cost in forward-pass equivalents:
  model evaluations (1 each; a swapped copy is a 2nd evaluation)
  + input generation: every candidate Gaussian vector drawn (also the rejected ones) at GEN_COST each
    (10 normals + sort of 10 + gap arithmetic ~ 200 FLOPs vs ~6e3 for a forward pass).
Inputs for all strata are drawn from ONE shared pool (stop when every stratum has its quota), so the number
of candidates is max_h n_h/P_h instead of sum_h n_h/P_h. Within each stratum the accepted draws are i.i.d.
from the stratum's conditional law, so the estimator stays unbiased.
Model-free precomputation reused across models (declared): stratum probabilities P (m16/stratified_design.json,
2e7 model-free draws) and the allocation (pilot on the ORIGINAL model only, check4.py).
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from estimators import P, SW_N, USE_SWAP, g_of, swap_minpair   # noqa: E402
from stratified import stratum, N_STRATA                        # noqa: E402

DEDUCTIVE = False
GEN_COST = 200 / 6e3


def pooled_draw(n, rng, chunk=4096):
    need = n.copy(); got = [[] for _ in range(N_STRATA)]; cand = 0
    while (need > 0).any():
        m = int(min(max(chunk, 1.05 * (need / P).max()), 2_000_000))
        x = rng.standard_normal((m, 10)); cand += m; h = stratum(x)
        for k in range(N_STRATA):
            if need[k] > 0:
                xk = x[h == k][:need[k]]; got[k].append(xk); need[k] -= len(xk)
    # note: candidates of a chunk beyond the last quota are wasted but still counted
    return [np.concatenate(g) for g in got], cand


def est(model, budget, rng):
    n = np.maximum(2, np.round(SW_N * budget).astype(int))
    xs, cand = pooled_draw(n, rng)
    acc, evals = 0.0, 0
    for h in range(N_STRATA):
        v = g_of(model, xs[h]); evals += len(v)
        if USE_SWAP[h]:
            v = (v + g_of(model, swap_minpair(xs[h])[0])) / 2; evals += len(v)
        acc += P[h] * v.mean()
    return float(acc), float(evals + GEN_COST * cand)
