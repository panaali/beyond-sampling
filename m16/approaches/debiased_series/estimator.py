"""Leaderboard entry: debiased probe series (two-level control variate on top of stratified sampling).

Depth rule (fixed in advance from variance_gain.py, which used the 300k-input variance table):
  budget < 128k  -> depth 0: no probes; plain stratified sampling (identical to m16/stratified.py)
  budget >= 128k -> depth 5: 7,200-eval probe surrogate as control variate, exactly-known E[g] from a
                    model-free table, stratified residual sampling.
Cost = model forward passes (probes + samples) + 0.037 per sample for the surrogate lookup (measured
time ratio). Input generation/rejection for strata is not counted (same as stratified.py).
Model-free precomputation reused across models (free, declared in README): strata probabilities
(stratified_design.json), table_it5.npy (16M model-free inputs). Allocation alloc_it5.json was set from a
pilot on the ORIGINAL model (design-time, like stratified_design.json).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds import est_ds, est_strat

DEDUCTIVE = False


def est(model, budget, rng):
    if budget >= 128_000:
        return est_ds(model, budget, rng, iters=5)
    return est_strat(model, budget, rng)
