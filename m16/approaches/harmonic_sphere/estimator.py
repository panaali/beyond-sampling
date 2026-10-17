"""Leaderboard interface (v3) for harmonic deduction-projection (T38). Deterministic: rng is ignored.

D is the largest degree whose cost (as implemented; FLOP counts of runs on seed0 and s0.02_0, NOT their accuracy
or truths, so no output is fitted) fits the budget, with a
floor of D=2 (at every leaderboard budget the floor applies: even the exact cone bookkeeping alone exceeds
16k FPE at T=4). Cost returned is the FLOPs actually counted by the run (harmonic cuts + double-description
cone updates + readout), converted at rule-1 FLOPs/FPE.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import hprop

SUPPORTS = [(16, 3), (16, 4), (16, 5), (16, 6)]
FIXED_COST = False
ADVICE = None
PRECOMPUTE_FLOPS = 0.0   # model-free cubature, basis and root order cells: < 1e10 FLOPs per (T, D); see README

# implemented cost per model (FPE) by (T, D), measured on seed0/s0.02_0 (README, conv_T*.log); used only to pick D
_COST = {3: {2: 2.0e3, 4: 1.1e4, 6: 4.2e4, 10: 2.5e5, 16: 1.4e6, 20: 3.3e6},
         4: {2: 3.2e5, 4: 5.8e6, 6: 4.2e7, 8: 1.9e8}}


def _fpe(H, T):
    return 6e3 * (T * (2 * H * H + 2 * H) + 2 * H * T) / (10 * (2 * 16 * 16 + 2 * 16) + 2 * 16 * 10)


def est(model, budget, rng):
    H, T = model[1].shape[0], model[2].shape[0]
    tab = _COST.get(T, {})
    D = max([d for d, c in tab.items() if c <= budget] or [2])
    e, flops = hprop.estimate(model, D, cells="first")
    return float(e), float(flops / _fpe(H, T))
