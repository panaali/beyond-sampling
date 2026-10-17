"""Leaderboard interface for exact_ladder: exact cone enumeration + deterministic cone measure (mechanistic).

est(model, budget, rng) ignores budget and rng (no random inputs); cost = counted FLOPs / FLOPs-per-FPE (rule 1).
A small in-process cache keyed by the weights only avoids recomputing the leaderboard's determinism-check calls;
it returns the same charged cost as the original computation.
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import exact_acc  # noqa: E402

SUPPORTS = [(16, 2), (16, 3), (16, 4), (16, 5)]
FIXED_COST = True
PRECOMPUTE_FLOPS = 0          # Grundmann-Moller node tables are model-free and tiny
_CACHE = {}


def _fpe_flops(H, T):
    f = lambda H, T: T * (2 * H * H + 2 * H) + 2 * H * T
    return 6e3 * f(H, T) / f(16, 10)


def est(model, budget, rng):
    key = hashlib.sha1(b"".join(np.ascontiguousarray(W).tobytes() for W in model)).hexdigest()
    if key not in _CACHE:
        a, info = exact_acc(model)
        H, T = len(model[0]), model[2].shape[0]
        _CACHE[key] = (a, info["flops"] / _fpe_flops(H, T))
        try:        # per-model log for the README tables (counts, FLOP breakdown); not used by the estimate
            import json
            info.update(key=key[:12], acc=a, fpe=_CACHE[key][1])
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "est_info.jsonl"), "a") as f:
                f.write(json.dumps({k: (float(v) if isinstance(v, np.floating) else v) for k, v in info.items()}) + "\n")
        except OSError:
            pass
    return _CACHE[key]
