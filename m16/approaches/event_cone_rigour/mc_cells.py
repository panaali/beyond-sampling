"""Plain MC (no proposal tuning) of gamma(cell ∩ N ∩ {g23 >= t0}) for every k-prefix cell of a triple at once:
draw N(0,1)^10, sort, keep g23 >= t0 and ||x||_inf <= NORM g23, put the top three on tri and the lows on a random
permutation of the other positions (exact: given the sorted values, every assignment has probability 1/720 and
the lows' assignment is uniform), run k steps, count prefixes. P(cell) = count / (720 * draws).
usage: python mc_cells.py p1,p2,p3 k t0 n_draws -> data/mc_cells_<tri>_k<k>_<t0>.npz"""
import sys

import numpy as np

import common
import rigour as R

tri = tuple(int(v) for v in sys.argv[1].split(",")); k = int(sys.argv[2]); t0 = float(sys.argv[3]); n = int(float(sys.argv[4]))
rng = np.random.default_rng(1234 + k)
lows = np.array([i for i in range(10) if i not in tri])
counts = {}; kept = 0; B = 2_000_000
for _ in range(n // B):
    Z = -np.sort(-rng.standard_normal((B, 10)), 1)
    g = Z[:, 1] - Z[:, 2]
    Z = Z[(g >= t0) & (np.abs(Z).max(1) <= R.NORM * g)]
    kept += len(Z)
    X = np.empty_like(Z); X[:, tri[0]] = Z[:, 0]; X[:, tri[1]] = Z[:, 1]; X[:, tri[2]] = Z[:, 2]
    perm = lows[np.argsort(rng.random((len(Z), 7)), 1)]
    np.put_along_axis(X, perm, Z[:, 3:], 1)
    hs = common.run(X)[:, 1:k + 1] > 0
    for r in np.packbits(hs, axis=2).reshape(len(Z), -1):
        key = r.tobytes(); counts[key] = counts.get(key, 0) + 1
keys = list(counts)
np.savez(f"data/mc_cells_{'-'.join(map(str, tri))}_k{k}_{t0}.npz", keys=np.array(keys, dtype=object),
         counts=np.array([counts[q] for q in keys]), draws=n, kept=kept)
print("draws", n, "kept", kept, "cells", len(keys))
