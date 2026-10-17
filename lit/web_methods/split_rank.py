"""Sanity test for the sequential low-rank (weighted-automaton / tensor-train) idea.
For a split after step t, build A[i,j] = 1[model correct on (prefix_i, suffix_j)] for iid Gaussian
prefixes/suffixes and report how fast its singular values decay: the relative L2 error of the best
rank-r approximation of the indicator (and of the error indicator 1-A) for several r.
Usage: OMP_NUM_THREADS=1 python lit/web_methods/split_rank.py [n] [seed]"""
import sys, numpy as np
sys.path.insert(0, 'm16')
from common import Wi, Wh, Wo, T
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
for t in (2, 3, 5, 7, 8):
    P = rng.standard_normal((n, t)); S = rng.standard_normal((n, T - t))
    h = np.zeros((n, len(Wi)))
    for k in range(t):
        h = np.maximum(h @ Wh.T + P[:, k:k+1] * Wi, 0)
    A = np.empty((n, n))
    for i in range(n):
        hh = np.repeat(h[i:i+1], n, 0)
        for k in range(T - t):
            hh = np.maximum(hh @ Wh.T + S[:, k:k+1] * Wi, 0)
        x = np.hstack([np.repeat(P[i:i+1], n, 0), S])
        A[i] = ((hh @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2])
    E = 1 - A
    s = np.linalg.svd(E, compute_uv=False)
    tot = (s**2).sum()
    tail = lambda r: np.sqrt(max(tot - (s[:r]**2).sum(), 0) / n**2)  # RMS pointwise error of rank-r approx
    rs = [1, 2, 5, 10, 20, 50, 100, 200, 400]
    print(f"t={t} err={E.mean():.4f}  RMS err of best rank-r approx of error indicator: " +
          " ".join(f"r{r}:{tail(r):.3f}" for r in rs if r < n), flush=True)
