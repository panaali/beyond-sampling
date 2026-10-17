"""Diagnostic only (random inputs, NOT part of the estimator): how many gate-prefix branches does an
exact branch-and-project method face at T=3..6?  Counts distinct gate prefixes per step (in n samples),
with and without conditioning on the (argmax, 2nd-argmax) order cell, and the error rate.
usage: python diag_branches.py [n] [T...]"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from leaderboard import _zoo

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
Ts = [int(a) for a in sys.argv[2:]] or [3, 4, 5, 6]
rng = np.random.default_rng(0)
for T in Ts:
    for sd in range(5):
        Wi, Wh, Wo = _zoo(16, T, sd)
        x = rng.standard_normal((n, T))
        h = np.zeros((n, 16)); keys = [np.zeros(n, dtype=object) for _ in range(1)]
        key = np.array([b""] * n, dtype=object)
        counts = []
        for t in range(T):
            z = h @ Wh.T + x[:, t:t + 1] * Wi[None]
            g = z > 0
            h = np.where(g, z, 0)
            pk = np.packbits(g, axis=1)
            key = np.array([k + bytes(r) for k, r in zip(key, pk)], dtype=object)
            u, c = np.unique(key, return_counts=True)
            counts.append((len(u), int((c == 1).sum())))
        o = np.argsort(x, 1)
        cell = o[:, -1] * T + o[:, -2]
        ck = np.array([k + bytes([c]) for k, c in zip(key, cell)], dtype=object)
        uc = len(np.unique(ck))
        err = (h @ Wo.T).argmax(1) != o[:, -2]
        print(f"T={T} seed{sd}: prefixes/step {[c[0] for c in counts]} (singletons at T {counts[-1][1]}); "
              f"leaves x cell {uc}; err {err.mean():.4f}", flush=True)
