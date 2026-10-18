"""Diagnostic: exact k-prefix region masses (prefix_mass) vs gate-word frequencies of 2e6 N(0,1) samples."""
import pickle
import sys

import numpy as np

import prefix_mass as P
from prefix_blind import target

k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
Wi, Wh, _ = target()
cells = pickle.load(open(f"data/cells_full_k{k}.pkl", "rb"))
m, _ = P.cone_masses(cells, k)
reg = {}
for (G, h), mi in zip(cells, m):
    reg[h] = reg.get(h, 0.0) + mi
n = 2_000_000; x = np.random.default_rng(7).standard_normal((n, k))
h = np.zeros((n, len(Wi))); words = []
for t in range(k):
    z = h @ Wh.T + x[:, t:t + 1] * Wi[None]; g = z > 0; h = np.where(g, z, 0); words.append(np.packbits(g, axis=1))
W = np.concatenate(words, 1)
u, cnt = np.unique(W, axis=0, return_counts=True)
emp = {bytes(r.tobytes()): c / n for r, c in zip(u, cnt)}
miss = [w for w in emp if w not in reg]
z = [(emp.get(w, 0) - p) / np.sqrt(max(p, 1e-12) * (1 - p) / n) for w, p in reg.items() if p > 1e-4]
print(f"k={k}: sampled words {len(emp)}, not among exact regions {len(miss)} (mass {sum(emp[w] for w in miss):.2e}); "
      f"z over {len(z)} regions with mass>1e-4: mean {np.mean(z):+.3f}, sd {np.std(z):.3f}, max|z| {np.abs(z).max():.2f}")
