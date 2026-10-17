"""Diagnostic (random inputs used ONLY to measure true leaf masses; not part of the estimator).
Splits the harmonic estimate's bias (cells-first mode) into
  misallocation  sum_b (c0_b - mu_b) a_b       (propagated leaf mass vs true leaf mass, scored at the true accuracy)
  readout        sum_b (r_b - c0_b a_b)        (how the leaf's own mass is split at the logit halfspaces)
with mu_b, a_b the true mass and true correct fraction of leaf b (cone = order cell x gate history).
Also: bias carried by leaves that are entirely wrong ('dead') and by small leaves.
usage: python diag_bias.py T D model [n]"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from leaderboard import models
import hprop

T, D, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 4_000_000
m = models(16, T)[name]; Wi, Wh, Wo = m
e, info = hprop.estimate(m, D, cells="first", return_info=True)
stats = {}
rng = np.random.default_rng(5)
for _ in range(n // 500_000):
    x = rng.standard_normal((500_000, T)); h = np.zeros((len(x), 16)); bits = []
    for t in range(T):
        z = h @ Wh.T + x[:, t:t+1] * Wi[None]; g = z > 0; h = np.where(g, z, 0); bits.append(np.packbits(g, axis=1))
    o = np.argsort(x, 1); ok = (h @ Wo.T).argmax(1) == o[:, -2]
    B = np.concatenate(bits, 1)
    for k in range(len(x)):
        key = ((int(o[k, -1]), int(o[k, -2])), B[k].tobytes())
        s = stats.setdefault(key, [0, 0]); s[0] += 1; s[1] += int(ok[k])
mu = {k: v[0] / n for k, v in stats.items()}; a = {k: v[1] / max(v[0], 1) for k, v in stats.items()}
mis = rd = dead_b = small_b = 0.0; unmatched = 0.0
for key, c0, r in info["leaves"]:
    mb, ab = mu.get(key, 0.0), a.get(key, 1.0)
    if key not in mu:
        unmatched += c0
    mis += (c0 - mb) * ab; rd += r - c0 * ab
    if r == 0: dead_b += r - mb * ab
    if mb < 1e-4: small_b += r - mb * ab
print(f"{name} T={T} D={D}: est-truth(MC on these n) = {100*(e - sum(v[1] for v in stats.values())/n):+.3f} pp; "
      f"misallocation {100*mis:+.3f} pp, readout {100*rd:+.3f} pp; leaves {len(info['leaves'])} "
      f"(sampled {len(stats)}), mass on never-sampled leaves {unmatched:.4f}; "
      f"bias on dead leaves {100*dead_b:+.3f} pp, on leaves with true mass<1e-4 {100*small_b:+.3f} pp")
