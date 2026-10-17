"""Implementation check on ARC's own setting (random halfspace intersections, Hilton 2025, S11):
P(x.v_i >= 0, i=1..k), v_i ~ N(0, I_n), estimated by sequential cut-projection at degree D, vs exact-enough MC
(4M samples; MC is used ONLY to check, not in any estimator).  usage: python check_hilton.py n k inst"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import sphere as S

n, k, inst = (int(a) for a in sys.argv[1:4])
rng = np.random.default_rng(11)
V = rng.standard_normal((inst, k, n)); V /= np.linalg.norm(V, axis=2, keepdims=True)
x = rng.standard_normal((4_000_000, n))
truth = np.array([np.all(x @ v.T >= 0, 1).mean() for v in V])
print(f"n={n} k={k}: mean p={truth.mean():.4f}; MC se ~{np.sqrt(truth.mean() / 4e6):.1e}")
for D in (2, 4, 6, 8, 12, 16):
    sp = S.Space(n, D)
    est = []
    for v in V:
        c = sp.one()[:, None]
        for a in v[:-1]:
            c = sp.cut(c, a[None])
        est.append(sp.mass_in(c, v[-1][None])[0])
    e = np.array(est) - truth
    print(f"D={D:2d} m={sp.m:5d}: RMSE {np.sqrt((e**2).mean()):.2e}  bias {e.mean():+.2e}  "
          f"MC-equivalent samples p(1-p)/MSE = {(truth*(1-truth)).mean()/(e**2).mean():.0f}", flush=True)
