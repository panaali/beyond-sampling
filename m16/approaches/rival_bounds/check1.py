"""T289 step-1 checks (diagnostic sampling): (a) bound validity l.lam <= m <= u.lam on sampled sector points, and how often the
bounds are informative; (b) poly_mass vs MC for random cuts. -> printed"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "understand_m43")]
import numpy as np, rb
from zoo44 import zoo
rng = np.random.default_rng(2894)
for T in (3, 4):
    m = zoo(4, T, 0); Wi, Wh, Wo = m
    for b in ("forward", "crown"):
        viol = 0; inf_lo = inf_up = n = 0; gapl = []
        for R, lab, key in rb.T279.roots(T):
            fl = rb.T279.Flops(); Lk, Uk, riv = (rb.forward_bounds if b == "forward" else rb.crown_bounds)(m, R, lab, fl)
            lam = rng.dirichlet(np.ones(len(R)), 20000); X = lam @ R; h = np.zeros((len(X), len(Wi)))
            for t in range(T): h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
            M = (h @ Wo.T)[:, [lab]] - (h @ Wo.T)[:, riv]; lo = lam @ Lk.T; up = lam @ Uk.T
            viol += int(((lo > M + 1e-9) | (up < M - 1e-9)).sum()); inf_lo += int((lo > 0).sum()); inf_up += int((up < 0).sum()); n += M.size
            gapl.append(np.median(up - lo) / max(np.median(np.abs(M)), 1e-12))
        print(f"H4T{T} {b}: violations {viol}/{n}; rival-points with lower>0 {inf_lo / n:.4f}, upper<0 {inf_up / n:.4f}; "
              f"median (u-l)/median|m| = {np.median(gapl):.1f}")
# (b) mass check
for d in (3, 4):
    R = np.linalg.qr(rng.standard_normal((d, d)))[0]; R = R * np.sign(np.linalg.det(R)); err = [0.0]
    for _ in range(3):
        a = rng.standard_normal(d); bvec = rng.standard_normal(d); fl = rb.T279.Flops()
        pm = rb.poly_mass(R, [a, bvec], fl, err)
        Z = rng.standard_normal((2_000_000, d)); lam = np.linalg.solve(R.T, Z.T).T
        ins = (lam >= 0).all(1); mc = (ins & (lam @ a <= 0) & (lam @ bvec <= 0)).mean()
        print(f"d={d} poly_mass {pm:.5f} MC {mc:.5f} +- {np.sqrt(mc / 2e6):.5f}")
