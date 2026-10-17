"""T80 variant (b) ceiling: can per-unit Hermite corrections with EXACT higher cumulants fix the first broken step?
For each unit of the trained M_16,T at step t, take the exact (MC) cumulants kappa_1..kappa_4 of its pre-activation
u_i and evaluate E[relu(u_i)] by the 1-D Edgeworth series around N(kappa_1, kappa_2) to orders 2 (Gaussian),
3 and 4 (incl. kappa_3^2) - what any per-neuron Hermite correction (ARC's nonlinear step) computes for the mean
when given perfect input cumulants. Compare with MC E[relu(u_i)].
Usage: python unit_ceiling.py T"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../m16"))
import leaderboard as LB  # noqa: E402

T = int(sys.argv[1]); H = 16
name, (Wi, Wh, Wo) = LB._base(H, T)
r = np.random.default_rng(3)
x = r.standard_normal((500_000, T))
z, w = np.polynomial.hermite_e.hermegauss(200); w = w / w.sum()
He3 = z ** 3 - 3 * z; He4 = z ** 4 - 6 * z ** 2 + 3; He6 = z ** 6 - 15 * z ** 4 + 45 * z ** 2 - 15
h = np.zeros((len(x), H))
print(f"model {name}: per-unit E[relu(u_i)], exact cumulants of u_i; rel. rms error over units (scale = rms MC mean)")
for t in range(T):
    u = h @ Wh.T + x[:, t:t + 1] * Wi[None]
    true = np.maximum(u, 0).mean(0)
    mu, sd = u.mean(0), u.std(0)
    s = (u - mu) / np.maximum(sd, 1e-12)
    g1, g2 = (s ** 3).mean(0), (s ** 4).mean(0) - 3
    f = np.maximum(mu[:, None] + sd[:, None] * z[None], 0)          # relu on the Gaussian grid (200 GH nodes)
    e2 = f @ w
    e3 = (f * (1 + g1[:, None] * He3 / 6)) @ w
    e4 = (f * (1 + g1[:, None] * He3 / 6 + g2[:, None] * He4 / 24 + g1[:, None] ** 2 * He6 / 72)) @ w
    sc = np.sqrt((true ** 2).mean())
    rel = lambda e: np.sqrt(((e - true) ** 2).mean()) / sc
    print(f"  step {t+1}: Gaussian {rel(e2):.3f}  +k3 {rel(e3):.3f}  +k4,k3^2 {rel(e4):.3f}   "
          f"(skewness range {g1.min():+.1f}..{g1.max():+.1f}, excess kurtosis up to {g2.max():.1f})")
    h = np.maximum(u, 0)
