"""T414 pre-test (T398 lead's suggestion, 2026-10-19): certified box half-width around slab points.
Exact slab sampling: x_p3 = r ~ phi(r) Phi(r)^7, x_p2 ~ phi(x)(1 - Phi(x)) on [r + t0, inf), x_p1 ~ phi on [x_p2, inf),
lows ~ phi on (-inf, r] (inverse-CDF on fine grids). For each point, u = homogeneity-reduced coordinates; bisection on the
largest half-width w of the u-cube (clipped to u_a, u_l >= 0) on which CROWN proves some L_j - L_p3 > 0. Points that are
events are reported separately. Kill line (lead): median w < ~0.1.
usage: python pretest_w.py p1,p2,p3 [n]"""
import sys

import numpy as np
import torch as th
from scipy import stats

import ibab

th.set_num_threads(1)
tri = tuple(int(v) for v in sys.argv[1].split(",")); p1, p2, p3 = tri; n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
P, q, _, _ = ibab.coords(tri); lows = [i for i in range(10) if i not in tri]; rng = np.random.default_rng(4142); t0 = ibab.T0
grid = np.linspace(-6, 6, 200001); pr = stats.norm.pdf(grid) * stats.norm.cdf(grid) ** 7 * stats.norm.sf(grid + t0) ** 2
cdf = np.cumsum(pr); cdf /= cdf[-1]; r = np.interp(rng.uniform(size=n), cdf, grid)    # x_p2 tail weight ~ sf^2 / 2
x = np.zeros((n, 10)); x[:, p3] = r
for i in range(n):                                                    # x_p2 | r: phi(x)(1 - Phi(x)) on [r + t0, inf)
    gg = np.linspace(r[i] + t0, r[i] + t0 + 8, 20001); d = stats.norm.pdf(gg) * stats.norm.sf(gg); c = np.cumsum(d); c /= c[-1]
    x[i, p2] = np.interp(rng.uniform(), c, gg)
    x[i, p1] = stats.norm.isf(rng.uniform() * stats.norm.sf(x[i, p2]))
    x[i, lows] = stats.norm.ppf(rng.uniform(size=7) * stats.norm.cdf(r[i]))
g = x[:, p2] - x[:, p3]; y = x / g[:, None]
u = th.as_tensor(np.stack([y[:, p3], y[:, p1] - y[:, p2]] + [y[:, p3] - y[:, l] for l in lows], 1))
m = ibab.forward_margin(th.as_tensor(y), p3); m = th.cat([m[:, :p3], m[:, p3 + 1:]], 1); ev = (m <= 0).all(1)
assert (g >= t0).all() and (u[:, 1:] >= 0).all()
lo_w, hi_w = th.full((n,), 1e-7, dtype=th.float64), th.full((n,), 2.0, dtype=th.float64)


def certified(w):
    lo = u - w[:, None]; hi = u + w[:, None]; lo[:, 1:] = lo[:, 1:].clamp(min=0)
    lb, _ = ibab.margins(lo, hi, P, q, p3); return (lb > 0).any(1)


ok0 = certified(lo_w)
for _ in range(40):                                                   # geometric bisection
    mid = (lo_w * hi_w).sqrt(); c = certified(mid); lo_w = th.where(c, mid, lo_w); hi_w = th.where(c, hi_w, mid)
w = th.where(ok0, lo_w, th.zeros_like(lo_w)).numpy(); ne = ~ev.numpy()
print("slab points %d (events %d); true best margin median %.2f" % (n, int(ev.sum()), float(m.max(1).values.median())))
print("certified u-half-width w over non-event points: median %.3g p10 %.3g p90 %.3g min %.3g; uncertifiable even at 1e-7: %d" % (
    np.median(w[ne]), np.quantile(w[ne], .1), np.quantile(w[ne], .9), w[ne].min(), int((w[ne] == 0).sum())))
print("same in x-units (w * g): median %.3g p10 %.3g" % (np.median((w * g)[ne]), np.quantile((w * g)[ne], .1)))
print("(lead's kill line: median w < ~0.1 -> >= 1e9 boxes)")
# covering estimate: boxes ~ int du / v(u) = E_slab[1 / (v p)], v = (2w)^9 (cube clipped at u_a, u_l = 0 ignored),
# p(u) = f(|y|^2) / Z the slab density in u, restricted to the highest-density 99.8% (0.2% = 1.1e-10 may stay uncharged)
s2 = (y ** 2).sum(1); ww = s2 * t0 ** 2 / 2
f = 16 * (2 * np.pi) ** -5 * s2 ** -5 * 24 * np.exp(-ww) * (1 + ww + ww ** 2 / 2 + ww ** 3 / 6 + ww ** 4 / 24)
Z = 5.5975e-8; pu = f / Z; keep = ne & (pu >= np.quantile(pu, .002)) & (w > 0)
est = np.mean(np.where(keep, 1 / ((2 * np.where(w > 0, w, 1)) ** 9 * pu), 0))
print("covering estimate (99.8%% of slab mass, cube boxes at the certified w): %.2e boxes; with w halved %.2e" % (est, est * 2 ** 9))
