# DIAGNOSIS ONLY: is the failure of Gaussian closure due to non-Gaussian (skewed) margins? Per final (p1,p2),
# compare empirical P(correct) with predictions from the first 2 / 4 cumulants of the margin to the strongest
# wrong logit, and of the margin to the true 3rd-largest's logit (single competitor).
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from scipy.stats import norm
from common import Wi, Wh, Wo, run
x = np.random.default_rng(3).standard_normal((1_000_000, 10)); hs = run(x)
L = hs[:, -1] @ Wo.T
o = np.argsort(-x, 1); p1, p2 = o[:, 0], o[:, 1]
tgt = L[np.arange(len(x)), p2]
L2 = L.copy(); L2[np.arange(len(x)), p2] = -np.inf
marg = tgt - L2.max(1)
def edgeworth_cdf0(m):
    mu, sd = m.mean(), m.std(); z = (0 - mu) / sd
    k3 = ((m - mu) ** 3).mean() / sd ** 3; k4 = ((m - mu) ** 4).mean() / sd ** 4 - 3
    # Edgeworth expansion of CDF at z
    He2, He3, He5 = z * z - 1, z ** 3 - 3 * z, z ** 5 - 10 * z ** 3 + 15 * z
    F = norm.cdf(z) - norm.pdf(z) * (k3 / 6 * He2 + k4 / 24 * He3 + k3 ** 2 / 72 * He5)
    return 1 - norm.cdf(z), 1 - F, k3
res = []
for a in range(10):
    for b in range(10):
        if a == b: continue
        s = (p1 == a) & (p2 == b); m = marg[s]
        g, e, k3 = edgeworth_cdf0(m)
        res.append((s.mean(), (m > 0).mean(), g, e, k3))
r = np.array(res); w = r[:, 0] / r[:, 0].sum()
print('true acc', (w * r[:, 1]).sum(), ' gaussian-margin prediction', (w * r[:, 2]).sum(), ' 4-cumulant Edgeworth', (w * np.clip(r[:, 3], 0, 1)).sum())
print('mean skewness of the margin', (w * r[:, 4]).sum())
# scale-normalised margin (divide by m1 = max input, homogeneous of degree 0)
mn = marg / np.abs(x).max(1)
res = []
for a in range(10):
    for b in range(10):
        if a == b: continue
        s = (p1 == a) & (p2 == b); m = mn[s]
        g, e, k3 = edgeworth_cdf0(m); res.append((s.mean(), (m > 0).mean(), g, e, k3))
r = np.array(res)
print('scale-normalised margin: gaussian', (w * r[:, 2]).sum(), ' Edgeworth', (w * np.clip(r[:, 3], 0, 1)).sum(), ' skew', (w * r[:, 4]).sum())
# finer mechanism states: (p1,p2) x model-free bins of the relative top gaps
s3 = -np.sort(-x, 1)
g12 = (s3[:, 0] - s3[:, 1]) / np.abs(s3[:, 0]); g23 = (s3[:, 1] - s3[:, 2]) / np.maximum(np.abs(s3[:, 1]), 1e-12)
for nb in [3, 6, 10]:
    e12 = np.quantile(g12, np.linspace(0, 1, nb + 1)[1:-1]); e23 = np.quantile(g23, np.linspace(0, 1, nb + 1)[1:-1])
    blind = s3[:, 1] <= 0
    key = (p1 * 10 + p2) * 1000 + np.searchsorted(e12, g12) * 30 + np.searchsorted(e23, g23) + blind * 900000
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True); inv = inv.ravel()
    order = np.argsort(inv); bnd = np.r_[0, np.cumsum(cnt)]
    G = E = 0
    for k in range(len(u)):
        m = marg[order[bnd[k]:bnd[k + 1]]]
        if len(m) < 5: G += (m > 0).sum(); E += (m > 0).sum(); continue
        g, e, _ = edgeworth_cdf0(m); G += g * len(m); E += np.clip(e, 0, 1) * len(m)
    print(nb, 'gap bins per gap:', len(u), 'cells; gaussian', round(G / len(x), 4), 'edgeworth', round(E / len(x), 4))
