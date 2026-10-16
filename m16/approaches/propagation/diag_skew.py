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
