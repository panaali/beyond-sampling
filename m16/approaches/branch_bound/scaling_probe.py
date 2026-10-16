"""Diagnostic (NOT part of the estimator): how big can certified cone-boxes be, and how many would tile the
mass? Boxes of half-width h in cone coordinates around Gaussian draws; certified fraction f(h) and box mass.
N(h) ~ E[1/mass(box)] estimates how many cells a partition at that resolution needs."""
import numpy as np, json
import cone
from core import bounds

rng = np.random.default_rng(5); N = 1500
x = rng.standard_normal((N, 10)); od = np.argsort(-x, 1)
ridx = {r: i for i, r in enumerate(cone.ROOTS)}
root = np.array([ridx[(o[0], o[1], 1 if x[n, o[1]] > 0 else -1)] for n, o in enumerate(od)])
w = np.zeros((N, 10)); w[:, 9] = 1
for n in range(N):
    a, b, s = cone.ROOTS[root[n]]; lam = abs(x[n, b]) + x[n, a] - x[n, b]
    w[n, 0] = abs(x[n, b]) / lam
    rest = [j for j in range(10) if j not in (a, b)]
    w[n, 1:9] = (x[n, b] - x[n, rest]) / lam
# check chart reconstruction
xr = np.einsum("ntd,nd->nt", cone.MS[root], w)
lam = np.abs(x[np.arange(N), od[:, 1]]) + x[np.arange(N), od[:, 0]] - x[np.arange(N), od[:, 1]]
assert np.allclose(xr * lam[:, None], x)
out = {}
for h in (0.3, 0.1, 0.03, 0.01, 0.003):
    wl = w - h; wu = w + h; wl[:, 9] = wu[:, 9] = 1
    wl[:, :9] = np.maximum(wl[:, :9], 0); wu[:, 0] = np.minimum(wu[:, 0], 1)
    lb, ub, _ = bounds(root, wl, wu, None, "crown", MS=cone.MS, TGT=cone.TGT)
    c = (lb > 0).all(1); wr = ((ub < 0) & np.isfinite(ub)).any(1)
    m = cone.measure(cone.SGN[root], wl, wu, 12)
    inv = 1 / np.maximum(m, 1e-300)
    out[h] = dict(cert_correct=c.mean(), cert_wrong=wr.mean(), med_mass=np.median(m),
                  N_tile_median=float(np.median(inv)), N_tile_q25=float(np.quantile(inv, .25)))
    print(h, out[h], flush=True)
json.dump(out, open("scaling_probe.json", "w"), indent=1, default=float)
