"""Float PRE-TEST (no rigour) of T398's option (c): homogeneous cuts c.x <= beta_c g23, beta_c = max over the event-
aware LP relaxation R of c.x / g23 (Charnes-Cooper LP), c in {+-e_i}. Measures by plain MC how much of the k-step
survivors' slab mass passes all cuts of its cell. usage: python cut_test.py p1,p2,p3 k t0 n_draws"""
import sys

import numpy as np
from scipy.optimize import linprog

import common
import verify as V

tri = tuple(int(v) for v in sys.argv[1].split(",")); k = int(sys.argv[2]); t0 = float(sys.argv[3]); n = int(float(sys.argv[4]))
name = "-".join(map(str, tri))
c = np.load(f"data/certs/surv_d0.333_{name}_k{k}.npz", allow_pickle=True); delta = float(c["delta"])
p2, p3 = tri[1], tri[2]
cuts = {}
for g in c["gates"]:
    g = g.astype(int)
    R = V.Relax(tri, g, k, delta); info = dict(lps=0)
    r = V._bounds(R, True, info)
    key = np.packbits(g[:k] > 0, axis=1).tobytes()
    if r is not None:
        cuts[key] = None; continue
    A, EA, b, Eb, lo, hi = R.program(R.nb, True)
    m, nv = A.shape
    # Charnes-Cooper: w = t v, t >= 0: A w - b t >= 0, lo t <= w <= hi t, g23(w) = 1; max c.w_x
    Aub = np.vstack([np.hstack([-A, b[:, None]]),
                     np.hstack([np.eye(nv), -hi[:, None]]), np.hstack([-np.eye(nv), lo[:, None]])])
    bub = np.zeros(len(Aub))
    geq = np.zeros(nv + 1); geq[p2], geq[p3] = 1, -1
    B = []
    for i in range(10):
        for s in (1, -1):
            cc = np.zeros(nv + 1); cc[i] = -s
            res = linprog(cc, A_ub=Aub, b_ub=bub, A_eq=geq[None], b_eq=[1], bounds=[(None, None)] * nv + [(0, None)], method="highs")
            B.append((i, s, -res.fun if res.status == 0 else np.inf))
    cuts[key] = B
print("cells", len(cuts), "with cuts", sum(v is not None for v in cuts.values()), flush=True)
rng = np.random.default_rng(778)
lows = np.array([i for i in range(10) if i not in tri])
inS = 0; passed = 0; Bn = 2_000_000
for _ in range(n // Bn):
    Z = -np.sort(-rng.standard_normal((Bn, 10)), 1)
    gg = Z[:, 1] - Z[:, 2]
    Z = Z[(gg >= t0) & (np.abs(Z).max(1) * delta <= gg)]
    X = np.empty_like(Z); X[:, tri[0]] = Z[:, 0]; X[:, tri[1]] = Z[:, 1]; X[:, tri[2]] = Z[:, 2]
    np.put_along_axis(X, lows[np.argsort(rng.random((len(Z), 7)), 1)], Z[:, 3:], 1)
    hs = common.run(X)
    keys = np.packbits(hs[:, 1:k + 1] > 0, axis=2).reshape(len(Z), -1)
    for x, r in zip(X, keys):
        cu = cuts.get(r.tobytes(), "out")
        if isinstance(cu, str):
            continue
        inS += 1
        if cu is None:
            continue
        g23 = x[p2] - x[p3]
        if all(s * x[i] <= beta * g23 + 1e-12 for i, s, beta in cu):
            passed += 1
f = 1 / (720 * n)
print("k", k, "C_true(survivors)=%.4g" % (inS * f), "after cuts=%.4g" % (passed * f), "ratio %.3g" % (passed / max(inS, 1)), "hits", inS, passed)
