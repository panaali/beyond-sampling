"""Diagnostic (NOT part of the estimator, uses sampling inside cells): what fraction of cone-boxes of half-width h
is truly all-correct (what a perfect, complete verifier could certify), vs what CROWN certifies, and how many
gate (neuron, step) states vary inside the cell."""
import numpy as np
import cone
from core import bounds, Wi0, Wh0, Wo0

rng = np.random.default_rng(7); N = 600; S = 300
x = rng.standard_normal((N, 10)); od = np.argsort(-x, 1)
ridx = {r: i for i, r in enumerate(cone.ROOTS)}
root = np.array([ridx[(o[0], o[1], 1 if x[n, o[1]] > 0 else -1)] for n, o in enumerate(od)])
w = np.zeros((N, 10)); w[:, 9] = 1
for n in range(N):
    a, b, s = cone.ROOTS[root[n]]; lam = abs(x[n, b]) + x[n, a] - x[n, b]
    w[n, 0] = abs(x[n, b]) / lam
    rest = [j for j in range(10) if j not in (a, b)]
    w[n, 1:9] = (x[n, b] - x[n, rest]) / lam
tg = cone.TGT[root]
for h in (0.1, 0.03, 0.01):
    wl = w - h; wu = w + h; wl[:, 9] = wu[:, 9] = 1
    wl[:, :9] = np.maximum(wl[:, :9], 0); wu[:, 0] = np.minimum(wu[:, 0], 1)
    lb, ub, _ = bounds(root, wl, wu, None, "crown", MS=cone.MS, TGT=cone.TGT)
    cert = (lb > 0).all(1)
    u = rng.uniform(size=(N, S, 10)); ws = wl[:, None] + u * (wu - wl)[:, None]
    xs = np.einsum("ntd,nsd->nst", cone.MS[root], ws).reshape(-1, 10)
    hs = np.zeros((len(xs), 16)); G = []
    for t in range(10):
        hs = np.maximum(hs @ Wh0.T + xs[:, t:t + 1] * Wi0[None], 0); G.append(hs > 0)
    G = np.stack(G, 1).reshape(N, S, 10, 16)
    pred = (hs @ Wo0.T).argmax(1).reshape(N, S)
    allc = (pred == tg[:, None]).all(1)
    nvar = (G.any(1) & ~G.all(1)).sum((1, 2))
    print(f"h={h}: CROWN-certified {cert.mean():.3f} | truly all-correct (sampled) {allc.mean():.3f} | "
          f"median #gates varying {np.median(nvar):.0f} | cells with no gate change {(nvar == 0).mean():.3f}",
          flush=True)
