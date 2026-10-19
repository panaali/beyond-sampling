"""Plain-MC truth of C_true(k) = P(x in some survivor cell at step k, top-3 at tri, g23 >= t0, N_delta), k = 1..K,
from T398's survivor files, in one pass (same sampler as mc_cells.py). Also counts event draws (pick = p3) and
checks that every event draw lies in the survivors at every k (soundness of the pruning: a miss would mean a
pruned cell contains an event point). usage: python surv_curve.py p1,p2,p3 t0 n_draws K"""
import sys

import numpy as np

import common

tri = tuple(int(v) for v in sys.argv[1].split(",")); t0 = float(sys.argv[2]); n = int(float(sys.argv[3])); K = int(sys.argv[4])
name = "-".join(map(str, tri))
surv = {}; delta = None
for k in range(1, K + 1):
    c = np.load(f"data/certs/surv_d0.333_{name}_k{k}.npz", allow_pickle=True)
    delta = float(c["delta"])
    surv[k] = {np.packbits(g[:k] > 0, axis=1).tobytes() for g in c["gates"]}
rng = np.random.default_rng(777)
lows = np.array([i for i in range(10) if i not in tri])
inS = {k: 0 for k in surv}; ev = 0; ev_miss = {k: 0 for k in surv}; kept = 0; B = 2_000_000
for _ in range(n // B):
    Z = -np.sort(-rng.standard_normal((B, 10)), 1)
    g = Z[:, 1] - Z[:, 2]
    Z = Z[(g >= t0) & (np.abs(Z).max(1) * delta <= g)]
    kept += len(Z)
    X = np.empty_like(Z); X[:, tri[0]] = Z[:, 0]; X[:, tri[1]] = Z[:, 1]; X[:, tri[2]] = Z[:, 2]
    np.put_along_axis(X, lows[np.argsort(rng.random((len(Z), 7)), 1)], Z[:, 3:], 1)
    hs = common.run(X)
    pick = (hs[:, -1] @ common.Wo.T).argmax(1) == tri[2]
    ev += int(pick.sum())
    for k in surv:
        keys = np.packbits(hs[:, 1:k + 1] > 0, axis=2).reshape(len(Z), -1)
        m = np.array([r.tobytes() in surv[k] for r in keys])
        inS[k] += int(m.sum()); ev_miss[k] += int((pick & ~m).sum())
f = 1 / (720 * n)
print("tri", tri, "t0", t0, "draws", n, "kept", kept, "P(slab, tri)=%.4g" % (kept * f), "events", ev, "P(E_tri)=%.4g" % (ev * f))
for k in surv:
    print("k", k, "survivors", len(surv[k]), "C_true=%.4g" % (inS[k] * f), "hits", inS[k], "event draws outside survivors (must be 0):", ev_miss[k])
