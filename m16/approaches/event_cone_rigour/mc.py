"""Independent MC for the mass of a cell (k-step gate prefix, event optional) ∩ N ∩ {g23 >= t0}: importance
sampling N(x*, s^2 I) around the float dominating point; membership by the forward pass (common.run), not by the
constraint algebra."""
import numpy as np

import common
import rigour as R


def mc_mass(tri, gates, t0=1.5, n=200_000, seed=0, s=1.0, x=None, k=10, event=True):
    p1, p2, p3 = tri
    if x is None:
        A, b = R.cell_rows_exact(tri, gates, k, t0, event=event)
        Af = np.array([[float(v) for v in r] for r in A]); bf = np.array([float(v) for v in b])
        x = R.dominating_point(Af, bf)
    rng = np.random.default_rng(seed); tot = []; hits = 0
    g = np.asarray(gates)[:k].astype(bool)
    for _ in range(max(1, n // 50_000)):
        X = x + s * rng.standard_normal((50_000, 10))
        hs = common.run(X)
        ok = ((hs[:, 1:k + 1] > 0).reshape(len(X), -1) == g.reshape(1, -1)).all(1)
        if event:
            ok &= (hs[:, -1] @ common.Wo.T).argmax(1) == p3
        srt = np.argsort(-X, 1)
        ok &= (srt[:, 0] == p1) & (srt[:, 1] == p2) & (srt[:, 2] == p3)
        g23 = X[:, p2] - X[:, p3]
        ok &= (g23 >= t0) & (np.abs(X).max(1) <= R.NORM * g23)
        lw = -0.5 * (X ** 2).sum(1) + 0.5 * ((X - x) ** 2).sum(1) / s ** 2 + 10 * np.log(s)
        w = np.where(ok, np.exp(lw), 0.0); tot.append(w); hits += int(ok.sum())
    w = np.concatenate(tot)
    return w.mean(), w.std() / np.sqrt(len(w)), hits
