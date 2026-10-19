"""Pick the target t0 (INTERFACE.md): approximate P(E1(t0)) on orig by the conditional law of v3 along x_p3
(T362's estimator) with GRID quadrature (NG points on (v4, v2 - t0)) instead of the exact walk. Approximate; only for
choosing t0. The formal truth for the chosen t0 is T362's cmc.py (exact walk, branch claude/t286-cheap-frontier)."""
import sys

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, "../..")
import common  # noqa: E402

T0S = (1.5, 2.0, 2.25, 2.5, 2.75, 3.0)
NG = 200


def main(ndraw=int(4e7), seed=398, batch=1_000_000):
    rng = np.random.default_rng(seed); acc = {t: [] for t in T0S}; n = 0
    while n < ndraw:
        x = rng.standard_normal((batch, 10)); n += batch
        o = np.argsort(-x, 1); r = np.arange(batch); v = x[r[:, None], o]
        ok = np.flatnonzero((v[:, 1] - v[:, 3]) > min(T0S))
        for t0 in T0S:
            ii = ok[(v[ok, 1] - t0) > v[ok, 3]]
            if not len(ii):
                continue
            v2, v4, p3 = v[ii, 1], v[ii, 3], o[ii, 2]
            for c in range(0, len(ii), 1000):
                sl = slice(c, c + 1000); m = len(ii[sl])
                u = v4[sl, None] + (np.arange(NG)[None] + .5) / NG * (v2[sl] - t0 - v4[sl])[:, None]
                X = np.repeat(x[ii[sl]], NG, 0); X[np.arange(m * NG), np.repeat(p3[sl], NG)] = u.ravel()
                pick = ((common.run(X)[:, -1] @ common.Wo.T).argmax(1) == np.repeat(p3[sl], NG)).reshape(m, NG)
                w = np.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi) * ((v2[sl] - t0 - v4[sl]) / NG)[:, None]
                z = (w * pick).sum(1) / (ndtr(v2[sl]) - ndtr(v4[sl]))
                acc[t0] += list(z[z > 0])
        print(n, {t: len(a) for t, a in acc.items()}, flush=True)
    for t0 in T0S:
        a = np.array(acc[t0]); m = a.sum() / n; se = np.sqrt((a ** 2).sum() / n - m ** 2) / np.sqrt(n)
        print(f"t0 {t0}: P ~ {m:.3g} +- {se:.2g} (nonzero lines {len(a)})", flush=True)


if __name__ == "__main__":
    main(int(float(sys.argv[1])) if len(sys.argv) > 1 else int(4e7))
