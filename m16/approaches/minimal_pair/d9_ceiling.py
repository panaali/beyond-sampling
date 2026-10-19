"""T389 ceiling (DEV2 only): oracle Neyman gain of finer allocation cells over the frozen sign strata, H16 T10, iid faces.
Cells: kind x sign stratum (frozen) | + #positives k | + tie positions (a,b) | + k x (a,b) | + k x (a,b) x u-quartile.
Oracle = per-cell sd known (no pilot cost). Upper bound for ANY predictor that only chooses node counts per cell."""
import sys
import numpy as np
import core as C, estimator as E, hom, walk as W
D = np.load("../../../data/dev2/weights.npz")
idx = [int(i) for i in sys.argv[1].split(",")]; N = int(sys.argv[2])
T = 10
for i in idx:
    m = (D["Wi"][i], D["Wh"][i], D["Wo"][i]); recs = []
    for j, (kind, h) in enumerate(E.strata(T)):
        U = np.random.default_rng(100 + j + 10 * i).random((N, T))
        y, a, b, tau, wt = E.face_points_strat(T, kind, h, U)
        r = np.linalg.norm(y, axis=1)
        F, c, _ = W.walk(m, y / r[:, None], a, b, np.minimum(tau / r, 1e8), kind, cdf=hom.tcdf(T))
        recs.append((j, E.stratum_mass(T, kind, h), wt * F, c, (y > 0).sum(1), a * T + b, y[np.arange(N), a]))
    err = sum(M * F.mean() for _, M, F, _, _, _, _ in recs); p = 1 - err
    def gain(keyf):
        S = 0
        for j, M, F, c, k, ab, u in recs:
            key = keyf(k, ab, u)
            for kk in np.unique(key):
                w = key == kk; S += M * w.mean() * F[w].std() * np.sqrt(c[w].mean())
        return p * (1 - p) / S ** 2
    uq = lambda u: np.digitize(u, np.quantile(u, [.25, .5, .75]))
    g0 = gain(lambda k, ab, u: np.zeros(len(k), int))
    out = {"k": gain(lambda k, ab, u: k), "ab": gain(lambda k, ab, u: ab), "k x ab": gain(lambda k, ab, u: k * 100 + ab),
           "k x ab x u4": gain(lambda k, ab, u: (k * 100 + ab) * 4 + uq(u))}
    cbar = np.mean([c.mean() for *_, c, _, _, _ in [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in recs]])
    print(f"DEV2[{i}] acc {p:.4f}: iid gain sign strata {g0:.2f}x; oracle finer / sign: " +
          ", ".join(f"{k} {v/g0:.2f}x" for k, v in out.items()) + f"  (mean c {cbar:.1f})", flush=True)
