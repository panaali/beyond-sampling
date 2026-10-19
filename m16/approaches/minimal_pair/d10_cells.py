"""T389 development (DEV2 only): per-model per-cell statistics, cell = (sign stratum j, tie positions a,b), H16 T10.
Saves W (cell weight within stratum), mean F, sd F, mean cost -> data/minimal_pair_d10.npz."""
import sys
import numpy as np
import core as C, estimator as E, hom, walk as W
D = np.load("../../../data/dev2/weights.npz")
idx = list(range(int(sys.argv[1]), int(sys.argv[2]))); N = int(sys.argv[3]); T = 10
NC = T * T
out = np.zeros((len(idx), 5, NC, 4))       # w, mean, sd, cost
for ii, i in enumerate(idx):
    m = (D["Wi"][i], D["Wh"][i], D["Wo"][i])
    for j, (kind, h) in enumerate(E.strata(T)):
        U = np.random.default_rng(300 + j + 10 * i).random((N, T))
        y, a, b, tau, wt = E.face_points_strat(T, kind, h, U)
        r = np.linalg.norm(y, axis=1)
        F, c, _ = W.walk(m, y / r[:, None], a, b, np.minimum(tau / r, 1e8), kind, cdf=hom.tcdf(T)); F = wt * F
        ab = a * T + b
        for cell in np.unique(ab):
            w = ab == cell
            out[ii, j, cell] = (w.mean(), F[w].mean(), F[w].std(), c[w].mean())
    print("done", i, flush=True)
    np.save(f"../../../data/minimal_pair_d10_{sys.argv[1]}_{sys.argv[2]}.npy", out)
