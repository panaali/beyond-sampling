"""Steps 1b and 2: per sampled cell of the config x dyadic-box partition, the sampled range of g (r0, zeroth-order
oracle), the smallest sampled range of g minus an affine function (r1, first-order oracle, LP) and the smooth flag
(one combinatorial signature for all lines in the cell). Saves cells_<model>.npz."""
import sys, time
import numpy as np
from scipy.optimize import linprog
import plwalk as P

LEVELS = (0, 6, 12, 18, 24, 36)
NCELL, NPT = 400, 64


def r1_lp(x, g):
    """min over affine l of max(g - l) - min(g - l) on the points (variables beta (9), up, lo)."""
    n, k = x.shape
    c = np.r_[np.zeros(k), 1.0, -1.0]
    A = np.block([[-x, -np.ones((n, 1)), np.zeros((n, 1))], [x, np.zeros((n, 1)), np.ones((n, 1))]])
    b = np.r_[-g, g]
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * (k + 2), method="highs")
    assert res.status == 0, res.message
    return res.fun


for name in sys.argv[1:]:
    m = P.load(name); t0 = time.time(); out = {}
    for j, (kind, h) in enumerate(P.STRATA):
        ncfg = len(P.C.configs(P.T, kind))
        for k in LEVELS:
            rng = np.random.default_rng([451, j, k])
            nsp = np.array([sum(1 for i in range(k) if i % 9 == c) for c in range(9)])
            ci = rng.integers(0, ncfg, NCELL)
            box = rng.integers(0, 2 ** nsp, (NCELL, 9))
            loc = rng.random((NCELL, NPT, 9))
            U = np.empty((NCELL, NPT, 10))
            U[:, :, 0] = ((ci + 0.5) / ncfg)[:, None]
            U[:, :, 1:] = (box[:, None, :] + loc) / 2.0 ** nsp
            o = P.run(m, j, U.reshape(-1, 10), want_sig=True)
            g = o["g"].reshape(NCELL, NPT); sig = o["sig"].reshape(NCELL, NPT)
            r0 = np.ptp(g, axis=1)
            r1 = np.array([r1_lp(loc[c], g[c]) if r0[c] > 0 else 0.0 for c in range(NCELL)])
            smooth = np.array([len(set(sig[c].tolist())) == 1 for c in range(NCELL)])
            out[f"r0_{j}_{k}"] = r0; out[f"r1_{j}_{k}"] = r1; out[f"sm_{j}_{k}"] = smooth
            out[f"M_{j}"] = P.E.stratum_mass(P.T, kind, h)
            print(f"{name} s{j} k{k}: r0 {r0.mean():.4f} r1 {r1.mean():.4f} smooth {smooth.mean():.3f} "
                  f"r1|smooth med {np.median(r1[smooth]) if smooth.any() else np.nan:.4g} ({time.time()-t0:.0f}s)",
                  flush=True)
    np.savez(f"cells_{name}.npz", **out)
