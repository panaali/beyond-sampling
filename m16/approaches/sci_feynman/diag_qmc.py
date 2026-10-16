"""Diagnosis: per position-config (a,b,p1), does a scrambled Sobol design over the face coordinates
(v, o1, 7 bath inputs in time order) estimate E[F | config] much better than iid points?
Random scrambles are used only to measure the error spread. usage: python diag_qmc.py <model> [nconf] [reps]"""
import sys, time
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
import wtool as W

nm = sys.argv[1] if len(sys.argv) > 1 else "orig"
nconf = int(sys.argv[2]) if len(sys.argv) > 2 else 8; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
M = W.get_model(nm)
rng = np.random.default_rng(5)


def face(U, a, b, p1):
    """U (n,9) in [0,1) -> face points for config (a,b,p1)."""
    n = len(U)
    v = W.inv_ybar(U[:, 0])
    hi = ndtri(np.clip(ndtr(v) + U[:, 1] * (1 - ndtr(v)), 1e-300, 1 - 1e-16))
    lo = ndtri(np.clip(U[:, 2:] * ndtr(v)[:, None], 1e-300, 1 - 1e-16))
    y = np.empty((n, 10)); y[:, a] = v; y[:, b] = v; y[:, p1] = hi
    bath = [k for k in range(10) if k not in (a, b, p1)]
    y[:, bath] = lo
    o2 = lo.max(1)
    tau = W.SQ2 * np.minimum(hi - v, v - o2)
    return y, tau


def Fm(U, a, b, p1):
    y, tau = face(U, a, b, p1); n = len(y)
    return W.walk(M, y, np.full(n, a), np.full(n, b), tau)["mass"]


confs = []
while len(confs) < nconf:
    a, b, p = rng.choice(10, 3, replace=False); a, b = min(a, b), max(a, b); confs.append((a, b, p))
for (a, b, p) in confs:
    ref = Fm(rng.random((20000, 9)), a, b, p); mu = ref.mean(); se = ref.std() / np.sqrt(len(ref))
    line = f"conf {a},{b},{p1 if False else p}: mean {mu:.4f}+-{se:.4f} CV {ref.std()/mu:.2f} |"
    for N in (16, 64, 256):
        iid = [Fm(rng.random((N, 9)), a, b, p).mean() for _ in range(reps)]
        sob = [Fm(qmc.Sobol(9, scramble=True, seed=rng).random(N), a, b, p).mean() for _ in range(reps)]
        line += f" N={N}: rel sd iid {np.std(iid)/mu:.3f} sobol {np.std(sob)/mu:.3f} |"
    print(line, flush=True)
