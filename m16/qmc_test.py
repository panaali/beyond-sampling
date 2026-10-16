"""Convergence of RMSE vs compute: plain sampling vs (randomised) quasi-Monte Carlo, with and without exact line integration."""
import sys
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
sys.path.insert(0, "m16")
from common import Wi, Wh, Wo, run, target
from lineint import line_accuracy

TRUTH = float(open("m16/truth.txt").read().split()[0])
M = (Wi, Wh, Wo)


def lines_estimate(N, seed, quasi):
    if quasi:
        u = qmc.Sobol(10, scramble=True, seed=seed).random(N)
    else:
        u = np.random.default_rng(seed).random((N, 10))
    j = np.minimum((u[:, 0] * 10).astype(int), 9)
    z = ndtri(np.clip(u[:, 1:], 1e-12, 1 - 1e-12))
    ps, cost = [], 0
    for k in range(N):
        x = np.insert(z[k], j[k], 0.0)
        p, n = line_accuracy(M, x, int(j[k])); ps.append(p); cost += n
    return float(np.mean(ps)), cost


def raw_estimate(N, seed, quasi):
    if quasi:
        u = qmc.Sobol(10, scramble=True, seed=seed).random(N)
    else:
        u = np.random.default_rng(seed).random((N, 10))
    x = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    return float(((run(x)[:, -1] @ Wo.T).argmax(1) == target(x)).mean()), N


if __name__ == "__main__":
    which = sys.argv[1]
    Ns = [int(v) for v in sys.argv[2].split(",")]
    reps = int(sys.argv[3])
    fn = lines_estimate if which == "lines" else raw_estimate
    for N in Ns:
        for quasi in (False, True):
            ests, costs = zip(*[fn(N, 100 + r, quasi) for r in range(reps)])
            mse = np.mean((np.array(ests) - TRUTH) ** 2); C = np.mean(costs)
            print(f"{which:5s} {'QMC' if quasi else 'MC ':3s} N={N:8d} compute={C:10.0f}  RMSE={np.sqrt(mse):.6f}  "
                  f"sampling RMSE at same compute={np.sqrt(TRUTH*(1-TRUTH)/C):.6f}  ratio={np.sqrt(mse)/np.sqrt(TRUTH*(1-TRUTH)/C):.2f}", flush=True)
