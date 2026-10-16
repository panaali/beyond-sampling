import sys
import numpy as np
from scipy.stats import qmc, norm
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/tie_lines")
from common import Wi, Wh, Wo
from tielines import correct
from rqmc_swap import swapmin
M = (Wi, Wh, Wo)
TRUTH = 0.95317945
R = 64
for n in [4096, 16384]:
    res = {k: [] for k in ["raw_iid", "raw_rqmc", "swap_iid", "swap_rqmc"]}
    for s in range(R):
        xq = norm.ppf(qmc.Sobol(10, scramble=True, seed=1000 + s).random(n))
        xi = np.random.default_rng(5000 + s).standard_normal((n, 10))
        res["raw_iid"].append(correct(M, xi).mean()); res["raw_rqmc"].append(correct(M, xq).mean())
        res["swap_iid"].append(swapmin(xi).mean()); res["swap_rqmc"].append(swapmin(xq).mean())
    ref = TRUTH * (1 - TRUTH) / n
    for k, v in res.items():
        v = np.array(v); cost = n if k.startswith("raw") else 2 * n
        mse = np.mean((v - TRUTH) ** 2)
        print(n, k, "mean %.5f mse %.3e  compute-adj gain vs sampling %.2f" % (v.mean(), mse, (TRUTH * (1 - TRUTH) / cost) / mse))
