"""DIAGNOSIS: for fixed top-3 configurations, is the 7-D rest-input integral E_rest[F] integrated faster by
randomly shifted QMC (scrambled Sobol) than by MC? (Tests tube's claim 'QMC gives no gain' in the sub-integral
that carries the irregularity; the outer (combo,u,xc) integral is low-dimensional.)"""
import numpy as np
from scipy.stats import qmc
from lp import *
from lp import _trunc_below
model = get_model("orig"); rng = np.random.default_rng(13)
ncfg, n, reps = 12, 256, 16
ratios = []
for j in range(ncfg):
    combo, u, xc, rest = rand23(1, rng)
    C = np.repeat(combo, n, 0); U_ = np.repeat(u, n); X = np.repeat(xc, n)
    def EF(Uu):
        f = build23(C, U_, X, _trunc_below(U_[:, None], Uu))
        e1, e2, _ = F_sides(model, f); return (e1 + e2).mean()
    mc = [EF(rng.random((n, 7))) for _ in range(reps)]
    q = [EF(qmc.Sobol(7, scramble=True, seed=int(rng.integers(1 << 30))).random(n)) for _ in range(reps)]
    ratios.append(np.var(mc) / np.var(q))
    print(combo[0], "mean %.4f  var MC/var QMC = %.2f" % (np.mean(mc), ratios[-1]), flush=True)
print("median variance ratio", np.median(ratios), "geo-mean", np.exp(np.mean(np.log(ratios))))
