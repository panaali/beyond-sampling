"""DIAGNOSIS: given the top-3 configuration (positions a<b, c and values u, xc), is the signed network tie
offset delta an ADDITIVE function of the 7 rest inputs (so its law is a convolution, computable by
generating functions)?  Backfitting additive model with binned components; R^2 for delta and for F."""
import sys
import numpy as np
from lp import *
from lp import _trunc_below
name = sys.argv[1] if len(sys.argv) > 1 else "orig"
model = get_model(name); rng = np.random.default_rng(5)
ncfg, per, nb = 30, 300, 6
out = []
for j in range(ncfg):
    combo, u, xc, rest = rand23(1, rng)
    combo = np.repeat(combo, per, 0); u = np.repeat(u, per); xc = np.repeat(xc, per)
    rest = _trunc_below(u[:, None], rng.random((per, 7)))
    f = build23(combo, u, xc, rest)
    dl, D0, _ = delta_ab(model, f)
    e1, e2, _ = F_sides(model, f); F = e1 + e2
    # quantile bins per variable (bins depend only on the uniform used to generate => equal mass)
    U = ndtr(rest) / ndtr(u[0])
    B = np.minimum((U * nb).astype(int), nb - 1)
    def backfit(y, it=30):
        comp = np.zeros((7, nb)); mu = y.mean()
        for _ in range(it):
            for t in range(7):
                r = y - mu - sum(comp[s][B[:, s]] for s in range(7) if s != t)
                comp[t] = [r[B[:, t] == k].mean() if (B[:, t] == k).any() else 0 for k in range(nb)]
                comp[t] -= comp[t].mean()
        pred = mu + sum(comp[s][B[:, s]] for s in range(7))
        return 1 - ((y - pred) ** 2).mean() / y.var()
    # 'in-sample' with 6 bins x 7 vars = 42 params out of 300: correct for overfit with a permutation null
    r2d, r2f = backfit(dl), backfit(F)
    null = backfit(rng.permutation(dl))
    out.append((r2d, r2f, null))
    print(combo[0], "u %.2f xc %.2f  R2 additive: delta %.3f  F %.3f  (null %.3f)  sd(delta) %.4f  mean F %.4f"
          % (u[0], xc[0], r2d, r2f, null, dl.std(), F.mean()), flush=True)
o = np.array(out); print("median", np.median(o, 0))
