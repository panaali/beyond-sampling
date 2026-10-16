"""DIAGNOSIS: is the signed network-tie offset delta(y) (degree-1 homogeneous) close to LINEAR in the face
coordinates (u, xc, rest) within a position triple?  R^2 of no-intercept regression per triple, pooled."""
import sys
import numpy as np
from lp import *
name = sys.argv[1] if len(sys.argv) > 1 else "orig"
model = get_model(name); rng = np.random.default_rng(2)
ncombo, per = 40, 400
idx = rng.choice(360, ncombo, replace=False)
r2s, r2abs, cv = [], [], []
for j in idx:
    combo, u, xc, rest = rand23(per, rng, np.repeat(COMBO[j][None], per, 0))
    f = build23(combo, u, xc, rest)
    dl, D0, cells = delta_ab(model, f)
    dl = np.clip(dl, -f["hi"], f["hi"])
    X = np.column_stack([u, xc, rest])
    coef, *_ = np.linalg.lstsq(X, dl, rcond=None)
    r2 = 1 - ((dl - X @ coef) ** 2).mean() / dl.var()
    # robust: Huber-ish via trimming 5% largest residuals
    res = dl - X @ coef; keep = np.abs(res) < np.quantile(np.abs(res), 0.95)
    coef2, *_ = np.linalg.lstsq(X[keep], dl[keep], rcond=None)
    pred = X @ coef2
    r2s.append(r2)
    r2abs.append(np.corrcoef(np.abs(dl), np.abs(pred))[0, 1])
    cv.append(np.abs(dl).std() / np.abs(dl).mean())
    print(COMBO[j], "R2 %.3f  corr|.| %.3f  mean|d| %.4f  sd(d) %.4f mean d %.4f cells %.1f" % (r2, r2abs[-1], np.abs(dl).mean(), dl.std(), dl.mean(), cells.mean()), flush=True)
print("median R2", np.median(r2s), "median corr|d|", np.median(r2abs))
