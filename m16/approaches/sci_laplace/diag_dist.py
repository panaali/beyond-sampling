import sys
import numpy as np
from lp import *
model = get_model("orig"); rng = np.random.default_rng(3)
combo, u, xc, rest = rand23(20000, rng)
f = build23(combo, u, xc, rest)
dl, D0, cells = delta_ab(model, f)
hit_end = np.abs(np.abs(dl) - f["hi"]) < 1e-9
k = np.abs(dl) / np.linalg.norm(f["x0"], axis=1)
print("frac reaching segment end", hit_end.mean())
print("quantiles |delta|/|x|", np.quantile(k, [.1, .25, .5, .75, .9, .95, .99]))
print("quantiles hi/|x|", np.quantile(f["hi"] / np.linalg.norm(f["x0"], axis=1), [.1, .5, .9]))
e1, e2, c = F_sides(model, f)
F = e1 + e2
Fd = W(0, np.minimum(np.abs(dl), f["hi"]))   # crude: band mass if error set = [0, |delta|] (ignores lo cut)
print("mean F", F.mean(), "mean W(0,|delta|)", Fd.mean(), "corr", np.corrcoef(F, Fd)[0, 1])
print("share of F from points with |delta| reaching end", F[hit_end].sum() / F.sum())
# share of F with a third class involved
np.savez("data/orig_faces.npz", combo=combo, u=u, xc=xc, rest=rest, dl=dl, D0=D0, F=F, hi=f["hi"])
