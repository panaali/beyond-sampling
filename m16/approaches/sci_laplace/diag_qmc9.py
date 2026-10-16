"""DIAGNOSIS: per position triple, 9-D face integral (u, xc, 7 rest) with radial integration (tube's F_faces):
variance of scrambled-Sobol vs MC estimates at small m (the per-triple sizes a 16k-64k budget allows)."""
import sys
import numpy as np
from scipy.stats import qmc
from lp import *
from lp import _trunc_below, _trunc_above, _inv
from tube import F_faces
model = get_model(sys.argv[1] if len(sys.argv) > 1 else "orig"); rng = np.random.default_rng(17)
for m in (16, 64):
    ratios = []
    for j in rng.choice(360, 10, replace=False):
        def EF(Uu):
            u = _inv(CDF23, Uu[:, 0]); xc = _trunc_above(u, Uu[:, 1]); rest = _trunc_below(u[:, None], Uu[:, 2:9])
            f = build23(np.repeat(COMBO[j][None], m, 0), u, xc, rest)
            f["tpos"], f["tneg"] = f["tpos"], f["tneg"]
            F, _, _ = F_faces(model, f, 400, radial=True); return F.mean()
        mc = [EF(rng.random((m, 9))) for _ in range(24)]
        q = [EF(qmc.Sobol(9, scramble=True, seed=int(rng.integers(1 << 30))).random(m)) for _ in range(24)]
        ratios.append(np.var(mc) / np.var(q))
    print("m=%d  var ratio MC/QMC per triple:" % m, np.round(ratios, 2), "geo-mean %.2f" % np.exp(np.mean(np.log(ratios))), flush=True)
