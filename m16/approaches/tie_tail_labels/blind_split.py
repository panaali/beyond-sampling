"""T218 stage 2: how much of the face integrand's second moment sits in the blind region (2nd-largest u <= 0), and is
the remainder (u > 0) light-tailed? Rule seed 700, 2^14 faces per kind, all 8 models. -> results/blind_split.json"""
import json
import os

import numpy as np

import core as C
from tail import HERE, get_model

out = {}
for name in ["z6s0", "z6s1", "z6s2", "z6s3", "z6s4", "orig", "s005", "s010"]:
    model = get_model(name); T = model[2].shape[0]; _, Z23, Z12 = C.tables(T); n = 2 ** 14
    g, u = [], []
    for kind, Z in (("23", Z23), ("12", Z12)):
        U = C.sobol(T, n, 700 + (0 if kind == "23" else 7))
        for i in range(0, n, 4096):
            y, a, b, tau = C.face_points(T, kind, U[i:i + 4096])
            m, _, _ = C.walk_mass(model, y, a, b, tau, kind)
            g.append(Z * m); u.append(y[np.arange(len(y)), a])
    g = np.concatenate(g); u = np.concatenate(u); g2 = g ** 2; bl = u <= 0

    def top(v, f):
        v = np.sort(v)[::-1]; return float(v[:int(np.ceil(f * len(v)))].sum() / v.sum())
    r = dict(err=2 * g.mean(), face_frac_blind=float(bl.mean()), err_share_blind=float(g[bl].sum() / g.sum()),
             g2_share_blind=float(g2[bl].sum() / g2.sum()), top5_all=top(g2, .05), top1_all=top(g2, .01),
             top5_nonblind=top(g2[~bl], .05), top1_nonblind=top(g2[~bl], .01), top5_blind=top(g2[bl], .05),
             cv_all=float(g.std() / g.mean()), cv_nonblind=float(g[~bl].std() / g[~bl].mean()),
             cv_blind=float(g[bl].std() / g[bl].mean()))
    out[name] = r; print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "results", "blind_split.json"), "w"), indent=1)
