"""Does the own-model V2 formula capture the soft tail? Fit on 1M (seed 269100+i), compare predicted vs observed
error mass (pp) on 1M held-out (seed 269400+i) in the non-blind bins min(g12,g23) in [0,.05), [.05,.1), [.1,.2), >=.2."""
import json
import numpy as np
from scipy.special import expit
from gfcommon import MODELS, make
from phase1 import design, logistic

out = {}
for i, name in enumerate(MODELS):
    err, V = make(MODELS[name], 1_000_000, 269100 + i)
    m = V["M2"] > 0
    w = logistic(design("V2", {k: v[m] for k, v in V.items()}), err[m].astype(float))
    e2, V2 = make(MODELS[name], 1_000_000, 269400 + i)
    m2 = V2["M2"] > 0
    f = np.zeros(len(e2))
    f[m2] = expit(design("V2", {k: v[m2] for k, v in V2.items()}) @ w)
    g = np.minimum(V2["M1"] - V2["M2"], V2["M2"] - V2["M3"])
    r = {}
    for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, np.inf)):
        b = m2 & (g >= lo) & (g < hi)
        r[f"[{lo},{hi})"] = {"observed_pp": float(100 * (e2 & b).mean()), "predicted_pp": float(100 * (f * b).mean())}
    out[name] = r
    print(name, " ".join(f"{k}: obs {v['observed_pp']:.3f} pred {v['predicted_pp']:.3f}" for k, v in r.items()), flush=True)
json.dump(out, open("results/tailcheck.json", "w"), indent=1)
