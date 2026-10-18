"""T364 DIAGNOSTIC (sampling): for conditional samples given g23 > t0 (truthc sampler), how many hits (pick == p3) keep
the hit when (a) the lows are set to the R1W closure (v4 at the latest free position, 6 lows at mu(v4)), (b) only the 6
lows are set to mu(v4) with v4 kept in place, (c) v1 at E[v1 | v2]? Locates which transverse term loses deep islands.
usage: python diag_deep.py model:t0:N ... -> results/diag_deep.json"""
import json
import os
import sys

import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "closure_floor"))
from deep import models  # noqa: E402
from truthc import sampler  # noqa: E402
from oracle import closure_point, tmean_sd  # noqa: E402


def pick(m, x):
    Wi, Wh, Wo = m; h = np.zeros((len(x), 16))
    for t in range(10):
        h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
    return (h @ Wo.T).argmax(1)


out = {}; ms = models()
for spec in sys.argv[1:]:
    nm, t0, N = spec.split(":"); t0 = float(t0); N = int(float(N)); m = tuple(np.asarray(w, float) for w in ms[nm])
    draw = sampler(t0); rng = np.random.default_rng(364_900); c = dict(hits=0, R1W=0, lows_mean=0, v1_mean=0, n=0); g12h = []
    while c["n"] < N:
        k = min(500_000, N - c["n"]); x, p3 = draw(rng, k); r = np.arange(k)
        hit = pick(m, x) == p3; c["hits"] += int(hit.sum())
        xb, *_ = closure_point(x, "R1W"); c["R1W"] += int((hit & (pick(m, xb) == p3)).sum())
        o = np.argsort(-x, 1); v4 = x[r, o[:, 3]]; mu4, _ = tmean_sd(v4)
        xl = x.copy(); lowpos = o[:, 4:]; xl[r[:, None], lowpos] = mu4[:, None]
        c["lows_mean"] += int((hit & (pick(m, xl) == p3)).sum())
        v2 = x[r, o[:, 1]]; xv = x.copy(); xv[r, o[:, 0]] = np.exp(-v2 ** 2 / 2) / np.sqrt(2 * np.pi) / ndtr(-v2)        # E[v1 | v1 > v2] (inverse Mills)
        c["v1_mean"] += int((hit & (pick(m, xv) == p3)).sum())
        v1 = x[r, o[:, 0]]; g12h.append((v1 - v2)[hit])
        c["n"] += k
    g = np.concatenate(g12h); c["g12_hits_quantiles_10_50_90"] = [float(q) for q in np.percentile(g, [10, 50, 90])] if len(g) else []
    c["frac_hits_g12_lt_0.1"] = float((g < 0.1).mean()) if len(g) else None
    out[spec] = c
    print(spec, c, {k: f"{c[k] / max(c['hits'], 1):.3f}" for k in ("R1W", "lows_mean", "v1_mean")}, flush=True)
json.dump(out, open(os.path.join(HERE, "results", "diag_deep.json"), "w"), indent=1)
