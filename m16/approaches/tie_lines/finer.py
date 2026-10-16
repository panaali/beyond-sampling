"""Do finer model-free strata (joint g12 x g23 bins, x9's rank, v2 magnitude) beat stratified.py's 7 strata?
Allocation fitted on half A of the original-model sample; variance evaluated on half B (and on noisy copies).
Reports gain = acc(1-acc) / Var_design*cost for plain terms and swap terms (swap where it pays on half A)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import estimators as E
from stratified import stratum
from arc_eval import noisy_models
from common import Wi, Wh, Wo
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
rng = np.random.default_rng(7)
x = rng.standard_normal((2 * N, 10))
o = np.argsort(-x, 1); xs = np.take_along_axis(x, o, 1)
g12 = (xs[:, 0] - xs[:, 1]) / np.abs(xs[:, 0]); g23 = (xs[:, 1] - xs[:, 2]) / np.maximum(np.abs(xs[:, 1]), 1e-12)
blind = xs[:, 1] <= 0
r9 = (o == 9).argmax(1); r9c = np.minimum(r9, 3)        # rank of x9: 0,1,2, >=3
ED = np.array([0, .0025, .005, .01, .02, .035, .05, .075, .1, .15, .2, .3, .5, np.inf])
b12 = np.searchsorted(ED, g12, side="right") - 1; b23 = np.searchsorted(ED, g23, side="right") - 1
v2b = np.searchsorted([0.3, 0.7, 1.2], xs[:, 1])
SCHEMES = {
    "strat7": stratum(x),
    "g12xg23": np.where(blind, 0, 1 + b12 * 20 + b23),
    "g12xg23xr9": np.where(blind, 0, 1 + (b12 * 20 + b23) * 4 + r9c),
    "g12xg23xv2": np.where(blind, 0, 1 + (b12 * 20 + b23) * 4 + v2b),
    "min_gap13": np.where(blind, 0, 1 + np.searchsorted(ED, np.minimum(g12, g23), side="right")),
}
A, B = slice(0, N), slice(N, 2 * N)
y = E.swap_minpair(x)[0]


def evaluate(m, tag):
    g = E.g_of(m, x); gs = E.g_of(m, y); sw = (g + gs) / 2
    acc = g[B].mean(); ref = acc * (1 - acc); out = {}
    for name, h in SCHEMES.items():
        ids, inv = np.unique(h, return_inverse=True); K = len(ids)
        def stats(v, sl):
            n = np.bincount(inv[sl], minlength=K).astype(float)
            s1 = np.bincount(inv[sl], v[sl], K); s2 = np.bincount(inv[sl], v[sl] ** 2, K)
            mu = s1 / np.maximum(n, 1); return n / n.sum(), np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mu ** 2, 0))
        PA, spA = stats(g, A); _, ssA = stats(sw, A)
        PB, spB = stats(g, B); _, ssB = stats(sw, B)
        P = np.bincount(inv, minlength=K) / len(inv)   # model-free stratum masses (full sample)
        for kind in ("plain", "swap"):
            if tag == "orig":
                if kind == "plain":
                    use = np.zeros(K, bool)
                else:
                    use = ssA * np.sqrt(2) < spA
                sigA = np.where(use, ssA, spA) + 1e-3; cost = np.where(use, 2.0, 1.0)
                a = P * sigA / np.sqrt(cost); a = a / (a * cost).sum()
                ALLOC[(name, kind)] = (a, use, cost)
            a, use, cost = ALLOC[(name, kind)]       # allocation from the ORIGINAL model only
            sigB = np.where(use, ssB, spB)
            var = (P ** 2 * sigB ** 2 / np.maximum(a, 1e-12)).sum()   # per unit budget (sum a*cost = 1)
            out[f"{name}/{kind}"] = round(ref / var, 3)
    print(tag, json.dumps(out), flush=True)
    return out


ALLOC = {}
res = {"orig": evaluate((Wi, Wh, Wo), "orig")}
if len(sys.argv) > 2:
    for s in (0.0025, 0.005):
        for k, m in enumerate(noisy_models(6, s)[:int(sys.argv[2])]):
            res[f"s{s}_{k}"] = evaluate(m, f"s{s}_{k}")
