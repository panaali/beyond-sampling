"""Ceiling tests across models. Reductions (low = inputs of rank > k):
 V1: lows keep their order, values = quantile of beta-mean order stat below c=k-th value
 V2: all lows set to one value: conditional mean of N(0,1) below c  (symbolic part = top-k positions only)
 V3: all lows set to conditional median below c"""
import sys; sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/bandwidth")
import numpy as np
from arc_eval import forward, Wi0, Wh0, Wo0, noisy_models
from scipy.special import ndtr, ndtri
from scipy.stats import norm
def zoo(sd):
    from alg_zoo import zoo_2nd_argmax
    m = zoo_2nd_argmax(16, 10, seed=sd); f = lambda t: t.detach().numpy().astype(np.float64)
    return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)
models = {"orig": (Wi0, Wh0, Wo0)}
for sd in range(5): models[f"seed{sd}"] = zoo(sd)
for s in (0.0025, 0.005):
    for i, m in enumerate(noisy_models(2, s, seed=0)): models[f"s{s}_{i}"] = m
ks = [int(a) for a in sys.argv[1:]] or [4]
N = 600_000
for name, M in models.items():
    rng = np.random.default_rng(7); res = {}
    tot = {}
    for ch in range(N // 200_000):
        x = rng.standard_normal((200_000, 10)); o = np.argsort(-x, 1); tgt = o[:, 1]
        ok = forward(M, x).argmax(1) == tgt; tot["true"] = tot.get("true", 0) + ok.sum()
        for k in ks:
            c = np.take_along_axis(x, o[:, k-1:k], 1)[:, 0]; n = 10 - k; j = np.arange(1, n + 1)
            v1 = ndtri(ndtr(c)[:, None] * (n + 1 - j)[None] / (n + 1))
            v2 = np.repeat((-norm.pdf(c) / ndtr(c))[:, None], n, 1)
            v3 = np.repeat(ndtri(ndtr(c) / 2)[:, None], n, 1)
            for nm, v in (("V1", v1), ("V2", v2), ("V3", v3)):
                xr = x.copy(); np.put_along_axis(xr, o[:, k:], v, 1)
                okr = forward(M, xr).argmax(1) == tgt
                key = f"{nm}k{k}"; tot[key] = tot.get(key, 0) + okr.sum()
    t = tot.pop("true") / N
    print(f"{name:10s} true {t:.4f} | " + " ".join(f"{kk} {100*(v/N-t):+.2f}" for kk, v in tot.items()), flush=True)
