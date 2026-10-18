"""T307 (1) DIAGNOSTIC: oracle floors of closure c-node / c-B5 / c-true on 10 DEV2 models, paired.
usage: python b5_oracle.py [n] -> results/b5_oracle.json"""
import json
import os
import sys

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r0_integral")); sys.path.insert(0, os.path.join(HERE, "..", "r1_walk"))
import r0  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
ms = dict(r0.dev2()); names = [list(ms)[i] for i in r0.DEV_IDX]; tr = r0.truths()["truths"]
QN = 64
_gx, _gw = np.polynomial.legendre.leggauss(QN)


def v4_law_nodes(v3):
    """Diagnostic quadrature of v4 | v3 (density 7 phi Phi^6 / Phi(v3)^7 on v < v3): GL in u = Phi(v)^7/Phi(v3)^7."""
    u = (_gx + 1) / 2; w = _gw / 2                                   # uniform in the CDF -> exact law
    v = r0.norm.ppf(u[None] ** (1 / 7) * ndtr(v3)[:, None]) if hasattr(r0, "norm") else norm.ppf(u[None] ** (1 / 7) * ndtr(v3)[:, None])
    return v, w


def cond_mean_v4(v3):
    v, w = v4_law_nodes(v3)
    return (v * w[None]).sum(1)


def run(model, x):
    Wi, Wh, Wo = model; h = np.zeros((len(x), 16))
    for t in range(10):
        h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
    return h @ Wo.T


out = {}
for k, nm in enumerate(names):
    model = ms[nm]; rng = np.random.default_rng(307_000 + k)
    x = rng.standard_normal((n, 10)); o = np.argsort(-x, 1); v = np.take_along_axis(x, o, 1); r = np.arange(n)
    y = o[:, 1]
    free = np.ones_like(x, bool); free[r[:, None], o[:, :3]] = False
    p4 = np.where(free, np.arange(10)[None], -1).max(1)               # latest free position
    v4n = cond_mean_v4(v[:, 2]); mun = r0.mu(v4n)
    def build(v4val):
        xc = np.repeat(mun[:, None], 10, 1)
        for j in range(3):
            xc[r, o[:, j]] = v[:, j]
        xc[r, p4] = v4val
        return xc
    acc = {}
    acc["c_node"] = (run(model, build(v4n)).argmax(1) == y).mean()
    acc["c_true"] = (run(model, build(v[:, 3])).argmax(1) == y).mean()     # true v4 value, declared position
    # B5: where p4 == 9, integrate x9 = v4 over its exact law (lows at mu(node)); elsewhere = c_node
    V, Wq = v4_law_nodes(v[:, 2]); pc = np.zeros(n)
    for q in range(QN):
        pc += Wq[q] * (run(model, build(V[:, q])).argmax(1) == y)
    node_ok = run(model, build(v4n)).argmax(1) == y
    acc["c_B5"] = np.where(p4 == 9, pc, node_ok).mean()
    out[nm] = {kk: float(vv - tr[nm]["acc"]) for kk, vv in acc.items()}
    print(nm, {kk: round(100 * vv, 3) for kk, vv in out[nm].items()}, flush=True)
fl = {c: float(100 * np.sqrt(np.mean([out[m][c] ** 2 for m in out]))) for c in ("c_node", "c_B5", "c_true")}
print("floors (RMS pp):", fl, "reduction B5 vs node: %.1f%%" % (100 * (1 - fl["c_B5"] / fl["c_node"])))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(dict(per_model=out, floors_pp=fl, n=n), open(os.path.join(HERE, "results", "b5_oracle.json"), "w"), indent=1)
