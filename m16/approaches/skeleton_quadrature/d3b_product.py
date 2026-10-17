"""D3b: fully DETERMINISTIC per-triple product Gauss-Legendre rules in the cube coordinates (U1, U2, U3), the same
rule for all 720 triples (structure shared: one value rule). Error vs the sampling truth; q at each node from M iid
inner draws (unbiased stand-in for exact q; its noise is subtracted). Also a 'midpoint-in-U1' variant is implied
by k1 = 1. usage: python m16/approaches/skeleton_quadrature/d3b_product.py MODEL [M] [rules like 1x2x4,2x2x2]
"""
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402
from d1_shape_lib import vals_from_U  # noqa: E402

name = sys.argv[1]
M = int(sys.argv[2]) if len(sys.argv) > 2 else 128
RULES = sys.argv[3].split(",") if len(sys.argv) > 3 else ["1x1x2", "1x1x4", "1x2x2", "1x1x8", "1x2x4", "2x2x2", "1x4x4", "2x2x4", "2x4x4", "1x4x8"]
model = rv.dev_model(name)
rng = np.random.default_rng(6006)
TR = np.array(list(itertools.permutations(range(10), 3)))
D = __file__.rsplit("/", 1)[0]
truth = json.load(open(f"{D}/truth_{name}.json"))
out = dict(model=name, M=M, truth=truth, rules={})
for r in RULES:
    ks = [int(a) for a in r.split("x")]
    gl = [np.polynomial.legendre.leggauss(k) for k in ks]
    pts = np.array(list(itertools.product(*[(g[0] + 1) / 2 for g in gl])))
    wts = np.prod(np.array(list(itertools.product(*[g[1] / 2 for g in gl]))), 1)
    k = len(pts)
    tri = np.repeat(np.arange(720), k)
    U = np.tile(pts, (720, 1))
    w = np.tile(wts, 720) / 720
    pos = TR[tri]; val = vals_from_U(U); free = rv.free_positions(pos)
    q = np.empty(len(tri))
    for a in range(0, len(tri), 64):
        b = slice(a, a + 64)
        low = rv.low_draws(val[b, 2], M, 7, rng)
        x = rv.assemble(pos[b], val[b], free[b], low)
        q[b] = (rv.run(model, x.reshape(-1, rv.T))[0].reshape(-1, M, 10).argmax(2) == pos[b, 1:2]).mean(1)
    est = float(w @ q); nv = float(np.sum(w ** 2 * q * (1 - q)) / (M - 1))
    err = est - truth["p"]
    out["rules"][r] = dict(N=720 * k, est=est, err_pp=100 * err, sd_noise_pp=100 * np.sqrt(nv + truth["se"] ** 2))
    print(r, out["rules"][r], flush=True)
    json.dump(out, open(f"{D}/d3b_{name}.json", "w"), indent=1)
