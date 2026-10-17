"""T97 strict op recount of F2L2 (fibre.est_fibre2, LOOK = 2) and the cost law.
For each model: author's count (fibre.Ops.n), reviewer's strict count (fibre_strict: + every uncharged elementwise op and
comparison), check that estimates and author counts are bit-identical between the two copies; reviewer's own
enumeration cost (rtruth.exact instrumented: 5 ops per vertex per clip + 26 per cut clip, 25 per solid-angle triangle,
plus the row compositions) and region count. Fits log-log slopes over H = 4, 5, 6, 8 (zoo seeds).
usage: OMP_NUM_THREADS=1 python recount.py  -> recount.json"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_ladder"))
import fibre  # noqa: E402
import fibre_strict  # noqa: E402
import rtruth as R  # noqa: E402

CNT = [0]
_clip, _tri, _split = R.clip, R.tri, R.split


def clip_c(poly, nrm):
    out = _clip(poly, nrm)
    if poly:
        CNT[0] += 5 * len(poly) + (26 if (out and out is not poly) else 0)
    return out


def tri_c(a, b, c):
    CNT[0] += 25
    return _tri(a, b, c)


R.clip, R.tri = clip_c, tri_c


def enum_cost(m):
    H = len(m[0])
    CNT[0] = 0
    acc, nr, nl = R.exact(m)
    # row compositions: per start polygon A2 (H*H), per step-2 leaf A3 (H*H*2) -- charged at H*H*2 per leaf
    return acc, nr, nl, CNT[0] + 48 * H * H + nl * 2 * H * H


rows = {}
for H in (4, 5, 6, 8):
    ms = {f"seed{s}": R.load(H, 3, s)[0] for s in range(5)}
    rng = np.random.default_rng(104)
    base = ms["seed0"]
    for i in range(3):
        ms[f"abs0.05_n104_{i}"] = tuple(w + 0.05 * rng.standard_normal(w.shape) for w in base)
    for name, m in ms.items():
        acc, nr, nl, ec = enum_cost(m)
        fibre.LOOK = 2; o = fibre.Ops(); inf = {}; e1 = fibre.est_fibre2(m, o, inf)
        fibre_strict.LOOK = 2; o2 = fibre_strict.Ops(); e2 = fibre_strict.est_fibre2(m, o2, {})
        assert e1 == e2 and o.n == o2.n, (H, name, e1, e2, o.n, o2.n)
        strict = o2.n + o2.x
        rows[f"H{H}_{name}"] = dict(H=H, truth=acc, est=e1, err=e1 - acc, regions=nr, leaves=nl, enum_ops=ec,
                                    author_ops=o.n, strict_ops=strict, traces=inf["traces"], strict_over_author=strict / o.n)
        print(f"H={H} {name:16s} err {e1 - acc:+.1e} regions {nr} author {o.n} strict {strict} ({strict / o.n:.2f}x) "
              f"own-enum {ec} -> F2L2/enum {strict / ec:.1f}x  traces {inf['traces']}", flush=True)

fit = {}
Hs = np.array([4, 5, 6, 8])
for k in ("regions", "author_ops", "strict_ops", "enum_ops", "traces"):
    y = np.array([np.mean([rows[f"H{H}_seed{s}"][k] for s in range(5)]) for H in Hs])
    sl = float(np.polyfit(np.log(Hs), np.log(y), 1)[0])
    # leave-one-out slope range
    loo = [float(np.polyfit(np.log(np.delete(Hs, i)), np.log(np.delete(y, i)), 1)[0]) for i in range(4)]
    fit[k] = dict(means=y.tolist(), slope=sl, loo_min=min(loo), loo_max=max(loo))
    print(f"{k:12s} means {np.round(y).astype(int).tolist()} slope {sl:.2f} (leave-one-out {min(loo):.2f}-{max(loo):.2f})")
per_region = {int(H): float(np.mean([rows[f"H{H}_seed{s}"]["strict_ops"] / rows[f"H{H}_seed{s}"]["regions"] for s in range(5)])) for H in Hs}
print("strict ops per region", per_region)
json.dump(dict(rows=rows, fit=fit, strict_ops_per_region=per_region), open(os.path.join(HERE, "recount.json"), "w"), indent=1)
