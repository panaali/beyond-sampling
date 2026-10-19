"""T415 identity and speed tests of geom_fast against the reference geometry (m5t_fast.cell_geometry = rigour's).
usage: python t415_test.py case KIND IDX     KIND: bench (k = 4, norm 10), region (k = 10, event), surv (k = 6)
       python t415_test.py all [kinds]       every pre-registered case -> results_t415_<kind>.json
       python t415_test.py patch             rigour.mass_cell with geom_fast.patch_rigour() vs without"""
import json
import sys
import time

import numpy as np

import geom_fast as G
import m5t_fast as F
import rigour as R
from flint import fmpq

BENCH = "data/bench_182_k4.npz"


def case_args(kind, idx):
    if kind == "bench":
        dat = np.load(BENCH, allow_pickle=True); k = int(dat["k"])
        g = np.full((10, 16), -1); g[:k] = np.unpackbits(np.frombuffer(dat["hist"][idx], np.uint8)).reshape(k, 16)
        t0s = [t for t in (2.25, 2.5) if any(o["i"] == idx for o in json.load(open(f"results_a_m5t_{t}_2000.json")))]
        return tuple(int(v) for v in dat["tri"]), g, k, dict(), t0s, 2000
    if kind == "region":
        dat = np.load("data/regions_1-8-2.npz")
        return tuple(int(v) for v in dat["tri"]), dat["gates"][idx].astype(int), 10, dict(event=True), [1.5], 4000
    dat = np.load("data/surv_d0.333_1-8-2_k6.npz", allow_pickle=True)
    return tuple(int(v) for v in dat["tri"]), dat["gates"][idx].astype(int), int(dat["k"]), dict(norm=1 / R.q(float(dat["delta"]))), [2.5], 2000


def reference(tri, g, k, kw):
    cap = {}
    oe, op = R.Poly.enumerate, R.pulling_triangulation

    def en(self, *a, **kk):
        V, inc = oe(self, *a, **kk); cap["V"], cap["inc"] = V, inc; return V, inc

    def pt(V, inc, n):
        S, d = op(V, inc, n); cap["S"] = S; return S, d
    R.Poly.enumerate, R.pulling_triangulation = en, pt
    try:
        t = time.perf_counter(); geo = F.cell_geometry(tri, g, k, **kw); tr = time.perf_counter() - t
    finally:
        R.Poly.enumerate, R.pulling_triangulation = oe, op
    return geo, cap, tr


def run_case(kind, idx):
    tri, g, k, kw, t0s, bud = case_args(kind, idx)
    st0 = dict(G.STATS)
    t = time.perf_counter(); out = G.cell_geometry_fast(tri, g, k, return_exact=True, **kw); tf = time.perf_counter() - t
    geo_r, cap, tr = reference(tri, g, k, kw)
    res = dict(kind=kind, idx=idx, t_ref=tr, t_fast=tf, speedup=tr / tf, stats={a: G.STATS[a] - st0[a] for a in st0})
    if out is None or geo_r is None:
        res.update(ok=(out is None and geo_r is None), empty=True)
        return res
    geo_f, (V, inc, S, M) = out
    mism = []
    if V != cap["V"]:
        mism.append("V")
    if inc != cap["inc"]:
        mism.append("inc")
    if S != cap["S"]:
        mism.append("S")
    for j, nm in enumerate(["d", "mlow", "jp3", "Mlo", "Mhi", "S", "vlo", "vhi", "nV"]):
        a, b = geo_f[j], geo_r[j]
        if not (np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b):
            mism.append(nm)
    res.update(geometry_ok=not mism, vertices=len(V), simplices=len(S))
    print("geometry", res, flush=True)
    del V, inc, S, M, cap
    for t0 in t0s:
        if len(geo_f[5]) >= bud:                        # no refinement: mass_cell_fast = the outward sums (chunked)
            mf, mr = _mass_chunked(geo_f, t0), _mass_chunked(geo_r, t0)
        else:
            mf = F.mass_cell_fast(tri, g, k, t0, budget=bud, geom=geo_f, **kw)
            mr = F.mass_cell_fast(tri, g, k, t0, budget=bud, geom=geo_r, **kw)
        if mf != mr:
            mism.append(f"mass_{t0}")
        res[f"ub_{t0}"] = mr[1]
    res.update(ok=not mism, mismatches=mism)
    return res


def _mass_chunked(geo, t0, chunk=50000):
    """mass_cell_fast's (lb, ub) for a cell whose root triangulation already meets the budget (no refinement)."""
    import math
    d, mlow, jp3, Mvl, Mvh, S, vlo, vhi, nV = geo
    P = F.Params(d, mlow, jp3, t0)
    los, his = [], []
    for a in range(0, len(S), chunk):
        s = S[a:a + chunk]
        lo, hi = F.bounds_batch(Mvl[s], Mvh[s], vlo[a:a + chunk], vhi[a:a + chunk], P)
        los.append(lo); his.append(hi)
    lo = np.concatenate(los); hi = np.concatenate(his)
    return max(0.0, float(F.dn(math.fsum(lo)))), float(F.up(math.fsum(hi)))


def cases(kind):
    if kind == "bench":
        return sorted({o["i"] for t in (2.25, 2.5) for o in json.load(open(f"results_a_m5t_{t}_2000.json"))})
    return list(range(5)) if kind == "region" else list(range(20))


def test_patch():
    """rigour.mass_cell (arb M5t) with geom_fast.patch_rigour() gives the same [lb, ub] (bench cells 3, 44; surv 0)."""
    out = []
    for kind, idx in (("bench", 3), ("bench", 44), ("surv", 0)):
        tri, g, k, kw, t0s, _ = case_args(kind, idx)
        args = dict(budget=200, tangent=True, zbins=24, zratio=fmpq(23, 20), **kw)
        oe, op = R.Poly.enumerate, R.pulling_triangulation
        t = time.perf_counter(); ref = R.mass_cell(tri, g, k, t0s[0], **args); tr = time.perf_counter() - t
        G.patch_rigour()
        try:
            t = time.perf_counter(); new = R.mass_cell(tri, g, k, t0s[0], **args); tf = time.perf_counter() - t
        finally:
            R.Poly.enumerate, R.pulling_triangulation = oe, op
        out.append(dict(kind=kind, idx=idx, ref=ref, new=new, equal=ref == new, t_ref=tr, t_new=tf)); print(out[-1], flush=True)
    json.dump(out, open("results_t415_patch.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "patch":
        test_patch()
    elif sys.argv[1] == "case":
        print(json.dumps(run_case(sys.argv[2], int(sys.argv[3])), indent=1))
    else:
        for kind in (sys.argv[2:] or ["bench", "surv", "region"]):
            rows = []
            for i in cases(kind):
                rows.append(run_case(kind, i)); print(rows[-1], flush=True)
            s = dict(kind=kind, cases=len(rows), mismatches=sum(not r["ok"] for r in rows),
                     t_ref=sum(r["t_ref"] for r in rows), t_fast=sum(r["t_fast"] for r in rows))
            s["speedup_total"] = s["t_ref"] / s["t_fast"]
            print(json.dumps(s))
            json.dump(dict(summary=s, cases=rows), open(f"results_t415_{kind}.json", "w"), indent=0)
