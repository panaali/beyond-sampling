"""T420 phase B: geom_fast vs the reference (m5t_fast.cell_geometry) and vs the independent checks (indep.py).
usage: python audit.py sample                     -> cells.json (the pre-registered B1 + B2 cell list)
       python audit.py run i0 i1 [noref]          -> results_b_<i0>.jsonl (one line per cell, resumable)"""
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ECR = os.path.join(HERE, "..", "event_cone_rigour")
sys.path.insert(0, ECR); sys.path.insert(0, HERE)
import geom_fast as G  # noqa: E402
import m5t_fast as F  # noqa: E402
import rigour as R  # noqa: E402
import indep  # noqa: E402

SURV = os.path.join(ECR, "data", "certs", "surv_d0.333_1-8-2_k10_%s.npz")
REG = os.path.join(ECR, "data", "regions_1-8-2.npz")


def sample():
    import glob
    import subprocess
    rng = np.random.default_rng(420)
    tier2 = []
    names = subprocess.run(["git", "ls-tree", "--name-only", "7cb41899", ECR + "/"], capture_output=True, text=True,
                           cwd=ECR).stdout.split()
    for f in sorted(n for n in names if "results_t10_surv_d0.333_1-8-2_k10_" in n):
        s = f.split("_k10_")[1].split("_")[0]
        rec = json.loads(subprocess.run(["git", "show", "7cb41899:./" + os.path.basename(f)], capture_output=True,
                                        text=True, cwd=ECR).stdout)
        tier2 += sorted({(s, r["i"]) for r in rec if r["tier"] == 2})
    tier2 = sorted(set(tier2))
    pick = [tier2[i] for i in sorted(rng.choice(len(tier2), min(200, len(tier2)), replace=False))]
    cells = [dict(kind="surv", shard=s, i=int(i)) for s, i in pick]
    while len(cells) < 200:
        s = ["s0", "s1"][int(rng.integers(2))]; i = int(rng.integers(len(np.load(SURV % s)["gates"])))
        if not any(c["kind"] == "surv" and c["shard"] == s and c["i"] == i for c in cells):
            cells.append(dict(kind="surv", shard=s, i=i, extra=True))
    nreg = len(np.load(REG)["gates"])
    extra = [int(x) for x in rng.choice(np.arange(6, nreg), 10, replace=False)]
    cells += [dict(kind="region", i=i) for i in list(range(6)) + sorted(extra)]
    json.dump(dict(n_tier2=len(tier2), cells=cells), open(os.path.join(HERE, "cells.json"), "w"), indent=0)
    print("tier-2 cells", len(tier2), "cells", len(cells))


def args_of(c):
    if c["kind"] == "surv":
        d = np.load(SURV % c["shard"], allow_pickle=True)
        return tuple(int(v) for v in d["tri"]), d["gates"][c["i"]].astype(int), 10, dict(norm=1 / R.q(float(d["delta"])), event=True)
    d = np.load(REG)
    return tuple(int(v) for v in d["tri"]), d["gates"][c["i"]].astype(int), 10, dict(event=True)


def fast(tri, g, k, kw):
    cap = {}
    oi, oe = R.irredundant, G.enumerate_fast

    def irr(A, b, *a, **kk):
        cap["Af"], cap["bf"] = A, b; return oi(A, b, *a, **kk)

    def en(Pl, v0, *a, **kk):
        cap["A"] = Pl.A; return oe(Pl, v0, *a, **kk)
    R.irredundant, G.enumerate_fast = irr, en
    st0 = dict(G.STATS)
    try:
        t = time.perf_counter(); out = G.cell_geometry_fast(tri, g, k, return_exact=True, **kw); tf = time.perf_counter() - t
    finally:
        R.irredundant, G.enumerate_fast = oi, oe
    return out, cap, tf, {a: G.STATS[a] - st0[a] for a in st0}


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


def compare(geo_f, ex, geo_r, cap):
    V, inc, S, M = ex
    mm = []
    if V != cap["V"]:
        mm.append("V")
    if list(inc) != list(cap["inc"]):
        mm.append("inc")
    if [tuple(s) for s in S] != [tuple(s) for s in cap["S"]]:
        mm.append("S")
    for j, name in enumerate(["d", "mlow", "jp3", "Mlo", "Mhi", "S_arr", "vlo", "vhi", "nV"]):
        a, b = geo_f[j], geo_r[j]
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            mm.append(name)
    return mm


def run_cell(c, do_ref=True):
    tri, g, k, kw = args_of(c)
    res = dict(c)
    out, cap, tf, st = fast(tri, g, k, kw)
    res.update(t_fast=tf, stats=st)
    if out is None:
        res["empty"] = True
    else:
        geo_f, ex = out
        V, inc, S, M = ex; n = len(V[0])
        res.update(nV=len(V), nS=len(S))
        if do_ref:
            try:
                geo_r, rc, tr = reference(tri, g, k, kw)
                res.update(t_ref=tr, ref_mismatch=compare(geo_f, ex, geo_r, rc) if geo_r is not None else ["ref None"])
                del rc, geo_r
            except MemoryError:
                res["ref"] = "MemoryError"
        t = time.perf_counter()
        res["indep"] = indep.check_cell(cap["Af"], cap["bf"], cap["A"], V, inc, [tuple(s) for s in S], geo_f[6], geo_f[7], n)
        res["t_indep"] = time.perf_counter() - t
    return res


def main():
    if sys.argv[1] == "sample":
        return sample()
    i0, i1 = int(sys.argv[2]), int(sys.argv[3]); do_ref = "noref" not in sys.argv
    cells = json.load(open(os.path.join(HERE, "cells.json")))["cells"]
    path = os.path.join(HERE, "results_b_%d.jsonl" % i0)
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["j"] for l in open(path)}
    for j in range(i0, min(i1, len(cells))):
        if j in done:
            continue
        try:
            r = run_cell(cells[j], do_ref)
        except Exception as e:  # noqa: BLE001
            r = dict(cells[j], error=repr(e)[:300])
        r["j"] = j; r["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        with open(path, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print({a: r.get(a) for a in ("j", "kind", "i", "nV", "nS", "t_fast", "t_ref", "ref_mismatch", "t_indep")},
              (r.get("indep") or {}).get("ok"), r.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
