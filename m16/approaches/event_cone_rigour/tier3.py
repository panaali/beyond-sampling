"""T416 tier 3: refine the heaviest k = 10 cells, whose root triangulations (5e5-1.5e6 simplices) exceed the M5t
budget, so tier 2 bounded them without refinement. Same rigorous scheme as m5t_fast.mass_cell_fast: the exact
geometry (geom_fast), bounds_batch per simplex, bisection of the longest m-space edge. The root simplices stay
referenced by index (bounded memory) and the children are stored explicitly. Each round splits the top
max(8, n_refined // 16) simplices by hi - lo. The tiling is exact at all times, so ub = Σ hi stays rigorous.
usage: python tier3.py RESULT_JSON SURV_FILE t0 child_budget i [i ...]   (updates RESULT_JSON in place)"""
import itertools
import json
import math
import sys
import time

import numpy as np

import geom_fast as GF
import m5t_fast as F
import rigour as R
from m5t_fast import Params, bounds_batch, dn, up


def refine_big(tri, gates, t0, norm, child_budget):
    gm = GF.cell_geometry_fast(tri, gates, 10, norm=norm, event=True)
    d, mlow, jp3, Mv_lo, Mv_hi, S, vlo, vhi, nV = gm
    I = sorted(set(range(10)) | set(tri))
    P = Params(d, mlow, jp3, t0, 24, R.fmpq(23, 20), jp2=I.index(tri[1]))
    CH = 100_000
    parts = [bounds_batch(Mv_lo[S[a:a + CH]], Mv_hi[S[a:a + CH]], vlo[a:a + CH], vhi[a:a + CH], P) for a in range(0, len(S), CH)]
    rlo = np.concatenate([p[0] for p in parts]); rhi = np.concatenate([p[1] for p in parts]); alive = np.ones(len(S), bool)
    ub0 = float(up(math.fsum(rhi)))
    Cl = np.zeros((0,) + Mv_lo.shape[1:][-1:] * 0 + (S.shape[1], Mv_lo.shape[1])) if False else None
    cMl = np.zeros((0, S.shape[1], Mv_lo.shape[1])); cMh = cMl.copy(); cvl = np.zeros(0); cvh = np.zeros(0); clo = np.zeros(0); chi = np.zeros(0)
    V = S.shape[1]; ij = np.array(list(itertools.combinations(range(V), 2)))
    while len(clo) < child_budget:
        gap = np.concatenate([np.where(alive, rhi - rlo, -np.inf), chi - clo])
        m = int(min(max(8, len(clo) // 16 + 8), child_budget - len(clo), len(gap)))
        top = np.argpartition(-gap, m - 1)[:m]
        tr = top[top < len(S)]; tc = top[top >= len(S)] - len(S)
        Pl = np.concatenate([Mv_lo[S[tr]], cMl[tc]]); Ph = np.concatenate([Mv_hi[S[tr]], cMh[tc]])
        vl = np.concatenate([vlo[tr], cvl[tc]]); vh = np.concatenate([vhi[tr], cvh[tc]])
        alive[tr] = False
        keep = np.ones(len(clo), bool); keep[tc] = False
        cMl, cMh, cvl, cvh, clo, chi = cMl[keep], cMh[keep], cvl[keep], cvh[keep], clo[keep], chi[keep]
        Pm = 0.5 * Pl + 0.5 * Ph
        el = ((Pm[:, ij[:, 0]] - Pm[:, ij[:, 1]]) ** 2).sum(-1); e = ij[el.argmax(1)]; r = np.arange(len(Pl))
        mlo = dn(dn(Pl[r, e[:, 0]] + Pl[r, e[:, 1]]) * 0.5); mhi = up(up(Ph[r, e[:, 0]] + Ph[r, e[:, 1]]) * 0.5)
        C1l, C1h, C2l, C2h = Pl.copy(), Ph.copy(), Pl.copy(), Ph.copy()
        C1l[r, e[:, 0]] = mlo; C1h[r, e[:, 0]] = mhi; C2l[r, e[:, 1]] = mlo; C2h[r, e[:, 1]] = mhi
        nvl = dn(vl * 0.5); nvh = up(vh * 0.5)
        NL = np.concatenate([C1l, C2l]); NH = np.concatenate([C1h, C2h]); NVL = np.concatenate([nvl, nvl]); NVH = np.concatenate([nvh, nvh])
        a, b = bounds_batch(NL, NH, NVL, NVH, P)
        cMl = np.concatenate([cMl, NL]); cMh = np.concatenate([cMh, NH]); cvl = np.concatenate([cvl, NVL]); cvh = np.concatenate([cvh, NVH])
        clo = np.concatenate([clo, a]); chi = np.concatenate([chi, b])
    ub = float(up(math.fsum(rhi[alive]) + math.fsum(chi)))
    lb = max(0.0, float(dn(math.fsum(rlo[alive]) + math.fsum(clo))))
    return lb, ub, ub0, len(S), nV


if __name__ == "__main__":
    rj, f, t0, cb = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    c = np.load(f, allow_pickle=True); tri = tuple(int(v) for v in c["tri"]); norm = 1 / R.q(float(c["delta"]))
    out = json.load(open(rj)); by = {o["i"]: o for o in out}
    for i in map(int, sys.argv[5:]):
        t = time.time(); lb, ub, ub0, nS, nV = refine_big(tri, c["gates"][i].astype(int), t0, norm, cb)
        o = by[i]; o["tier3"] = dict(ub=ub, lb=lb, ub_root=ub0, child_budget=cb, roots=nS, t=time.time() - t)
        o["ub"] = min(o["ub"], ub); o["lb"] = max(o.get("lb", 0.0), lb)
        print(i, o["tier3"], flush=True)
        json.dump(out, open(rj, "w"), indent=0)
