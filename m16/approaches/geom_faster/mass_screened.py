"""T452 option (b): screened tier-2 mass for cells whose root triangulation already exceeds the budget (no
refinement runs there, so m5t_fast.mass_cell_fast evaluates the full M5t bound on every root simplex).

Each root simplex sigma_j of the exact root triangulation (which tiles the cell's slice polytope exactly) gets a cheap
rigorous upper bound, the plain M5 bound:
    m_j = c0 vol_j E_sigma[ int_t0^inf z^(d-1) exp(-z^2 s(y)/2) Phi(z y_p3)^mlow dz ]
        <= c0 vol_j int_t0^inf z^(d-1) exp(-z^2 s_min/2) dz = c0 vol_j G(d, s_min, t0, inf),
since s(y) = |m(y)|^2 >= s_min on sigma_j and Phi^mlow <= 1. Here s_min is a lower bound on |m|^2 over sigma_j from the
separating hyperplane through the vertex-centroid direction w (min_i w.M_i / |w|, outward), and >= 1/2 on the slice when
jp2 is given (as m5t_fast). Simplices are taken by decreasing cheap bound; the full M5t bound (m5t_fast.bounds_batch,
audited) is evaluated on them chunk by chunk until the remaining cheap bounds sum to <= eps x the evaluated M5t sum.
    ub = sum_evaluated hi_j + sum_rest cheap_j      (every simplex keeps a valid upper bound)
    lb = sum_evaluated lo_j                          (a subset of disjoint simplices; integrand >= 0)
Both sums are math.fsum (correctly rounded) and rounded outward. Cells below the budget use mass_cell_fast unchanged."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour"))
import m5t_fast as F  # noqa: E402

STATS = dict(cells=0, roots=0, evaluated=0)


def cheap_hi(Mlo, Mhi, vhi, P):
    """Rigorous per-simplex upper bound c0 vol G(d, s_min, t0, inf) (vectorised, outward rounding)."""
    Mf = 0.5 * Mlo + 0.5 * Mhi
    w = Mf.mean(1)                                       # any float vector
    dlo, _ = F.idot(w[:, None, :], Mlo, Mhi)
    L = dlo.min(1)
    w2 = F.sumsq_up(w)
    smin = np.where((L > 0) & (w2 > 0), F.dn(F.dn(L * L) / np.where(w2 > 0, w2, 1.0)), 0.0)
    if P.jp2 is not None:
        if np.any(Mlo[:, :, P.jp2] > F.up(1 + Mhi[:, :, P.jp3])) or np.any(Mhi[:, :, P.jp2] < F.dn(1 + Mlo[:, :, P.jp3])):
            raise ValueError("vertex enclosures inconsistent with m_p2 = 1 + m_p3")
        smin = np.maximum(smin, 0.5)
    out = np.full(len(smin), np.inf)
    ok = smin > 0
    if ok.any():
        A = np.full((ok.sum(), 1), P.edges[0]); B = np.full((ok.sum(), 1), np.inf)
        e, gm = F.G_fac(P, smin[ok][:, None], A, B, True)
        val = F.up(F.exp_up(e[:, 0]) * gm[:, 0])
        out[ok] = F.up(F.up(P.c0[1] * np.asarray(vhi)[ok]) * val)
    return np.where(np.isnan(out), np.inf, out)


def mass_cell_screened(tri, gates, k, t0, budget=4000, norm=None, event=False, geom=None, eps=1e-3, chunk=4096,
                       return_info=False, **kw):
    """mass_cell_fast for cells under budget; the screened root bound otherwise. Returns (lb, ub[, info])."""
    import rigour as R
    norm = R.NORM if norm is None else norm
    gm = geom
    if gm is None:
        return F.mass_cell_fast(tri, gates, k, t0, budget=budget, norm=norm, event=event, return_info=return_info, **kw)
    d, mlow, jp3, Mvl, Mvh, S, vlo, vhi, nV = gm
    if len(S) < budget:
        return F.mass_cell_fast(tri, gates, k, t0, budget=budget, norm=norm, event=event, return_info=return_info, geom=gm, **kw)
    I = sorted(set(range(k)) | set(tri))
    P = F.Params(d, mlow, jp3, t0, kw.get("zbins", 24), kw.get("zratio", F.fmpq(23, 20)), jp2=I.index(tri[1]))
    cheap = np.concatenate([cheap_hi(Mvl[S[a:a + chunk]], Mvh[S[a:a + chunk]], vhi[a:a + chunk], P)
                            for a in range(0, len(S), chunk)])
    order = np.argsort(-cheap, kind="stable")
    his, los = [], []
    done = 0
    while done < len(order):
        idx = order[done:done + chunk]
        lo, hi = F.bounds_batch(Mvl[S[idx]], Mvh[S[idx]], vlo[idx], vhi[idx], P)
        his.append(hi); los.append(lo); done += len(idx)
        rest = math.fsum(cheap[order[done:]]) if done < len(order) else 0.0
        if rest <= eps * math.fsum(np.concatenate(his)):
            break
    hi_e = np.concatenate(his); lo_e = np.concatenate(los)
    ub = float(F.up(math.fsum(np.concatenate([hi_e, cheap[order[done:]]]))))
    lb = max(0.0, float(F.dn(math.fsum(lo_e))))
    STATS["cells"] += 1; STATS["roots"] += len(S); STATS["evaluated"] += done
    info = dict(d=d, mlow=mlow, vertices=nV, simplices0=len(S), simplices=len(S), evaluated=done, screened=True)
    return (lb, ub, info) if return_info else (lb, ub)
