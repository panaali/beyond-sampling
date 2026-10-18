"""T289 (pre-registered in README): shared forward affine margin bounds per sector, inner/outer rival-error cones, exact
polytope cone masses (d <= 4), Bonferroni/Hunter combination -> accuracy interval. Also the CROWN reference (T279 code)."""
import itertools, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(HERE, "vendor"), os.path.join(ROOT, "m16", "approaches", "cone_mass4")]
import numpy as np
from scipy.spatial import Delaunay, QhullError
import mass4 as M4
import tree as T279                     # vendored unmodified (SHA in README): roots, _relax, _back, Flops


def forward_bounds(model, V, label, fl):
    """Shared forward affine bounds: -> (Lk, Uk) rival-margin lower/upper coefficient matrices (q x d) in lambda."""
    Wi, Wh, Wo = model; H, T = Wh.shape[0], V.shape[1]; d = V.shape[0]
    Wp, Wn = np.maximum(Wh, 0), np.minimum(Wh, 0)
    Al = np.zeros((H, d)); Au = np.zeros((H, d))
    for t in range(T):
        inp = np.outer(Wi, V[:, t]); fl.add(H * d)
        zl = fl.mm(Wp, Al) + fl.mm(Wn, Au) + inp; zu = fl.mm(Wp, Au) + fl.mm(Wn, Al) + inp; fl.add(4 * H * d)
        lo, hi = zl.min(1), zu.max(1); fl.add(2 * H * d)
        alpha, s, b = T279._relax(lo, hi); fl.add(6 * H)
        Al = alpha[:, None] * zl; Au = s[:, None] * zu + b[:, None]; fl.add(4 * H * d)
    riv = [k for k in range(Wo.shape[0]) if k != label]; C = Wo[label][None] - Wo[riv]; fl.add(C.size)
    Cp, Cn = np.maximum(C, 0), np.minimum(C, 0)
    Lk = fl.mm(Cp, Al) + fl.mm(Cn, Au); Uk = fl.mm(Cp, Au) + fl.mm(Cn, Al); fl.add(2 * C.shape[0] * d)
    return Lk, Uk, riv


def crown_bounds(model, V, label, fl):
    """CROWN reference via T279's _back/_relax (same steps as T279's crown(), returning the coefficient matrices)."""
    Wi, Wh, Wo = model; H, T = Wh.shape[0], V.shape[1]; rel = []; I = np.eye(H)
    for t in range(T):
        lo = T279._back(model, V, I, t, rel, fl, lower=True).min(1); hi = T279._back(model, V, I, t, rel, fl, lower=False).max(1)
        rel.append(T279._relax(lo, hi))
    riv = [k for k in range(Wo.shape[0]) if k != label]; C = Wo[label][None] - Wo[riv]
    alpha, s, b = rel[T - 1]; pos, neg = np.maximum(C, 0), np.minimum(C, 0)
    AL, cL = pos * alpha[None] + neg * s[None], neg @ b; AU, cU = pos * s[None] + neg * alpha[None], pos @ b
    Lk = T279._back(model, V, AL, T - 1, rel, fl, True) + cL[:, None]; Uk = T279._back(model, V, AU, T - 1, rel, fl, False) + cU[:, None]
    return Lk, Uk, riv


# ---------------- exact Gaussian masses of {lambda in simplex, a_i . lambda <= 0} mapped through rays R (d <= 4)
def _cone_mass(rays, fl):
    d = rays.shape[0]
    if d == 3:
        a, b, c = rays / np.linalg.norm(rays, axis=1)[:, None]; fl.add(60)
        return 2 * math.atan2(abs(a @ np.cross(b, c)), 1 + a @ b + b @ c + c @ a) / (4 * math.pi)
    ops = M4.Ops(); m = M4.mass4_rays(rays, ops); fl.add(2 * ops.n); return m


def poly_mass(R, cuts, fl, err):
    """R: d x T unit rays (sector simplex); cuts: list of a (d,) with a . lambda <= 0. err[0] += mass error bound."""
    d = R.shape[0]; G = [np.eye(d)[i] for i in range(d)] + [-np.asarray(a, float) for a in cuts]   # G . lambda >= 0
    verts = []
    for act in itertools.combinations(range(len(G)), d - 1):
        A = np.vstack([np.ones(d)] + [G[i] for i in act]); rhs = np.r_[1.0, np.zeros(d - 1)]; fl.add(2 * d ** 3)
        if abs(np.linalg.det(A)) < 1e-14:
            continue
        lam = np.linalg.solve(A, rhs); fl.add(len(G) * 2 * d)
        if all(g @ lam >= -1e-12 for g in G):
            verts.append(lam)
    if len(verts) < d:
        return 0.0
    P = np.unique(np.round(np.array(verts), 13), axis=0)
    if len(P) < d:
        return 0.0
    if len(P) == d:
        simp = [np.arange(d)]
    else:
        try:
            simp = Delaunay(P[:, :d - 1]).simplices
        except QhullError:
            return 0.0                               # flat (zero-volume) polytope
        fl.add(len(P) ** 2 * d)
    tot = 0.0
    for sx in simp:
        L = P[sx]
        if abs(np.linalg.det(np.c_[L[:, :d - 1], np.ones(d)])) < 1e-12:
            continue
        try:
            tot += _cone_mass(L @ R, fl); err[0] += 1e-7 if d == 4 else 1e-15
        except np.linalg.LinAlgError:
            continue                                 # numerically flat simplex: zero mass
    return tot


def sector_interval(R, Lk, Uk, prio, fl, err):
    """-> (P(S), error-mass lower, error-mass upper) for one sector."""
    PS = poly_mass(R, [], fl, err)
    pI = [poly_mass(R, [Uk[k]], fl, err) for k in prio]; pO = [poly_mass(R, [Lk[k]], fl, err) for k in prio]
    pII = sum(poly_mass(R, [Uk[a], Uk[b]], fl, err) for a, b in itertools.combinations(prio, 2))
    chain = sum(poly_mass(R, [Lk[a], Lk[b]], fl, err) for a, b in zip(prio[:-1], prio[1:]))
    lo = max(max(pI), sum(pI) - pII, 0.0); up = min(PS, sum(pO) - chain)
    return PS, lo, up


def estimate(model, bound="forward"):
    Wi, Wh, Wo = model; T = Wo.shape[0]; fl = T279.Flops(); flm = T279.Flops(); err = [0.0]
    lo_err = up_err = tot = 0.0; info = dict(inner_mass=0.0, outer_gap=0.0, sectors_inner=0, sectors_outer=0, n_sectors=2 * T * (T - 1))
    for R, lab, key in T279.roots(T):
        f0 = fl.n
        Lk, Uk, riv = (forward_bounds if bound == "forward" else crown_bounds)(model, R, lab, fl)
        C = Wo[lab][None] - Wo[riv]; prio = list(np.argsort(-np.abs(C).sum(1)))
        PS, lo, up = sector_interval(R, Lk, Uk, prio, flm, err)
        tot += PS; lo_err += lo; up_err += up; info["inner_mass"] += lo; info["outer_gap"] += PS - up
        info["sectors_inner"] += lo > 1e-9; info["sectors_outer"] += up < PS - 1e-9
    e = err[0]
    return dict(l=1 - up_err - e, u=1 - lo_err + e, mid=1 - (up_err + lo_err) / 2, mass_total=tot, mass_err=e,
                flops_bounds=fl.n, flops_mass=flm.n, **{k: float(v) for k, v in info.items()})
