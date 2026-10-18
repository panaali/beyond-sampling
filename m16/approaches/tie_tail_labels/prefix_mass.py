"""T218 part 2 (T222 independent replication): exact Gaussian masses of the target's k-step prefix regions and the
mass-coverage curve N_delta(k).

Cells come from prefix_blind.py --full --dump (polyhedral cones {G x >= 0}, k = 2, 3, 4). Each cone -> extreme rays
(all (k-1)-subsets of rows, null vector, kept if feasible) -> triangulated into simplicial cones (fan for k <= 3,
Delaunay of the section for k = 4) -> orthant probability P(lambda >= 0), lambda ~ N(0, (R^T R)^-1):
k=2: 1/4 + asin(c)/(2 pi); k=3: 1/8 + sum asin(c_ij)/(4 pi); k=4: Plackett's identity
P = 1/16 + int_0^1 sum_{i<j} C_ij phi2(s C_ij) P2(cond. corr of the other pair | Y_i = Y_j = 0) ds,
rewritten per pair with rho = sin(phi) on [0, asin C_ij] (smooth), Gauss-Legendre with n and 2n nodes.
All asin(C_ij) come from robust normal angles (atan2), so thin cones keep full precision.

usage: python prefix_mass.py [K ...]  -> results/prefix_mass.json
"""
import itertools
import json
import os
import pickle
import sys

import numpy as np
from scipy.integrate import quad
from scipy.spatial import Delaunay

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-10


def rays(G, k):
    """Extreme rays of the pointed cone {G x >= 0}: null vectors of all (k-1)-row subsets (cofactors, batched),
    kept if feasible and the subset has rank k-1; duplicates merged."""
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    S = np.array(list(itertools.combinations(range(len(G)), k - 1)))
    A = G[S]                                                   # (nsub, k-1, k)
    if k == 2:
        V = np.stack([-A[:, 0, 1], A[:, 0, 0]], 1)
    else:
        V = np.stack([(-1) ** j * np.linalg.det(np.delete(A, j, axis=2)) for j in range(k)], 1)
    nv = np.linalg.norm(V, axis=1); ok = nv > 1e-9
    V = V[ok] / nv[ok, None]
    V = np.concatenate([V, -V])
    V = V[(V @ G.T >= -TOL).all(1)]
    out = []
    for v in V:
        if not any(np.abs(v - w).max() < 1e-9 for w in out):
            out.append(v)
    return np.array(out)


def simplices(Rr, k):
    """Triangulate the pointed cone spanned by rays Rr (m, k) into simplicial cones (lists of k ray indices)."""
    m = len(Rr)
    if m == k:
        return [list(range(k))]
    c = Rr.sum(0); c /= np.linalg.norm(c)
    P = Rr / (Rr @ c)[:, None]                    # section c.x = 1
    B = np.linalg.svd(np.eye(k) - np.outer(c, c))[0][:, :k - 1]
    Q = P @ B
    if k == 2:
        raise ValueError("k=2 cone with >2 rays")
    if k == 3:                                    # convex polygon: sort by angle, fan from vertex 0
        ctr = Q.mean(0); o = np.argsort(np.arctan2(Q[:, 1] - ctr[1], Q[:, 0] - ctr[0]))
        return [[o[0], o[i], o[i + 1]] for i in range(1, m - 1)]
    return [list(s) for s in Delaunay(Q).simplices]


def normals_angles(Rs):
    """Simplicial cone with ray columns Rs (k,k): unit facet normals N (rows; n_i . r_j = 0 for j != i, n_i . r_i > 0)
    and the angles theta_ij between them, computed robustly (2 atan2(|a-b|, |a+b|)).
    lambda = R^-1 x has correlation C_ij = cos theta_ij, and asin(C_ij) = pi/2 - theta_ij without cancellation."""
    k = Rs.shape[0]
    N = np.empty((k, k))
    for i in range(k):
        A = np.delete(Rs, i, axis=1).T                       # (k-1, k): the other rays
        n = np.array([(-1) ** j * np.linalg.det(np.delete(A, j, axis=1)) for j in range(k)])
        if n @ Rs[:, i] < 0:
            n = -n
        N[i] = n / np.linalg.norm(n)
    D = np.linalg.norm(N[:, None] - N[None], axis=2); S = np.linalg.norm(N[:, None] + N[None], axis=2)
    return N, 2 * np.arctan2(D, S)


def orth2(c):
    return 0.25 + np.arcsin(np.clip(c, -1, 1)) / (2 * np.pi)


PAIRS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)), ((1, 2), (0, 3)), ((1, 3), (0, 2)), ((2, 3), (0, 1))]


def _cond(C, s, i, j, k, l):
    """P(Y_k > 0, Y_l > 0 | Y_i = Y_j = 0) under R(s) = (1-s) I + s C, batch (n,4,4), s (n,)."""
    Rm = s[:, None, None] * C + (1 - s)[:, None, None] * np.eye(4)
    a = [i, j]; b = [k, l]
    Saa = Rm[:, a][:, :, a]; Sab = Rm[:, a][:, :, b]; Sbb = Rm[:, b][:, :, b]
    Sc = Sbb - np.transpose(Sab, (0, 2, 1)) @ np.linalg.solve(Saa, Sab)
    return orth2(Sc[:, 0, 1] / np.sqrt(Sc[:, 0, 0] * Sc[:, 1, 1]))


def orth4(C, TH, n=48):
    """Plackett: P4(C) = 1/16 + sum_{i<j} (1/2pi) int_0^{phi_ij} f_ij(sin(phi)/C_ij) dphi, phi_ij = asin C_ij
    (= pi/2 - theta_ij, robust); f_ij = conditional 2-D orthant probability. Gauss-Legendre n vs 2n per pair."""
    def gl(nn):
        t, w = np.polynomial.legendre.leggauss(nn); t = 0.5 * (t + 1); w = 0.5 * w
        tot = np.full(len(C), 1 / 16)
        for (i, j), (k, l) in PAIRS:
            if i > j:
                continue
            phi = np.pi / 2 - TH[:, i, j]; c = C[:, i, j]
            acc = np.zeros(len(C))
            for ti, wi in zip(t, w):
                ph = phi * ti
                s = np.where(np.abs(c) > 1e-300, np.sin(ph) / np.where(c == 0, 1, c), ti)
                acc += wi * _cond(C, np.clip(s, 0, 1), i, j, k, l)
            tot += phi * acc / (2 * np.pi)
        return tot
    p1, p2 = gl(n), gl(2 * n)
    bad = np.flatnonzero(np.abs(p1 - p2) > 1e-13)
    if len(bad):                                             # composite GL, graded geometrically toward phi_ij
        p2[bad] = graded(C[bad], TH[bad])
    return p2, len(bad), float(np.abs(p1 - p2).max()) if len(C) else 0.0


def graded(C, TH, levels=45, nn=12):
    t, w = np.polynomial.legendre.leggauss(nn); t = 0.5 * (t + 1); w = 0.5 * w
    edges = np.concatenate([[0.0], 1 - 0.5 ** np.arange(1, levels + 1), [1.0]])
    tot = np.full(len(C), 1 / 16)
    for (i, j), (k, l) in PAIRS:
        if i > j:
            continue
        phi = np.pi / 2 - TH[:, i, j]; c = C[:, i, j]; acc = np.zeros(len(C))
        for lo, hi in zip(edges[:-1], edges[1:]):
            for ti, wi in zip(lo + (hi - lo) * t, (hi - lo) * w):
                s = np.where(c != 0, np.sin(phi * ti) / np.where(c == 0, 1, c), ti)
                acc += wi * _cond(C, np.clip(s, 0, 1), i, j, k, l)
        tot += phi * acc / (2 * np.pi)
    return tot


def cone_masses(cells, k):
    simp_C, owner = [], []
    for ci, (G, _) in enumerate(cells):
        Rr = rays(G, k)
        for s in simplices(Rr, k):
            Rs = Rr[s].T
            if abs(np.linalg.det(Rs)) < 1e-14:
                continue
            N, TH = normals_angles(Rs)
            simp_C.append((N @ N.T, TH)); owner.append(ci)
    C = np.array([c for c, _ in simp_C]); TH = np.array([t for _, t in simp_C]); owner = np.array(owner)
    asn = np.pi / 2 - TH                                     # asin(C_ij), robust
    if k == 2:
        p = 0.25 + asn[:, 0, 1] / (2 * np.pi); info = {}
    elif k == 3:
        p = 0.125 + (asn[:, 0, 1] + asn[:, 0, 2] + asn[:, 1, 2]) / (4 * np.pi); info = {}
    else:
        p, nbad, d = orth4(C, TH); info = dict(gl_disagree_gt_1e13=nbad, max_gl_n_vs_2n=d)
    m = np.bincount(owner, weights=p, minlength=len(cells))
    return m, dict(simplices=len(C), **info)


def selftest():
    C = np.full((1, 4, 4), 0.5); C[0][np.diag_indices(4)] = 1
    p, _, _ = orth4(C, np.arccos(np.clip(C, -1, 1)))
    assert abs(p[0] - 0.2) < 1e-12, p                       # exchangeable rho = 1/2: 1/(n+1)
    rng = np.random.default_rng(1); A = rng.standard_normal((4, 4)); S = A @ A.T
    d = np.sqrt(np.diag(S)); C = (S / np.outer(d, d))[None]
    p, _, _ = orth4(C, np.arccos(np.clip(C, -1, 1)))
    x = rng.multivariate_normal(np.zeros(4), C[0], 4_000_000)
    mc = (x > 0).all(1).mean()
    print(f"selftest: P4(rho=1/2)=0.2 ok; random C: Plackett {p[0]:.6f} vs MC {mc:.6f} +- {np.sqrt(mc*(1-mc)/4e6):.6f}")


def coverage(m, deltas):
    s = np.sort(m)[::-1]; cum = np.cumsum(s)
    return {f"{d:g}": int(np.searchsorted(cum, 1 - d) + 1) for d in deltas}


if __name__ == "__main__":
    selftest()
    ks = [int(a) for a in sys.argv[1:]] or [2, 3, 4]
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    fn = os.path.join(HERE, "results", "prefix_mass.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for k in ks:
        cells = pickle.load(open(os.path.join(HERE, "data", f"cells_full_k{k}.pkl"), "rb"))
        m, info = cone_masses(cells, k)
        reg = {}
        for (G, h), mi in zip(cells, m):
            reg[h] = reg.get(h, 0.0) + mi
        rm = np.array(list(reg.values()))
        r = dict(k=k, cells=len(cells), regions=len(reg), mass_sum=float(m.sum()), mass_sum_err=float(m.sum() - 1),
                 min_region_mass=float(rm.min()), N_delta=coverage(rm, deltas),
                 exp_entropy=float(np.exp(-(rm[rm > 0] * np.log(rm[rm > 0])).sum())), **info)
        res[str(k)] = r
        np.save(os.path.join(HERE, "data", f"region_mass_k{k}.npy"), np.sort(rm)[::-1])
        print(json.dumps(r), flush=True)
    for k in range(3, 5):
        if str(k) in res and str(k - 1) in res:
            res[str(k)]["growth"] = {d: res[str(k)]["N_delta"][d] / res[str(k - 1)]["N_delta"][d]
                                     for d in res[str(k)]["N_delta"]}
            res[str(k)]["growth_regions"] = res[str(k)]["regions"] / res[str(k - 1)]["regions"]
    json.dump(res, open(fn, "w"), indent=1)
    for k in sorted(res):
        print(k, res[k].get("growth"))
