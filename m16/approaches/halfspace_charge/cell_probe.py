"""T437 secondary probe in T398's cell setting (non-rigorous, not for the kill decision): forward cells = gate
patterns of steps 1..k (exactly linear there), steps k+1..10 relaxed by CROWN with LP intermediate bounds over
cell ∩ T414's root u-box. For each cell holding slab samples: prune if an LP lower bound of some margin is > 0;
otherwise the halfspaces a_j.u + b_j <= 0 (the cell's own linear lower bounds) cut the cell to Q(cell), and we
measure the slab-mass fraction of the cell inside Q by the slab samples that fall in it.
Slab samples: Gibbs sampler on x ~ N(0, I) | {x_p1 >= x_p2 >= x_p3 + 2.5, lows <= x_p3}.
usage: python cell_probe.py p1,p2,p3 K [NSAMP] OUT.json"""
import json
import sys
import time

import numpy as np
import torch as th
from scipy.optimize import linprog
from scipy.stats import truncnorm

import t414_ibab as ib

th.set_num_threads(1)
D, H, T, T0 = 10, 16, 10, 2.5
Wi, Wh, Wo = (w.numpy() for w in (ib.Wi, ib.Wh, ib.Wo))


def gibbs(tri, n, burn=300, thin=20, chains=200, seed=437):
    p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]; rng = np.random.default_rng(seed)
    x = np.zeros((chains, D)); x[:, p3] = -1.3; x[:, p2] = 1.3; x[:, p1] = 1.6; x[:, lows] = -1.6
    out = []; it = 0
    while len(out) * chains < n:
        for i in range(D):
            if i == p1:
                lo, hi = x[:, p2], np.full(chains, np.inf)
            elif i == p2:
                lo, hi = x[:, p3] + T0, x[:, p1]
            elif i == p3:
                lo, hi = x[:, lows].max(1), x[:, p2] - T0
            else:
                lo, hi = np.full(chains, -np.inf), x[:, p3]
            x[:, i] = truncnorm.rvs(lo, hi, random_state=rng)
        it += 1
        if it > burn and it % thin == 0:
            out.append(x.copy())
    return np.concatenate(out)[:n]


def to_u(x, tri):
    p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]; g = x[:, p2] - x[:, p3]; y = x / g[:, None]
    return np.column_stack([y[:, p3], y[:, p1] - y[:, p2]] + [y[:, p3] - y[:, l] for l in lows])


def gates(y, k):
    h = np.zeros((len(y), H)); gs = []
    for t in range(k):
        z = h @ Wh.T + y[:, t:t + 1] * Wi[None]; gs.append(z > 0); h = np.maximum(z, 0)
    return np.concatenate(gs, 1)


def cell_bounds(gpat, k, P, q, lo_u, hi_u, p3):
    """returns (lb (9,), A (9, 9), b (9,), feasible) for the cell with gate pattern gpat (k*16 bools)"""
    Gs = gpat.reshape(k, H).astype(float); Hy = np.zeros((H, D)); rows = []
    for t in range(k):
        Z = Wh @ Hy; Z[:, t] += Wi
        rows.append(np.where(Gs[t][:, None] > 0, -Z, Z)); Hy = Gs[t][:, None] * Z    # z >= 0 for on, <= 0 off
    Gy = np.concatenate(rows); Au = Gy @ P; bu = -(Gy @ q)                            # Gy (Pu + q) <= 0
    bnds = list(zip(lo_u, hi_u))

    def lpmin(c):
        r = linprog(c, A_ub=Au, b_ub=bu, bounds=bnds, method="highs")
        return r.fun if r.status == 0 else None
    if lpmin(np.zeros(9)) is None:
        return None
    Hu, hc = Hy @ P, Hy @ q                                                            # h_k = Hu u + hc
    lo, up = {}, {}

    def back(C, s_top):                                                              # C on z_{s_top}, s_top > k
        Cu = np.zeros((C.shape[0], 9)); c0 = np.zeros(C.shape[0])
        for s in range(s_top, k, -1):
            ci = C @ Wi; Cu += np.outer(ci, P[s - 1]); c0 += ci * q[s - 1]
            Ch = C @ Wh
            if s == k + 1:
                Cu += Ch @ Hu; c0 += Ch @ hc; break
            l, u = lo[s - 1], up[s - 1]; unst = (l < 0) & (u > 0)
            su = np.where(unst, u / np.maximum(u - l, 1e-300), (l >= 0).astype(float))
            bu_ = np.where(unst, -u * l / np.maximum(u - l, 1e-300), 0.0); sl = np.where(unst, (u > -l).astype(float), (l >= 0).astype(float))
            pos = Ch >= 0; c0 += np.where(pos, 0, Ch * bu_[None]).sum(1); C = Ch * np.where(pos, sl[None], su[None])
        return Cu, c0
    for s in range(k + 1, T + 1):
        Cu, c0 = back(np.eye(H), s); l = np.zeros(H); u = np.zeros(H)
        for j in range(H):
            l[j] = lpmin(Cu[j]) + c0[j]; u[j] = -lpmin(-Cu[j]) + c0[j]
        lo[s], up[s] = l, u
    others = [j for j in range(D) if j != p3]; Dm = Wo[others] - Wo[p3]
    l, u = lo[T], up[T]; unst = (l < 0) & (u > 0)
    su = np.where(unst, u / np.maximum(u - l, 1e-300), (l >= 0).astype(float)); bu_ = np.where(unst, -u * l / np.maximum(u - l, 1e-300), 0.0)
    sl = np.where(unst, (u > -l).astype(float), (l >= 0).astype(float)); pos = Dm >= 0
    ch = np.where(pos, 0, Dm * bu_[None]).sum(1); Cz = Dm * np.where(pos, sl[None], su[None])
    if T == k:
        raise ValueError
    A, b = back(Cz, T); b = b + ch
    lb = np.array([lpmin(A[j]) + b[j] for j in range(9)])
    return lb, A, b, True


def main():
    tri = tuple(int(v) for v in sys.argv[1].split(",")); k = int(sys.argv[2]); n = int(sys.argv[3]); out = sys.argv[4]
    P, q, lo_u, hi_u = (v.numpy() for v in ib.coords(tri)); p3 = tri[2]
    t0 = time.time(); x = gibbs(tri, n); u = to_u(x, tri)
    inb = ((u >= lo_u) & (u <= hi_u)).all(1); u = u[inb]; y = u @ P.T + q
    gp = gates(y, k); key = np.packbits(gp, axis=1); cells, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel(); print("samples", len(u), "in-box share", inb.mean(), "cells", len(cells), round(time.time() - t0, 1), flush=True)
    rec = []
    for c in range(len(cells)):
        idx = np.nonzero(inv == c)[0]; r = cell_bounds(gp[idx[0]], k, P, q, lo_u, hi_u, p3)
        if r is None:
            rec.append(dict(n=len(idx), status="lp-infeasible")); continue
        lb, A, b, _ = r; uu = u[idx]; lin = uu @ A.T + b
        assert (lin <= np.inf).all()
        rec.append(dict(n=len(idx), pruned=bool((lb > 0).any()), lbmax=float(lb.max()),
                        fracQ=float((lin <= 0).all(1).mean()), linmax_at_samples=float(lin.max(1).max())))
        if c % 50 == 0:
            print(c, rec[-1], round(time.time() - t0, 1), flush=True)
    N = sum(r["n"] for r in rec)
    pr = sum(r["n"] for r in rec if r.get("pruned")); qm = sum(r["n"] * r["fracQ"] for r in rec if "fracQ" in r and not r["pruned"])
    res = dict(tri=tri, k=k, samples=N, cells=len(rec), pruned_mass_share=pr / N, open_mass_share=1 - pr / N,
               charged_share_halfspace=qm / N, cells_with_any_cut=sum(1 for r in rec if "fracQ" in r and not r["pruned"] and r["fracQ"] < 1),
               cpu_s=time.process_time(), cells_detail=rec)
    print({kk: v for kk, v in res.items() if kk != "cells_detail"}, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
