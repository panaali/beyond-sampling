"""T218 part 2: independent checks of T192's exact prefix region masses (results/prefix/k{4,5}_regionmass.npz on
branch claude/t192-gpu-cost-law, copied to data/t192/).
(a) k=4: all 21,032 region masses vs prefix_mass.py (matched by gate pattern).
(b) k=5: N_delta from T192's masses with this folder's coverage code.
(c) k=5: 200 region masses recomputed here: my k=4 cells lifted to R^5 (split by x4 sign), split by the step-5 gates
    (same LP test as prefix_blind.py), masses by a generic-k Plackett rule (rho = sin phi per pair, conditional
    (k-2)-D orthant in closed form for k-2 = 3) with robust facet angles and a graded Gauss-Legendre mesh.
usage: python compare_t192.py -> results/compare_t192.json
"""
import itertools
import json
import os
import pickle

import numpy as np
from scipy.spatial import Delaunay

import prefix_mass as P
from prefix_blind import range_over, target, TOL as LPTOL

HERE = os.path.dirname(os.path.abspath(__file__))
DELTAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def orth3_batch(R):
    """3-D orthant probability for correlation batch R (n,3,3), closed form."""
    a = np.arcsin(np.clip(R[:, 0, 1], -1, 1)) + np.arcsin(np.clip(R[:, 0, 2], -1, 1)) + np.arcsin(np.clip(R[:, 1, 2], -1, 1))
    return 0.125 + a / (4 * np.pi)


def orth5(C, TH, levels=45, nn=12):
    """P5(C) = 1/32 + sum_{i<j} (1/2pi) int_0^{asin C_ij} P3(cond | Y_i = Y_j = 0; R(sin phi / C_ij)) dphi."""
    t, w = np.polynomial.legendre.leggauss(nn); t = 0.5 * (t + 1); w = 0.5 * w
    edges = np.concatenate([[0.0], 1 - 0.5 ** np.arange(1, levels + 1), [1.0]])
    tot = np.full(len(C), 1 / 32)
    for i, j in itertools.combinations(range(5), 2):
        rest = [x for x in range(5) if x not in (i, j)]
        phi = np.pi / 2 - TH[:, i, j]; c = C[:, i, j]; acc = np.zeros(len(C))
        for lo, hi in zip(edges[:-1], edges[1:]):
            for ti, wi in zip(lo + (hi - lo) * t, (hi - lo) * w):
                s = np.clip(np.where(c != 0, np.sin(phi * ti) / np.where(c == 0, 1, c), ti), 0, 1)
                Rm = s[:, None, None] * C + (1 - s)[:, None, None] * np.eye(5)
                Saa = Rm[:, [i, j]][:, :, [i, j]]; Sab = Rm[:, [i, j]][:, :, rest]; Sbb = Rm[:, rest][:, :, rest]
                Sc = Sbb - np.transpose(Sab, (0, 2, 1)) @ np.linalg.solve(Saa, Sab)
                d = np.sqrt(np.einsum("nii->ni", Sc))
                acc += wi * orth3_batch(Sc / (d[:, :, None] * d[:, None, :]))
        tot += phi * acc / (2 * np.pi)
    return tot


def cone_mass5(G):
    Rr = P.rays(G, 5)
    if len(Rr) == 5:
        S = [list(range(5))]
    else:
        c = Rr.sum(0); c /= np.linalg.norm(c)
        Pt = Rr / (Rr @ c)[:, None]
        B = np.linalg.svd(np.eye(5) - np.outer(c, c))[0][:, :4]
        S = [list(s) for s in Delaunay(Pt @ B).simplices]
    Cs, Ts = [], []
    for s in S:
        Rs = Rr[s].T
        if abs(np.linalg.det(Rs)) < 1e-14:
            continue
        N, TH = P.normals_angles(Rs); Cs.append(N @ N.T); Ts.append(TH)
    return float(orth5(np.array(Cs), np.array(Ts)).sum()) if Cs else 0.0


def A_from_hist(hist, k, Wi, Wh):
    A = np.zeros((len(Wi), k))
    for t in range(k):
        g = np.unpackbits(np.frombuffer(hist[2 * t:2 * t + 2], np.uint8))[:len(Wi)].astype(float)
        Z = Wh @ A; Z[:, t] += Wi; A = g[:, None] * Z
    return A


def region_mass_k5(pattern, cells4, Wi, Wh):
    """Mass of the k=5 region with gate pattern (10 bytes): lift matching k=4 cells, split at step 5."""
    pre, last = pattern[:8], np.unpackbits(np.frombuffer(pattern[8:10], np.uint8))[:len(Wi)]
    tot = 0.0; ncells = 0
    for G4, h in cells4:
        if h != pre:
            continue
        A = np.hstack([A_from_hist(h, 4, Wi, Wh), np.zeros((len(Wi), 1))])
        Z = Wh @ A; Z[:, 4] += Wi
        for sg in (1.0, -1.0):
            pieces = [np.vstack([np.hstack([G4, np.zeros((len(G4), 1))]), sg * np.eye(5)[4]])]
            for j in range(len(Wi)):
                nxt = []
                for Gp in pieces:
                    lo, hi = range_over(Gp, Z[j], 5)
                    want = last[j]
                    if hi <= LPTOL:
                        if want == 0:
                            nxt.append(Gp)
                    elif lo >= -LPTOL:
                        if want == 1:
                            nxt.append(Gp)
                    else:
                        nxt.append(np.vstack([Gp, Z[j] if want else -Z[j]]))
                pieces = nxt
                if not pieces:
                    break
            for Gp in pieces:
                tot += cone_mass5(Gp); ncells += 1
    return tot, ncells


def coverage(m):
    s = np.sort(m)[::-1]; cum = np.cumsum(s)
    return {f"{d:g}": int(np.searchsorted(cum, 1 - d) + 1) for d in DELTAS}


def main():
    Wi, Wh, _ = target(); out = {}
    # (a) k=4 one-to-one
    d4 = np.load(os.path.join(HERE, "data", "t192", "k4_regionmass.npz"))
    cells4 = pickle.load(open(os.path.join(HERE, "data", "cells_full_k4.pkl"), "rb"))
    mk = os.path.join(HERE, "data", "my_region_mass_k4.pkl")
    if os.path.exists(mk):
        mine = pickle.load(open(mk, "rb"))
    else:
        m, _ = P.cone_masses(cells4, 4); mine = {}
        for (G, h), mi in zip(cells4, m):
            mine[h] = mine.get(h, 0.0) + mi
        pickle.dump(mine, open(mk, "wb"))
    theirs = {bytes(p.tobytes()): float(x) for p, x in zip(d4["patterns"], d4["mass"])}
    common = set(mine) & set(theirs)
    diff = np.array([mine[h] - theirs[h] for h in common]); ref = np.array([theirs[h] for h in common])
    out["k4"] = dict(mine=len(mine), t192=len(theirs), matched=len(common), max_abs_diff=float(np.abs(diff).max()),
                     max_rel_diff_mass_gt_1e8=float((np.abs(diff) / ref)[ref > 1e-8].max()),
                     t192_sum_minus_1=float(d4["mass"].sum() - 1), N_delta_t192=coverage(d4["mass"]),
                     N_delta_mine=coverage(np.array(list(mine.values()))))
    print("k4", json.dumps(out["k4"]), flush=True)
    # (b) k=5 coverage from T192 masses
    d5 = np.load(os.path.join(HERE, "data", "t192", "k5_regionmass.npz"))
    N5 = coverage(d5["mass"])
    out["k5"] = dict(regions=int(len(d5["mass"])), complete=bool(d5["complete"]), sum_minus_1=float(d5["mass"].sum() - 1),
                     min_mass=float(d5["mass"].min()), N_delta=N5,
                     growth_4_5={k: N5[k] / out["k4"]["N_delta_mine"][k] for k in N5},
                     exp_entropy=float(np.exp(-(lambda m: (m * np.log(m)).sum())(d5["mass"][d5["mass"] > 0]))))
    print("k5", json.dumps(out["k5"]), flush=True)
    # (c) 200 spot checks: 100 uniform over regions + 100 mass-weighted
    rng = np.random.default_rng(11); M = d5["mass"]
    idx = np.concatenate([rng.choice(len(M), 100, replace=False),
                          rng.choice(len(M), 100, replace=False, p=np.clip(M, 0, None) / np.clip(M, 0, None).sum())])
    rows = []
    for n, i in enumerate(idx):
        pat = bytes(d5["patterns"][i].tobytes())
        mm, nc = region_mass_k5(pat, cells4, Wi, Wh)
        rows.append(dict(i=int(i), t192=float(M[i]), mine=mm, cells=nc))
        if n % 20 == 0:
            print(n, rows[-1], flush=True)
    a = np.array([r["t192"] for r in rows]); b = np.array([r["mine"] for r in rows])
    out["k5_spot"] = dict(n=len(rows), max_abs_diff=float(np.abs(a - b).max()),
                          max_rel_diff=float((np.abs(a - b) / np.maximum(a, 1e-300)).max()),
                          median_rel_diff=float(np.median(np.abs(a - b) / np.maximum(a, 1e-300))),
                          n_rel_gt_1e6=int((np.abs(a - b) / np.maximum(a, 1e-300) > 1e-6).sum()), rows=rows)
    print("k5_spot", {k: v for k, v in out["k5_spot"].items() if k != "rows"}, flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "compare_t192.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
