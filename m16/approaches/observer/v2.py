"""v2: v1 + rest inputs stratified into q equal-mass bins of N(0,1)|<c, assigned to the 7 rest positions by a
Latin-hypercube design over K=q anchors (each rest position visits each bin once). Internal noise per anchor =
within-bin variance propagated through the exact local map (a local, not global, linearisation)."""
import sys
import numpy as np
from scipy.special import ndtr
sys.path.insert(0, '/home/user/clean-neurons/m16')
from core import T, TRIPLES, forward_gates, logit_map, bin_moments, bin_medians
from quad import grid
from v1 import tie_levels


def build(kind, nv, ng, q, seed=0, drel=0.02):
    vq, ev, gq, eg = tie_levels(kind, nv, ng)
    rng = np.random.default_rng(seed)
    X = []; meta = []
    for ti, (p1, p2, p3) in enumerate(TRIPLES):
        rest = [p for p in range(T) if p not in (p1, p2, p3)]
        for i in range(nv):
            for j in range(ng):
                lhs = np.array([rng.permutation(q) for _ in range(7)])     # (7 positions, q anchors) -> bin
                v, gap = vq[i], gq[i, j]; h = drel * v / 2
                a, b, c = (v + gap, v + h, v - h) if kind == '23' else (v + h, v - h, v - h - gap)
                med = bin_medians(c, q)
                for k in range(q):
                    x = np.empty(T); x[[p1, p2, p3]] = a, b, c; x[rest] = med[lhs[:, k]]
                    X.append(x); meta.append((ti, i, j, k)); 
                    meta[-1] = (ti, i, j, k) + tuple(lhs[:, k])
    return np.array(X), np.array(meta), (vq, ev, gq, eg)


def estimate(model, nv=3, ng=3, q=4, nq=20, seed=0, drel=0.02):
    C, U, G, W = grid(nq, nq, nq)
    B = C + U; A = B + G
    BM, BV = bin_moments(C, q)           # (q, nodes)
    out = {}; evals = 0
    for kind in ('23', '12'):
        X, meta, (vq, ev, gq, eg) = build(kind, nv, ng, q, seed, drel)
        L, Gt = forward_gates(model, X); Amap = logit_map(model, Gt); evals += len(X)
        lev = (B + C) / 2 if kind == '23' else (A + B) / 2
        gap = G if kind == '23' else U
        iv = np.clip(np.searchsorted(ev, lev) - 1, 0, nv - 1)
        jg = np.zeros_like(iv)
        for i in range(nv):
            m = iv == i
            jg[m] = np.clip(np.searchsorted(eg[i], gap[m]) - 1, 0, ng - 1)
        cell = iv * ng + jg
        masks = [cell == c for c in range(nv * ng)]
        errs = np.zeros(len(TRIPLES))
        for r in range(len(X)):
            ti, i, j, k = meta[r][:4]; bins = meta[r][4:]
            p1, p2, p3 = TRIPLES[ti]
            comp = p3 if kind == '23' else p1
            m = masks[i * ng + j]
            rest = [p for p in range(T) if p not in (p1, p2, p3)]
            d = Amap[r, p2] - Amap[r, comp]
            dr = d[rest]
            mu = d[p1] * A[m] + d[p2] * B[m] + d[p3] * C[m] + (dr[:, None] * BM[bins][:, m]).sum(0)
            sd = np.sqrt((dr[:, None] ** 2 * BV[bins][:, m]).sum(0) + 1e-12)
            errs[ti] += (W[m] * ndtr(-mu / sd)).sum() / 720 / q
        out[kind] = errs
    return out, evals


if __name__ == "__main__":
    from common import Wi, Wh, Wo
    nv, ng, q = (int(a) for a in sys.argv[1:4]); drel = float(sys.argv[4])
    o, ev = estimate((Wi, Wh, Wo), nv, ng, q, drel=drel)
    print(nv, ng, q, drel, 'evals', ev, 'err23 %.5f err12 %.5f' % (o['23'].sum(), o['12'].sum()))
    np.savez(f'v2_{nv}{ng}{q}.npz', **o)
