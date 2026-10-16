"""v1: tie-anchored psychometric observer.
For each position triple and each tie type (2nd~3rd, 1st~2nd), anchors are placed AT the tie on a small grid
of (tie level v, off-tie gap) values, with the 7 'rest' inputs at truncated-normal quantiles in K arrangements.
The exact local linear map at each anchor gives the competitor margin as a linear function of the inputs;
the rest inputs are treated as iid truncated normals -> Gaussian 'internal noise'. Accuracy loss =
quadrature of Phi(-mu/sigma) against the closed-form top-3 density."""
import sys
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
sys.path.insert(0, '/home/user/clean-neurons/m16')
from core import T, TRIPLES, forward_gates, logit_map, trunc_moments, trunc_quantiles, phi
from quad import grid


def tie_levels(kind, nv, ng):
    """Representative (v, gap) values for the tie-conditional density.
    kind '23': b=c=v, a=v+gap. density ~ phi(v+g) phi(v)^2 Phi(v)^7.  kind '12': a=b=v, c=v-gap,
    density ~ phi(v)^2 phi(v-g) Phi(v-g)^7. Restricted to v>0. Returns medians of equal-mass bins."""
    v = np.linspace(0.001, 4.5, 900); g = np.linspace(0.0005, 4, 800)
    V, Gg = np.meshgrid(v, g, indexing='ij')
    if kind == '23':
        D = phi(V + Gg) * phi(V) ** 2 * ndtr(V) ** 7
    else:
        D = phi(V) ** 2 * phi(V - Gg) * ndtr(V - Gg) ** 7
    pv = D.sum(1); cv = np.cumsum(pv) / pv.sum()
    vb = np.searchsorted(cv, np.linspace(0, 1, nv + 1)[1:-1])       # bin edges (indices)
    vq = v[np.searchsorted(cv, (np.arange(nv) + 0.5) / nv)]
    edges_v = np.concatenate([[0], v[vb], [np.inf]])
    gq = np.zeros((nv, ng)); edges_g = np.zeros((nv, ng + 1))
    for i in range(nv):
        m = (v >= edges_v[i]) & (v < edges_v[i + 1])
        pg = D[m].sum(0); cg = np.cumsum(pg) / pg.sum()
        gq[i] = g[np.searchsorted(cg, (np.arange(ng) + 0.5) / ng)]
        edges_g[i] = np.concatenate([[0], g[np.searchsorted(cg, np.linspace(0, 1, ng + 1)[1:-1])], [np.inf]])
    return vq, edges_v, gq, edges_g


def build_anchors(kind, nv, ng, K, seed=0, drel=0.02):
    vq, ev, gq, eg = tie_levels(kind, nv, ng)
    rng = np.random.default_rng(seed)
    X = []; meta = []
    for ti, (p1, p2, p3) in enumerate(TRIPLES):
        rest = [p for p in range(T) if p not in (p1, p2, p3)]
        perms = [rng.permutation(7) for _ in range(K)]
        for i in range(nv):
            for j in range(ng):
                v, gap = vq[i], gq[i, j]
                h = drel * v / 2
                if kind == '23':
                    a, b, c = v + gap, v + h, v - h
                else:
                    a, b, c = v + h, v - h, v - h - gap
                rq = trunc_quantiles(c, 7)
                for k in range(K):
                    x = np.empty(T); x[[p1, p2, p3]] = a, b, c; x[rest] = rq[perms[k]]
                    X.append(x); meta.append((ti, i, j, k))
    return np.array(X), np.array(meta), (vq, ev, gq, eg)


def estimate(model, nv=3, ng=3, K=2, nq=20, seed=0, drel=0.02):
    C, U, G, W = grid(nq, nq, nq)
    B = C + U; A = B + G
    mc, vc = trunc_moments(C)
    out = {}
    evals = 0
    for kind in ('23', '12'):
        X, meta, (vq, ev, gq, eg) = build_anchors(kind, nv, ng, K, seed, drel)
        L, Gt = forward_gates(model, X); Amap = logit_map(model, Gt); evals += len(X)
        # quadrature node -> anchor cell
        lev = (B + C) / 2 if kind == '23' else (A + B) / 2
        gap = G if kind == '23' else U
        iv = np.clip(np.searchsorted(ev, lev) - 1, 0, nv - 1)
        jg = np.zeros_like(iv)
        for i in range(nv):
            m = iv == i
            jg[m] = np.clip(np.searchsorted(eg[i], gap[m]) - 1, 0, ng - 1)
        cell = iv * ng + jg
        errs = np.zeros(len(TRIPLES))
        for r in range(len(X)):
            ti, i, j, k = meta[r]
            p1, p2, p3 = TRIPLES[ti]
            comp = p3 if kind == '23' else p1
            m = cell == i * ng + j
            if not m.any():
                continue
            rest = [p for p in range(T) if p not in (p1, p2, p3)]
            d = Amap[r, p2] - Amap[r, comp]
            mu = d[p1] * A[m] + d[p2] * B[m] + d[p3] * C[m] + mc[m] * d[rest].sum()
            sd = np.sqrt(vc[m] * (d[rest] ** 2).sum() + 1e-12)
            errs[ti] += (W[m] * ndtr(-mu / sd)).sum() / 720 / K
        out[kind] = errs
    return out, evals


if __name__ == "__main__":
    from common import Wi, Wh, Wo
    nv, ng, K = (int(a) for a in sys.argv[1:4])
    drel = float(sys.argv[4])
    o, ev = estimate((Wi, Wh, Wo), nv, ng, K, drel=drel)
    print(nv, ng, K, 'evals', ev, 'err23', o['23'].sum(), 'err12', o['12'].sum())
    np.savez(f'v1_{nv}{ng}{K}.npz', **o)
