"""v4 = v3 with cone-aware internal noise: each rest input acts through d_t*clip(x_t, lo_t, hi_t), where
[lo_t, hi_t] is the exact interval on which the anchor's gates (hence its linear map) stay valid. Optionally the
top-3 inputs are clipped the same way (clip_top).
Original v3 doc:
v3: psychometric observer with anchors along the near-tie gap axis.
For each triple and each tie kind ('23': competitor = rank3, '12': competitor = rank1):
  cells = (tie level v: nv bins) x (other gap: ng bins) x (relative near-tie gap r: len(R_EDGES)-1 bins)
  in each cell q anchors whose 7 rest inputs follow a Latin-hypercube design over q equal-mass bins of
  N(0,1)|<c. At each anchor the network's exact local linear map gives the competitor margin; inside the cell
  the margin is extrapolated linearly in (a,b,c) and the rest inputs contribute within-bin Gaussian noise.
  err = sum over nodes of a deterministic quadrature of the top-3 density x Phi(-mu/sigma)."""
import sys
import numpy as np
from scipy.special import ndtr
sys.path.insert(0, '/home/user/clean-neurons/m16')
from core import T, TRIPLES, map_and_cone, bin_moments, bin_medians, clipped_bin_moments
from quad import grid
from v1 import tie_levels

R_EDGES = np.array([0, .01, .03, .08, .2, .5, np.inf])
R_MID = np.array([.005, .018, .05, .13, .32, .8])
REST = np.array([[p for p in range(T) if p not in tr] for tr in TRIPLES])     # (720,7)


def cells(kind, A, B, C, ev, eg, redges):
    lev = B if kind == '23' else A          # level = the larger member of the tied pair
    og = (A - B) if kind == '23' else (B - C)
    r = (B - C) / B if kind == '23' else (A - B) / A
    nv, ng = len(ev) - 1, eg.shape[1] - 1
    iv = np.clip(np.searchsorted(ev, lev) - 1, 0, nv - 1)
    jg = np.zeros_like(iv)
    for i in range(nv):
        m = iv == i
        jg[m] = np.clip(np.searchsorted(eg[i], og[m]) - 1, 0, ng - 1)
    kr = np.clip(np.searchsorted(redges, r, side='right') - 1, 0, len(redges) - 2)
    return iv, jg, kr


def estimate(model, nv=2, ng=2, q=2, nq=16, seed=0, redges=R_EDGES, rmid=R_MID, noise=1.0, clip_top=False):
    C, U, G, W = grid(nq, nq, nq)
    B = C + U; A = B + G
    BM, BV = bin_moments(C, q)                          # (q, nodes)
    nr = len(rmid); rng = np.random.default_rng(seed)
    out = {}; evals = 0
    for kind in ('23', '12'):
        vq, ev, gq, eg = tie_levels(kind, nv, ng)
        iv, jg, kr = cells(kind, A, B, C, ev, eg, redges)
        errs = np.zeros(len(TRIPLES)); node_err = np.zeros(len(W))
        P1, P2, P3 = TRIPLES.T
        comp = P3 if kind == '23' else P1
        for i in range(nv):
            for j in range(ng):
                for k in range(nr):
                    m = (iv == i) & (jg == j) & (kr == k)
                    if not m.any():
                        continue
                    v, og, r = vq[i], gq[i, j], rmid[k]
                    if kind == '23':
                        a, b, c = v + og, v, v * (1 - r)
                    else:
                        a, b = v, v * (1 - r); c = b - og
                    med = bin_medians(c, q)
                    lhs = np.argsort(rng.random((len(TRIPLES), 7, q)), axis=2)   # (720,7,q) bin per anchor
                    X = np.empty((len(TRIPLES), q, T))
                    X[np.arange(720)[:, None], :, TRIPLES[:, :1]] = 0  # placeholder
                    for kk in range(q):
                        X[np.arange(720), kk, P1] = a; X[np.arange(720), kk, P2] = b; X[np.arange(720), kk, P3] = c
                        X[np.arange(720)[:, None], kk, REST] = med[lhs[:, :, kk]]
                    Xf = X.reshape(-1, T)
                    _, Am, LO, HI = map_and_cone(model, Xf); Am = Am.reshape(720, q, T, T); LO = LO.reshape(720, q, T); HI = HI.reshape(720, q, T); evals += len(Xf)
                    for kk in range(q):
                        D = Am[np.arange(720), kk, P2] - Am[np.arange(720), kk, comp]        # (720,10)
                        dr = np.take_along_axis(D, REST, 1)                                  # (720,7)
                        bins = lhs[:, :, kk]                                                  # (720,7)
                        lo_r = np.take_along_axis(LO[:, kk], REST, 1); hi_r = np.take_along_axis(HI[:, kk], REST, 1)
                        EM = np.zeros((720, 7)); EV = np.zeros((720, 7))
                        for rb in range(q):
                            sel = bins == rb
                            e_, v_ = clipped_bin_moments(c, q, rb, lo_r[sel], hi_r[sel])
                            EM[sel] = e_; EV[sel] = v_
                        ii = np.arange(720)
                        if clip_top:
                            tA = np.clip(A[m][None], LO[ii, kk, P1][:, None], HI[ii, kk, P1][:, None])
                            tB = np.clip(B[m][None], LO[ii, kk, P2][:, None], HI[ii, kk, P2][:, None])
                            tC = np.clip(C[m][None], LO[ii, kk, P3][:, None], HI[ii, kk, P3][:, None])
                        else:
                            tA, tB, tC = A[m][None], B[m][None], C[m][None]
                        mu = (D[ii, P1][:, None] * tA + D[ii, P2][:, None] * tB + D[ii, P3][:, None] * tC
                              + (dr * EM).sum(1)[:, None])
                        sd = np.sqrt(noise * (dr ** 2 * EV).sum(1)[:, None] + 1e-9)
                        pe = W[m] * ndtr(-mu / sd) / 720 / q
                        errs += pe.sum(1); node_err[m] += pe.sum(0)
        out[kind] = errs; out[kind + '_node'] = node_err
    return out, evals


if __name__ == "__main__":
    from common import Wi, Wh, Wo
    nv, ng, q = (int(a) for a in sys.argv[1:4])
    o, ev = estimate((Wi, Wh, Wo), nv, ng, q)
    print(nv, ng, q, 'evals', ev, 'err23 %.5f err12 %.5f' % (o['23'].sum(), o['12'].sum()))
