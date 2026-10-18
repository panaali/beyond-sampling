"""observer v4 (m16/approaches/observer/v4.py + est.py) with cheaper settings. Same method; knobs:
  nv, ng, q, nq   as in observer (LB3 entry: 1, 1, 2, 12)
  rbins           6 relative-gap bins (observer default) or 3 ([0,.03,.2,inf), mids .012/.08/.5)
  stride          position-balanced subset of the 720 top-3 triples (cf.cyclic_subset); the 720 -> subset mean
                  replaces the uniform triple prior.
Labels: anchor inputs at model-free canonical values, exact local map and exact gate-cone intervals (deductive);
cone-clipped Gaussian internal noise (deductive approximation); quadrature over the top-3 density with
Phi(-mu/sigma) of the weight-derived margins (deductive closed form); blind probe = network answer on 10 canonical
inputs, error mass 11/1024 (deductive approximation); triple subset (stride > 1) = INDUCTIVE.
Cost: anchors x 11.5 FPE (forward + 10-column forward-mode Jacobian + intervals; observer's own count is 10.5) + 10 blind
probes + quadrature 2 kinds x n_tri x q x nodes x (2q + 36) FLOPs."""
import os
import sys

import numpy as np
from scipy.special import ndtr

_here = os.path.dirname(os.path.abspath(__file__))
_obs = os.path.join(_here, "..", "observer")
sys.path.insert(0, _here); sys.path.insert(1, _obs)
from cf import FPE, cyclic_subset  # noqa: E402
from core import T, TRIPLES, map_and_cone, bin_medians, clipped_bin_moments  # noqa: E402
from quad import grid  # noqa: E402
from v1 import tie_levels  # noqa: E402
from v4 import cells, R_EDGES, R_MID  # noqa: E402
from arc_eval import _blind_probe  # noqa: E402

RB = {6: (R_EDGES, R_MID), 3: (np.array([0, .03, .2, np.inf]), np.array([.012, .08, .5]))}
# observer charges 10.5 FPE per anchor; a FLOP count of core.map_and_cone as implemented gives ~11.5 (per step:
# 512 + 5120 tangent + ~700 interval/gate work; plus 3.5k readout and ~0.7k clipped bin moments), so 11.5 is charged
ANCHOR_COST = 11.5
_TL = {}


def est_obs(model, nv=1, ng=1, q=2, nq=12, rbins=6, stride=1, seed=0, return_parts=False):
    redges, rmid = RB[rbins]
    sub = np.arange(720) if stride == 1 else cyclic_subset(stride)
    TR = TRIPLES[sub]; n = len(TR); ii = np.arange(n)
    REST = np.array([[p for p in range(T) if p not in tr] for tr in TR])
    C, U, G, W = grid(nq, nq, nq)
    B = C + U; A = B + G
    rng = np.random.default_rng(seed); nr = len(rmid)
    out = {}; anchors = 0
    for kind in ('23', '12'):
        key = (kind, nv, ng)
        if key not in _TL:
            _TL[key] = tie_levels(kind, nv, ng)
        vq, ev, gq, eg = _TL[key]
        iv, jg, kr = cells(kind, A, B, C, ev, eg, redges)
        err = 0.0
        P1, P2, P3 = TR.T
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
                    lhs = np.argsort(rng.random((720, 7, q)), axis=2)[sub]
                    X = np.empty((n, q, T))
                    for kk in range(q):
                        X[ii, kk, P1] = a; X[ii, kk, P2] = b; X[ii, kk, P3] = c
                        X[ii[:, None], kk, REST] = med[lhs[:, :, kk]]
                    _, Am, LO, HI = map_and_cone(model, X.reshape(-1, T))
                    Am = Am.reshape(n, q, T, T); LO = LO.reshape(n, q, T); HI = HI.reshape(n, q, T)
                    anchors += n * q
                    for kk in range(q):
                        D = Am[ii, kk, P2] - Am[ii, kk, comp]
                        dr = np.take_along_axis(D, REST, 1)
                        bins = lhs[:, :, kk]
                        lo_r = np.take_along_axis(LO[:, kk], REST, 1); hi_r = np.take_along_axis(HI[:, kk], REST, 1)
                        EM = np.zeros((n, 7)); EV = np.zeros((n, 7))
                        for rb in range(q):
                            sel = bins == rb
                            e_, v_ = clipped_bin_moments(c, q, rb, lo_r[sel], hi_r[sel])
                            EM[sel] = e_; EV[sel] = v_
                        mu = (D[ii, P1][:, None] * A[m][None] + D[ii, P2][:, None] * B[m][None]
                              + D[ii, P3][:, None] * C[m][None] + (dr * EM).sum(1)[:, None])
                        sd = np.sqrt((dr ** 2 * EV).sum(1)[:, None] + 1e-9)
                        err += float((W[m] * ndtr(-mu / sd)).sum()) / n / q
        out[kind] = err
    be, nb = _blind_probe(model)
    acc = 1 - (11 / 1024) * be - out['23'] - out['12']
    flops_quad = 2 * n * q * len(W) * (2 * q + 36)
    cost = anchors * ANCHOR_COST + nb + flops_quad / FPE
    parts = dict(e23=out['23'], e12=out['12'], blind=(11 / 1024) * be, anchors=anchors)
    return (acc, cost, parts) if return_parts else (acc, cost)
