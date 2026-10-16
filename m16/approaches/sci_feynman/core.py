"""Core of the sci_feynman estimator: exact model-free partition of input space into tie segments,
exact walks along each segment, deterministic node sets.

Partition (all exact, model-free; same construction as m16/approaches/tube, re-derived here):
  B   = {2nd largest <= 0}                     (mass 11/1024)
  R23 = B^c & {v2 - v3 < v1 - v2}: each input lies on one segment x_a = u + s/2, x_b = u - s/2 (a<b ranks 2,3),
        other coords fixed; face point (s=0): x_p1 = o1 > u, 7 bath < u.  |s| < S23 = min(2(o1-u)/3, 2(u-o2)),
        and B^c requires u + |s|/2 > 0.
  R12 = B^c & {v1 - v2 <= v2 - v3}: segment of the top pair around w, 8 bath < w; |s| < min(2(w-o3)/3, 2w).
  dx_a dx_b = du ds, phi(u+s/2)phi(u-s/2) = phi(u)^2 exp(-s^2/4), so with t = s/sqrt2:
  P(err & R23) = sum_{360 configs} Z23 * E_face[ 2 sqrt(pi) * N(0,1)-mass of wrong t on the segment ]
  with Z23 = int phi^2 (1-Phi) Phi^7 du (per config) and the face law: u ~ prop. phi^2(1-Phi)Phi^7,
  o1 ~ N|>u, bath iid N|<u.  R12 likewise with Z12 = int phi^2 Phi^8, 45 configs.
"""
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm

T = 10
SQ2 = np.sqrt(2.0)
_v = np.linspace(-8, 8, 400001)
_d23 = norm.pdf(_v) ** 2 * norm.sf(_v) * norm.cdf(_v) ** 7
_d12 = norm.pdf(_v) ** 2 * norm.cdf(_v) ** 8
Z23 = np.trapezoid(_d23, _v); Z12 = np.trapezoid(_d12, _v)
_c23 = np.cumsum(_d23); _c23 = (_c23 - _c23[0]) / (_c23[-1] - _c23[0])
_c12 = np.cumsum(_d12); _c12 = (_c12 - _c12[0]) / (_c12[-1] - _c12[0])


def q23(p):
    return np.interp(p, _c23, _v)


def q12(p):
    return np.interp(p, _c12, _v)


def tn_above(u, lo):      # quantile u of N(0,1) | > lo
    p = ndtr(lo); return ndtri(np.clip(p + u * (1 - p), 1e-300, 1 - 1e-16))


def tn_below(u, hi):      # quantile u of N(0,1) | < hi
    return ndtri(np.clip(u * ndtr(hi), 1e-300, 1 - 1e-16))


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def walk_mass(model, y, a, b, tau, tex, top=False, eps=1e-9, max_iter=600):
    """Exact N(0,1) mass (in t) of the misclassified part of {y + t d : tex <= |t| <= tau}, d=(e_a-e_b)/sqrt2.
    Target is a for t>0, b for t<0 (a,b arrays). Returns (mass, cost) with cost in forward-pass equivalents:
    each linear cell costs value+tangent recomputed from the first time step whose gate changed (incremental)."""
    Wi, Wh, Wo = model
    N = len(y); H = Wh.shape[0]
    rows_all = np.arange(N)
    d = np.zeros_like(y); d[rows_all, a] = 1 / SQ2; d[rows_all, b] = -1 / SQ2
    first = np.minimum(a, b)                        # steps before min(a,b) never change along the segment
    mass = np.zeros(N); cost = (first / T).astype(float)   # shared prefix once per segment
    s = np.where(tau > tex, -tau, tau).astype(float)
    act = s < tau
    prevG = None; Gs = np.zeros((N, T, H), bool); started = np.zeros(N, bool)
    for _ in range(max_iter):
        if not act.any():
            break
        idx = np.nonzero(act)[0]
        X0, D, S = y[idx], d[idx], s[idx] + eps
        h = np.zeros((len(idx), H)); dh = np.zeros_like(h)
        nxt = tau[idx].copy()
        G = np.zeros((len(idx), T, H), bool)
        for t in range(T):
            z = h @ Wh.T + (X0[:, t] + S * D[:, t])[:, None] * Wi[None]
            dz = dh @ Wh.T + D[:, t:t + 1] * Wi[None]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = S[:, None] - z / dz
            r = np.where((r > S[:, None] + 1e-12) & np.isfinite(r), r, np.inf).min(1)
            nxt = np.minimum(nxt, r)
            g = z > 0; G[:, t] = g
            h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        # cost: recompute from first step whose gate differs from the previous cell (first cell: from min(a,b))
        ch = (G != Gs[idx]).any(2)
        tf = np.where(ch.any(1), ch.argmax(1), T)
        tf = np.where(started[idx], tf, first[idx])
        cost[idx] += 2.0 * (T - tf) / T
        Gs[idx] = G; started[idx] = True
        # jump the excluded zone (-tex, tex): cells there are not needed
        lg = h @ Wo.T; dlg = dh @ Wo.T
        lg = lg - dlg * eps; S0 = s[idx]
        aa = S0.copy(); rows = np.arange(len(idx))
        hi_c = nxt.copy()
        for _ in range(80):
            live = aa < hi_c
            if not live.any():
                break
            val = lg + dlg * (aa - S0)[:, None]
            k = np.argmax(val + dlg * 1e-7, 1)
            vk, dk = val[rows, k], dlg[rows, k]
            with np.errstate(divide="ignore", invalid="ignore"):
                c = aa[:, None] + (vk[:, None] - val) / (dlg - dk[:, None])
            c = np.where((dlg > dk[:, None]) & (c > aa[:, None] + 1e-13), c, np.inf).min(1)
            bb = np.minimum(hi_c, c)
            ii = idx[live]
            lo_, hi_ = aa[live], bb[live]; kk = k[live]
            # piece may straddle 0: split
            lp, hp = np.clip(lo_, 0, None), np.clip(hi_, 0, None)
            ln, hn = np.clip(lo_, None, 0), np.clip(hi_, None, 0)
            tx = tex[ii]
            lp = np.maximum(lp, tx); hn = np.minimum(hn, -tx)
            tp, tn = (b[ii], a[ii]) if top else (a[ii], b[ii])   # top pair: t>0 makes a the max, b the 2nd
            add = np.where((kk != tp) & (hp > lp), ndtr(hp) - ndtr(lp), 0.0) + \
                np.where((kk != tn) & (hn > ln), ndtr(hn) - ndtr(ln), 0.0)
            np.add.at(mass, ii, add)
            aa = np.where(live, bb, aa)
        snew = nxt
        # skip the excluded zone
        tx = tex[idx]
        snew = np.where((snew > -tx) & (snew < tx), tx, snew)
        s[idx] = snew
        act = s < tau
    return mass, cost
