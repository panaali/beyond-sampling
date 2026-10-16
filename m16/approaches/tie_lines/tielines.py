"""Tie-direction line sampling (conditional MC) for the accuracy of 2nd-argmax ReLU RNNs.

Construction (unbiased for any weights):
  Draw x ~ N(0, I). Let f(x) = {i, j} be a selected pair of positions (e.g. the ranks-2,3 pair).
  Write x = u + t d with d = (e_i - e_j)/sqrt(2), t = d.x ~ N(0,1) independent of u.
  The set of t with f(u + t d) = {i,j} is an explicit window W(u) (model-free); e.g. for the 2-3 pair,
  |t| < tau = sqrt(2) min(o1 - m, m - o2), m = (x_i+x_j)/2, o1 > o2 the two largest other coordinates.
  Z = P_t(correct, t in W | u) / P_t(t in W | u)  =  E[correct | u, f = {i,j}].
  E[Z] = sum_ij E[ 1{f=ij} A_ij(u_ij)/B_ij(u_ij) ] = sum_ij E[A_ij] = acc  (since E[1{f=ij}|u_ij] = B_ij).
  The numerator is computed exactly by walking the network's linear pieces inside W only.
  Any u-measurable rule may decide to use the plain indicator g(x) instead of Z (E[g|u,f]=Z): unbiased.
"""
import numpy as np
from scipy.special import ndtr

SQ2 = np.sqrt(2.0)


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def correct(model, x):
    return forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]


def _fwd_tan(model, x, dx):
    """Batched forward with tangent. Returns min positive root step r (distance to next gate change along dx,
    in units of the line parameter), logits, logit slopes."""
    Wi, Wh, Wo = model
    N, T = x.shape
    h = np.zeros((N, Wh.shape[0])); dh = np.zeros_like(h)
    rmin = np.full(N, np.inf)
    for t in range(T):
        z = h @ Wh.T + x[:, t:t + 1] * Wi[None]
        dz = dh @ Wh.T + dx[:, t:t + 1] * Wi[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -z / dz
        r = np.where((r > 1e-12) & np.isfinite(r), r, np.inf)
        rmin = np.minimum(rmin, r.min(1))
        g = z > 0
        h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
    return rmin, h @ Wo.T, dh @ Wo.T


def line_mass(model, base, d, a, b, tpos, tneg, max_iter=400):
    """Exact N(0,1) mass of {t in [a,b] : argmax logits(base + t d) == target(t)}, target = tpos if t>0 else tneg.
    Batched over lines. Returns (mass, pieces_per_line)."""
    N = len(base)
    s = a.copy(); mass = np.zeros(N); pieces = np.zeros(N, int)
    active = s < b
    eps = 1e-9
    it = 0
    while active.any() and it < max_iter:
        it += 1
        idx = np.nonzero(active)[0]
        si = s[idx]
        x = base[idx] + (si + eps)[:, None] * d[idx]
        r, lg, dlg = _fwd_tan(model, x, d[idx])
        pieces[idx] += 1
        lg = lg - dlg * eps
        nxt = np.minimum(b[idx], si + eps + r)
        nxt = np.where((si < 0) & (nxt > 0), 0.0, nxt)          # target switches at t = 0
        nxt = np.maximum(nxt, si + 1e-12)
        k = np.where(si + eps > 0, tpos[idx], tneg[idx])
        # interval where logit k >= all others on [si, nxt]: lg_k - lg_l + (dlg_k - dlg_l)(t - si) >= 0
        rows = np.arange(len(idx))
        D0 = lg[rows, k][:, None] - lg; D1 = dlg[rows, k][:, None] - dlg
        D0[rows, k] = 1.0; D1[rows, k] = 0.0
        L = si.copy(); R = nxt.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            root = si[:, None] - D0 / D1
        posD = D1 > 0; negD = D1 < 0; zer = D1 == 0
        L = np.maximum(L, np.where(posD, root, -np.inf).max(1))
        R = np.minimum(R, np.where(negD, root, np.inf).min(1))
        dead = (zer & (D0 < 0)).any(1)
        ok = (R > L) & ~dead
        mass[idx] += np.where(ok, ndtr(R) - ndtr(L), 0.0)
        s[idx] = nxt
        active[idx] = nxt < b[idx]
    return mass, pieces


def pair_lines(x, kind):
    """For each x, the selected pair, direction, window [-tau, tau] and targets.
    kind '23': ranks 2,3 pair.  kind '12': ranks 1,2 pair."""
    N, T = x.shape
    o = np.argsort(-x, 1); xs = np.take_along_axis(x, o, 1)
    rows = np.arange(N)
    if kind == "23":
        i, j = o[:, 1], o[:, 2]
        m = (xs[:, 1] + xs[:, 2]) / 2
        tau = SQ2 * np.minimum(xs[:, 0] - m, m - xs[:, 3])
        tpos, tneg = i, j                        # t>0 -> x_i bigger -> x_i is rank 2
    else:
        i, j = o[:, 0], o[:, 1]
        m = (xs[:, 0] + xs[:, 1]) / 2
        tau = SQ2 * (m - xs[:, 2])
        tpos, tneg = j, i                        # t>0 -> x_i bigger -> x_j is rank 2
    d = np.zeros((N, T)); d[rows, i] = 1 / SQ2; d[rows, j] = -1 / SQ2
    t0 = (x[rows, i] - x[rows, j]) / SQ2
    base = x - t0[:, None] * d
    return base, d, tau, tpos, tneg, m, xs


def Z_lines(model, x, kind="23"):
    base, d, tau, tpos, tneg, m, xs = pair_lines(x, kind)
    A, pieces = line_mass(model, base, d, -tau, tau, tpos, tneg)
    B = ndtr(tau) - ndtr(-tau)
    return A / B, pieces


# ------------------------------------------------------------------ narrow-window hybrid estimator
def sel23(xs, c23):
    """xs sorted descending. Selected iff 0 < gap23 < c23 * mid23."""
    mid = (xs[:, 1] + xs[:, 2]) / 2
    return (mid > 0) & (xs[:, 1] - xs[:, 2] < c23 * mid)


def hybrid_terms(model, x, c23=0.05, c12=0.05):
    """Per-input unbiased terms and costs (forward-pass equivalents; a tangent piece counts 2).
    f(x) = 23-pair if sel23, elif 12-pair if sel12, else none (plain indicator)."""
    N, T = x.shape
    o = np.argsort(-x, 1); xs = np.take_along_axis(x, o, 1); rows = np.arange(N)
    s23 = sel23(xs, c23)
    m12 = (xs[:, 0] + xs[:, 1]) / 2
    s12 = ~s23 & (m12 > 0) & (xs[:, 0] - xs[:, 1] < c12 * m12)
    val = np.zeros(N); cost = np.zeros(N)
    rest = ~(s23 | s12)
    if rest.any():
        val[rest] = correct(model, x[rest]); cost[rest] = 1
    if s23.any():
        X = x[s23]; O = o[s23]; S = xs[s23]; r = np.arange(len(X))
        i, j = O[:, 1], O[:, 2]; m = (S[:, 1] + S[:, 2]) / 2
        tau = np.minimum(SQ2 * np.minimum(S[:, 0] - m, m - S[:, 3]), c23 * m / SQ2)
        d = np.zeros_like(X); d[r, i] = 1 / SQ2; d[r, j] = -1 / SQ2
        base = X.copy(); base[r, i] = m; base[r, j] = m
        A, p = line_mass(model, base, d, -tau, tau, i, j)
        val[s23] = A / (ndtr(tau) - ndtr(-tau)); cost[s23] = 2 * p
    if s12.any():
        X = x[s12]; O = o[s12]; S = xs[s12]; r = np.arange(len(X))
        i, j = O[:, 0], O[:, 1]; m = (S[:, 0] + S[:, 1]) / 2; o1 = S[:, 2]
        # window: pair stays top-2 (s < m - o1), gap12 = 2s < c12 m, and NOT sel23 along the line:
        # sel23 <=> (m - s + o1)/2 > 0 and (m - s - o1) < c23 (m - s + o1)/2  <=> s > s_star (if mid>0)
        s_star = (m - o1 - c23 * (m + o1) / 2) / (1 - c23 / 2)
        # mid23>0 needs s < m + o1; if the c23 condition holds where mid23 <= 0 it is not selected: handle by
        # limiting s < s_star unless s_star >= m + o1 (then sel23 never holds in range)
        lim = np.minimum(m - o1, c12 * m / 2)
        lim = np.where(s_star < m + o1, np.minimum(lim, np.maximum(s_star, 0)), lim)
        tau = SQ2 * lim
        d = np.zeros_like(X); d[r, i] = 1 / SQ2; d[r, j] = -1 / SQ2
        base = X.copy(); base[r, i] = m; base[r, j] = m
        A, p = line_mass(model, base, d, -tau, tau, j, i)
        val[s12] = A / (ndtr(tau) - ndtr(-tau)); cost[s12] = 2 * p
    return val, cost, s23, s12


# ------------------------------------------------------------------ stratum-restricted tie lines
def _stratum_vals(v1, v2, v3, edges):
    """stratum as in m16/stratified.py from the top-3 values (vectorized)."""
    blind = v2 <= 0
    g = np.minimum((v1 - v2) / np.abs(v1), (v2 - v3) / np.maximum(np.abs(v2), 1e-12))
    b = 1 + np.clip(np.searchsorted(edges, g, side="right") - 1, 0, len(edges) - 2)
    return np.where(blind, 0, b), (v1 - v2) / np.abs(v1) <= (v2 - v3) / np.maximum(np.abs(v2), 1e-12)


def stratum_lines(x, edges):
    """For x (not blind), the tie line of its min-gap pair and the window {|t| in [lo,hi]} on which the
    stratum AND the min-gap pair kind are unchanged (model-free, by bisection on monotone conditions).
    Returns base, d, lo, hi, tpos, tneg, h, kind12."""
    N, T = x.shape
    o = np.argsort(-x, 1); xs = np.take_along_axis(x, o, 1); r = np.arange(N)
    h, k12 = _stratum_vals(xs[:, 0], xs[:, 1], xs[:, 2], edges)
    i = np.where(k12, o[:, 0], o[:, 1]); j = np.where(k12, o[:, 1], o[:, 2])
    m = (x[r, i] + x[r, j]) / 2
    a = np.where(k12, xs[:, 2], xs[:, 0])          # the other top value (rank 3 for 12-kind, rank 1 for 23-kind)
    b4 = np.where(k12, xs[:, 2], xs[:, 3])          # value just below the pair
    smax = np.where(k12, m - xs[:, 2], np.minimum(xs[:, 0] - m, m - xs[:, 3])) * SQ2   # pair keeps its ranks

    def ok(s):
        hi_v = m + s / SQ2; lo_v = m - s / SQ2
        v1 = np.where(k12, hi_v, a); v2 = np.where(k12, lo_v, hi_v); v3 = np.where(k12, b4, lo_v)
        hh, kk = _stratum_vals(v1, v2, v3, edges)
        return (hh == h) & (kk == k12) & (s < smax)

    s0 = np.abs(x[r, i] - x[r, j]) / SQ2
    assert ok(s0).all()
    def bis(lo_, hi_, inside_is_lo):
        for _ in range(60):
            mid = (lo_ + hi_) / 2; good = ok(mid)
            if inside_is_lo:
                lo_ = np.where(good, mid, lo_); hi_ = np.where(good, hi_, mid)
            else:
                hi_ = np.where(good, mid, hi_); lo_ = np.where(good, lo_, mid)
        return lo_, hi_
    up, _ = bis(s0.copy(), smax.copy(), True)
    zero_ok = ok(np.zeros(N))
    _, dn = bis(np.zeros(N), s0.copy(), False)
    dn = np.where(zero_ok, 0.0, dn)
    d = np.zeros((N, T)); d[r, i] = 1 / SQ2; d[r, j] = -1 / SQ2
    base = x.copy(); base[r, i] = m; base[r, j] = m
    tpos = np.where(k12, j, i); tneg = np.where(k12, i, j)
    return base, d, dn, up, tpos, tneg, h, k12


def Z_stratum(model, x, edges):
    base, d, lo, hi, tp, tn, h, k = stratum_lines(x, edges)
    A1, p1 = line_mass(model, base, d, lo, hi, tp, tn)
    A2, p2 = line_mass(model, base, d, -hi, -lo, tp, tn)
    B = 2 * (ndtr(hi) - ndtr(lo))
    return (A1 + A2) / B, p1 + p2
