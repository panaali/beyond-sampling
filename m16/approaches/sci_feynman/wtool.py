"""Exact band-width measurement on the 2-3 tie surface of M_16,10-type RNNs.

A surface point: positions a < b with x_a = x_b = v, exactly one other input above v (the max, at p1) and seven
below (so the tie sits at ranks 2-3). Normal line: x(t) = y + t d, d = (e_a - e_b)/sqrt2, so for t > 0 the
target (2nd largest) is a, for t < 0 it is b. The segment is kept while the tie stays at ranks 2-3:
|t| < tau = sqrt2 * min(o1 - v, v - o2).

Along the line the network is exactly piecewise linear (no biases, ReLU). `walk` records every
(start, end, argmax) segment on [-tau, tau] by walking the gate breakpoints (same algorithm as the validated
m16/approaches/boundary/zones.py:walk_wrong, which was checked against indicator sampling) and returns:
  wp, wn   : length of the misclassified band adjacent to the tie on the t>0 / t<0 side (0 if correct at 0+/0-)
  delta    : signed band width wp - wn (the 'w' of the brief, signed: negative = band on the b-wins side)
  mass     : exact N(0,1) mass of all misclassified t in [-tau, tau] (the integrand of the error estimate)
  madj     : the part of `mass` in the two adjacent bands
  k0p, k0n : model output at 0+ and 0-
  pieces   : linear pieces walked
`jac_at_tie` gives the exact gated Jacobian J (logits x inputs) at t=0+ (logits = J x exactly), the first-order
offset delta_lin = -m/s and the gate pattern.
"""
import sys
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm

sys.path.insert(0, "/home/user/clean-neurons/m16")
T = 10
SQ2 = np.sqrt(2.0)
_v = np.linspace(-7, 7, 200001)
_s = np.sqrt(0.5)
_c = np.cumsum(norm.pdf(_v, scale=_s) * norm.sf(_v) * norm.cdf(_v) ** 7)
_c = (_c - _c[0]) / (_c[-1] - _c[0])


def inv_ybar(u):
    return np.interp(u, _c, _v)


def get_model(name):
    """'orig', 'seed0'..'seed4', 'noisy<sigma>_<k>' (arc_eval.noisy_models(6, sigma, seed=0)[k])."""
    if name == "orig":
        from common import Wi, Wh, Wo
        return (Wi.copy(), Wh.copy(), Wo.copy())
    if name.startswith("seed"):
        from alg_zoo import zoo_2nd_argmax
        m = zoo_2nd_argmax(16, 10, seed=int(name[4:]))
        return (m.rnn.weight_ih_l0.detach().numpy()[:, 0].astype(np.float64),
                m.rnn.weight_hh_l0.detach().numpy().astype(np.float64),
                m.linear.weight.detach().numpy().astype(np.float64))
    if name.startswith("noisy"):
        import os
        cwd = os.getcwd(); os.chdir("/home/user/clean-neurons")
        from arc_eval import noisy_models
        os.chdir(cwd)
        sig, k = name[5:].split("_")
        return noisy_models(6, float(sig), seed=0)[int(k)]
    raise ValueError(name)


def surface(n, rng):
    """Model-free surface points (same law as boundary/direct.py 'A'), canonicalised so a < b.
    Returns y (n,10), a, b, p1, v, o1 (max value), o2 (largest of the seven below)."""
    v = inv_ybar(rng.random(n))
    lo = ndtri(np.clip(rng.random((n, 7)) * ndtr(v)[:, None], 1e-300, 1 - 1e-16))
    p = ndtr(v); hi = ndtri(np.clip(p + rng.random(n) * (1 - p), 1e-300, 1 - 1e-16))
    perm = np.argsort(rng.random((n, T)), 1)
    vals = np.column_stack([v, v, hi, lo])
    y = np.empty((n, T)); rows = np.arange(n)[:, None]
    y[rows, perm] = vals
    a = np.minimum(perm[:, 0], perm[:, 1]); b = np.maximum(perm[:, 0], perm[:, 1]); p1 = perm[:, 2]
    return y, a, b, p1, v, hi, lo.max(1)


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def walk(model, y, a, b, tau, eps=1e-9, max_iter=400, maxseg=600):
    Wi, Wh, Wo = model
    N = len(y); rows_all = np.arange(N)
    d = np.zeros_like(y); d[rows_all, a] = 1 / SQ2; d[rows_all, b] = -1 / SQ2
    S_lo = np.full((N, maxseg), np.nan); S_hi = np.full((N, maxseg), np.nan); S_k = np.full((N, maxseg), -1)
    cnt = np.zeros(N, int)
    s = -tau.copy(); pieces = np.zeros(N, int)
    act = s < tau
    for _ in range(max_iter):
        if not act.any():
            break
        idx = np.nonzero(act)[0]
        X0, D, S = y[idx], d[idx], s[idx] + eps
        h = np.zeros((len(idx), Wh.shape[0])); dh = np.zeros_like(h)
        nxt = tau[idx].copy()
        for t in range(T):
            z = h @ Wh.T + (X0[:, t] + S * D[:, t])[:, None] * Wi[None]
            dz = dh @ Wh.T + D[:, t:t + 1] * Wi[None]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = S[:, None] - z / dz
            r = np.where((r > S[:, None] + 1e-12) & np.isfinite(r), r, np.inf).min(1)
            nxt = np.minimum(nxt, r)
            g = z > 0
            h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        lg = h @ Wo.T; dlg = dh @ Wo.T
        lg = lg - dlg * eps; S0 = s[idx]
        pieces[idx] += 1
        aa = S0.copy(); rows = np.arange(len(idx))
        for _ in range(60):
            live = aa < nxt
            if not live.any():
                break
            val = lg + dlg * (aa - S0)[:, None]
            k = np.argmax(val + dlg * 1e-7, 1)
            vk, dk = val[rows, k], dlg[rows, k]
            with np.errstate(divide="ignore", invalid="ignore"):
                c = aa[:, None] + (vk[:, None] - val) / (dlg - dk[:, None])
            c = np.where((dlg > dk[:, None]) & (c > aa[:, None] + 1e-13), c, np.inf).min(1)
            bb = np.minimum(nxt, c)
            li = idx[live]; ci = cnt[li]
            ok = ci < maxseg
            S_lo[li[ok], ci[ok]] = aa[live][ok]; S_hi[li[ok], ci[ok]] = bb[live][ok]; S_k[li[ok], ci[ok]] = k[live][ok]
            cnt[li] += 1
            aa = np.where(live, bb, aa)
        s[idx] = nxt
        act = s < tau
    # summarise
    tgt = np.where(S_hi <= 0, b[:, None], a[:, None])          # segments are split at 0? not necessarily
    # split segments straddling 0: handle via clipped lengths on each side
    valid = S_k >= 0
    lo_p = np.clip(S_lo, 0, None); hi_p = np.clip(S_hi, 0, None)
    lo_n = np.clip(S_lo, None, 0); hi_n = np.clip(S_hi, None, 0)
    wrong_p = valid & (S_k != a[:, None]) & (hi_p > lo_p)
    wrong_n = valid & (S_k != b[:, None]) & (hi_n > lo_n)
    with np.errstate(invalid="ignore"):
        mass = np.nansum(np.where(wrong_p, ndtr(hi_p) - ndtr(lo_p), 0), 1) + \
            np.nansum(np.where(wrong_n, ndtr(hi_n) - ndtr(lo_n), 0), 1)
    # adjacent bands: walk outward from 0 over contiguous wrong segments
    wp = np.zeros(N); wn = np.zeros(N); k0p = np.full(N, -1); k0n = np.full(N, -1)
    for i in range(N):
        n_ = cnt[i]
        lo_, hi_, k_ = S_lo[i, :n_], S_hi[i, :n_], S_k[i, :n_]
        j = np.searchsorted(hi_, 0.0, side="right")   # first segment with hi > 0
        if j < n_:
            k0p[i] = k_[j]
            e = 0.0
            while j < n_ and k_[j] != a[i]:
                e = hi_[j]; j += 1
            wp[i] = e if (j < n_) else tau[i]
            if k0p[i] == a[i]:
                wp[i] = 0.0
        j = np.searchsorted(lo_, 0.0, side="left") - 1  # last segment with lo < 0
        if j >= 0:
            k0n[i] = k_[j]
            e = 0.0
            while j >= 0 and k_[j] != b[i]:
                e = lo_[j]; j -= 1
            wn[i] = -e if j >= 0 else tau[i]
            if k0n[i] == b[i]:
                wn[i] = 0.0
    madj = (ndtr(wp) - 0.5) + (0.5 - ndtr(-wn))
    return dict(wp=wp, wn=wn, delta=wp - wn, mass=mass, madj=madj, k0p=k0p, k0n=k0n, pieces=pieces, nseg=cnt)


def jac_at_tie(model, y, a, b, side=+1, eps=1e-7):
    """Exact gated Jacobian at y + side*eps*d. Returns J (N,10 logits,10 inputs), gates (N,10 steps,16),
    margin m = (J_a - J_b).y, slope s = d/dt (L_a - L_b), delta_lin = -m/s."""
    Wi, Wh, Wo = model
    N = len(y); rows = np.arange(N)
    d = np.zeros_like(y); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    x = y + side * eps * d
    H = Wh.shape[0]
    h = np.zeros((N, H)); Jh = np.zeros((N, H, T)); G = np.zeros((N, T, H), bool)
    for t in range(T):
        z = h @ Wh.T + x[:, t:t + 1] * Wi[None]
        Jz = np.einsum("ij,njk->nik", Wh, Jh); Jz[:, :, t] += Wi[None]
        g = z > 0; G[:, t] = g
        h = np.where(g, z, 0.0); Jh = np.where(g[:, :, None], Jz, 0.0)
    J = np.einsum("oi,nik->nok", Wo, Jh)
    L = np.einsum("nok,nk->no", J, y)
    m = L[rows, a] - L[rows, b]
    s = ((J[rows, a, a] - J[rows, a, b]) - (J[rows, b, a] - J[rows, b, b])) / SQ2
    with np.errstate(divide="ignore", invalid="ignore"):
        dl = -m / s
    return dict(J=J, G=G, L=L, m=m, s=s, delta_lin=dl)


def build(model, n, seed, chunk=4000):
    rng = np.random.default_rng(seed)
    y, a, b, p1, v, o1, o2 = surface(n, rng)
    tau = SQ2 * np.minimum(o1 - v, v - o2)
    out = {k: [] for k in ["wp", "wn", "delta", "mass", "madj", "k0p", "k0n", "pieces", "nseg",
                           "m", "s", "delta_lin", "G", "Jab"]}
    for i0 in range(0, n, chunk):
        sl = slice(i0, i0 + chunk)
        W = walk(model, y[sl], a[sl], b[sl], tau[sl])
        Jt = jac_at_tie(model, y[sl], a[sl], b[sl])
        for k in ["wp", "wn", "delta", "mass", "madj", "k0p", "k0n", "pieces", "nseg"]:
            out[k].append(W[k])
        for k in ["m", "s", "delta_lin"]:
            out[k].append(Jt[k])
        out["G"].append(np.packbits(Jt["G"].reshape(len(Jt["G"]), -1), 1))
        r = np.arange(len(Jt["J"]))
        out["Jab"].append(Jt["J"][r, a[sl]] - Jt["J"][r, b[sl]])     # (n,10): margin weights per input
    res = {k: np.concatenate(vv) for k, vv in out.items()}
    res.update(y=y, a=a, b=b, p1=p1, v=v, o1=o1, o2=o2, tau=tau)
    return res
