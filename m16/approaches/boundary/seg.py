"""Batched exact integration of the RNN's correctness along short line segments (piecewise-linear walking).

Line: x(t) = x0 + t*d, |d| = 1, x0 . d = 0, so t ~ N(0,1) independent of x0 under x ~ N(0,I).
Allowed set: tlo <= t <= thi (per line). Target: tgt_neg for t < 0, tgt_pos for t > 0 (the rank-2 index
changes only at t = 0 inside a rank-preserving segment).
Returns the Gaussian mass of {t in allowed set: model correct}, the total allowed mass, and the number of
linear pieces walked (each piece = one forward pass carrying value + slope).
"""
import numpy as np
from scipy.special import ndtr

T = 10


def walk(model, x0, d, tlo, thi, tgt_neg, tgt_pos, eps=1e-9, max_iter=400):
    Wi, Wh, Wo = model
    N = len(x0)
    s = tlo.astype(float).copy()
    correct = np.zeros(N)
    pieces = np.zeros(N, dtype=int)
    act = s < thi
    it = 0
    while act.any() and it < max_iter:
        it += 1
        idx = np.nonzero(act)[0]
        X0, D, S = x0[idx], d[idx], s[idx] + eps
        h = np.zeros((len(idx), Wh.shape[0])); dh = np.zeros_like(h)
        nxt = thi[idx].copy()
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
        # argmax envelope on [S0, nxt]
        a = S0.copy(); rows = np.arange(len(idx))
        tn, tp = tgt_neg[idx], tgt_pos[idx]
        lo_, hi_ = tlo[idx], thi[idx]
        for _ in range(50):
            live = a < nxt
            if not live.any():
                break
            val = lg + dlg * (a - S0)[:, None]
            k = np.argmax(val + dlg * 1e-7, 1)
            vk, dk = val[rows, k], dlg[rows, k]
            with np.errstate(divide="ignore", invalid="ignore"):
                c = a[:, None] + (vk[:, None] - val) / (dlg - dk[:, None])
            c = np.where((dlg > dk[:, None]) & (c > a[:, None] + 1e-13), c, np.inf).min(1)
            b = np.minimum(nxt, c)
            # negative part
            l, u = np.maximum(a, lo_), np.minimum(np.minimum(b, 0.0), hi_)
            m = np.where(live & (k == tn) & (u > l), ndtr(u) - ndtr(np.minimum(l, u)), 0.0)
            l, u = np.maximum(np.maximum(a, 0.0), lo_), np.minimum(b, hi_)
            m += np.where(live & (k == tp) & (u > l), ndtr(u) - ndtr(np.minimum(l, u)), 0.0)
            correct[idx] += m
            a = np.where(live, b, a)
        s[idx] = nxt
        act = s < thi
    total = ndtr(thi) - ndtr(tlo)
    return correct, total, pieces
