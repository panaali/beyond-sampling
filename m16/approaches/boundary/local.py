"""Boundary-layer (local) integration around the tie point t = 0 of a rank-preserving segment.

One forward pass (value + slope) at the tie point gives the exact affine logits on the whole linear piece
containing it. The misclassified band adjacent to the tie is found exactly by walking outward from t = 0
until the first correct point (usually inside that first piece). Errors in the rest of the segment
(non-adjacent bands) are estimated without bias by one random point (indicator x remaining mass).
"""
import numpy as np
from scipy.special import ndtr

T = 10


def _pass(model, x0, d, S):
    """Forward with slope at t = S. Returns logits, slopes, nearest gate breakpoint above / below S."""
    Wi, Wh, Wo = model
    h = np.zeros((len(x0), Wh.shape[0])); dh = np.zeros_like(h)
    up = np.full(len(x0), np.inf); dn = np.full(len(x0), -np.inf)
    for t in range(T):
        z = h @ Wh.T + (x0[:, t] + S * d[:, t])[:, None] * Wi[None]
        dz = dh @ Wh.T + d[:, t:t + 1] * Wi[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = S[:, None] - z / dz
        ok = np.isfinite(r)
        up = np.minimum(up, np.where(ok & (r > S[:, None] + 1e-12), r, np.inf).min(1))
        dn = np.maximum(dn, np.where(ok & (r < S[:, None] - 1e-12), r, -np.inf).max(1))
        g = z > 0
        h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
    return h @ Wo.T, dh @ Wo.T, up, dn


def _envelope(lg, dlg, a0, end, tgt):
    """Logits lg + dlg*(u - a0) on [a0, end], u >= 0 measured from the tie. Wrong mass from a0 until the
    first u where argmax == tgt. Returns (wrong_mass, stop_u, found_correct)."""
    N = len(lg); rows = np.arange(N)
    a = a0.copy(); wrong = np.zeros(N); found = np.zeros(N, bool); live = a < end
    for _ in range(60):
        live &= ~found & (a < end)
        if not live.any():
            break
        val = lg + dlg * (a - a0)[:, None]
        k = np.argmax(val + dlg * 1e-7, 1)
        vk, dk = val[rows, k], dlg[rows, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = a[:, None] + (vk[:, None] - val) / (dlg - dk[:, None])
        c = np.where((dlg > dk[:, None]) & (c > a[:, None] + 1e-13), c, np.inf).min(1)
        b = np.minimum(end, c)
        corr = live & (k == tgt)
        found |= corr
        w = live & ~corr
        wrong += np.where(w, ndtr(b) - ndtr(a), 0.0)
        a = np.where(w, b, a)
    return wrong, a, found


def _side(model, x0, d, lg, dlg, brk, end, tgt, sign, max_pieces=60):
    """Walk one side (sign=+1: t>0 along d; -1: t<0, reflected) from the tie until first correct point."""
    N = len(x0)
    a0 = np.zeros(N)
    e = np.minimum(brk, end)
    wrong, stop, found = _envelope(lg, dlg, a0, e, tgt)
    passes = np.zeros(N, int)
    cont = ~found & (stop < end)
    for _ in range(max_pieces):
        if not cont.any():
            break
        i = np.nonzero(cont)[0]
        S = stop[i] + 1e-9
        lg2, dlg2, up, _ = _pass(model, x0[i], sign * d[i], S)
        passes[i] += 1
        lg2 = lg2 - dlg2 * 1e-9
        w2, st2, f2 = _envelope(lg2, dlg2, stop[i], np.minimum(up, end[i]), tgt[i])
        wrong[i] += w2; stop[i] = st2; found[i] = f2
        cont[i] = ~f2 & (st2 < end[i])
    return wrong, stop, passes


def local_band(model, x0, d, T_pos, T_neg, tgt_pos, tgt_neg):
    """Exact misclassified mass of the band adjacent to t=0, over the segment [-T_neg, T_pos].
    Returns band_wrong, right_stop, left_stop (as positive distance), passes."""
    N = len(x0)
    lg, dlg, up, dn = _pass(model, x0, d, np.zeros(N))
    wr, sr, pr = _side(model, x0, d, lg, dlg, up, T_pos, tgt_pos, +1)
    wl, sl, pl = _side(model, x0, d, lg, -dlg, -dn, T_neg, tgt_neg, -1)
    return wr + wl, sr, sl, 1 + pr + pl


def point_wrong(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1) != np.argsort(x, 1)[:, -2]


def remainder_sample(model, x0, d, lo, hi, sl, sr, rng):
    """One-point unbiased estimate of wrong mass on [lo, -sl] U [sr, hi] (lo<0<hi)."""
    mL = np.maximum(ndtr(-sl) - ndtr(lo), 0); mR = np.maximum(ndtr(hi) - ndtr(sr), 0)
    m = mL + mR
    u = rng.random(len(x0)) * m
    left = u < mL
    q = np.where(left, ndtr(lo) + u, ndtr(sr) + (u - mL))
    from scipy.special import ndtri
    t = ndtri(np.clip(q, 1e-300, 1 - 1e-16))
    t = np.where(left, np.clip(t, lo, -sl), np.clip(t, sr, hi))
    has = m > 1e-15
    wrong = np.zeros(len(x0))
    if has.any():
        i = np.nonzero(has)[0]
        wrong[i] = point_wrong(model, x0[i] + t[i, None] * d[i])
    return m * wrong, has.astype(int)
