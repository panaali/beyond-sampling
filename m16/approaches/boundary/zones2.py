"""Cheaper boundary-layer integral: exact walk only over the misclassified band adjacent to the tie, plus an
unbiased one-point estimate of any other misclassified mass inside the zone (same line, Gaussian-weighted point).
Same zones and unbiased decomposition as zones.py."""
import numpy as np
from scipy.special import ndtr, ndtri
import zones as Z

T = 10
SQ2 = np.sqrt(2.0)


def _pass(model, x0, d, S):
    Wi, Wh, Wo = model
    h = np.zeros((len(x0), Wh.shape[0])); dh = np.zeros_like(h)
    up = np.full(len(x0), np.inf); dn = np.full(len(x0), -np.inf)
    kup = np.zeros(len(x0), int); kdn = np.zeros(len(x0), int)
    for t in range(T):
        z = h @ Wh.T + (x0[:, t] + S * d[:, t])[:, None] * Wi[None]
        dz = dh @ Wh.T + d[:, t:t + 1] * Wi[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = S[:, None] - z / dz
        ok = np.isfinite(r)
        u_ = np.where(ok & (r > S[:, None] + 1e-12), r, np.inf).min(1)
        d_ = np.where(ok & (r < S[:, None] - 1e-12), r, -np.inf).max(1)
        kup = np.where(u_ < up, t, kup); kdn = np.where(d_ > dn, t, kdn)
        up = np.minimum(up, u_); dn = np.maximum(dn, d_)
        g = z > 0
        h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
    return h @ Wo.T, dh @ Wo.T, up, dn, kup, kdn


def _env(lg, dlg, a0, end, tgt, mlo, mhi):
    """Affine logits lg + dlg (u - a0) on [a0, end]. Walk while argmax != tgt; masked wrong mass."""
    N = len(lg); rows = np.arange(N)
    a = a0.copy(); wrong = np.zeros(N); found = np.zeros(N, bool)
    for _ in range(60):
        live = ~found & (a < end)
        if not live.any():
            break
        val = lg + dlg * (a - a0)[:, None]
        k = np.argmax(val + dlg * 1e-7, 1)
        vk, dk = val[rows, k], dlg[rows, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = a[:, None] + (vk[:, None] - val) / (dlg - dk[:, None])
        c = np.where((dlg > dk[:, None]) & (c > a[:, None] + 1e-13), c, np.inf).min(1)
        b = np.minimum(end, c)
        wr = k != tgt
        found |= live & ~wr
        w = live & wr
        l, u = np.maximum(a, mlo), np.minimum(b, mhi)
        wrong += np.where(w & (u > l), ndtr(u) - ndtr(np.minimum(l, u)), 0.0)
        a = np.where(w, b, a)
    return wrong, a


def _env_all(lg, dlg, a0, s0, end, tgt, mlo, mhi):
    """Affine logits lg + dlg (u - a0) (piece starting at a0); exact masked wrong mass over [s0, end]."""
    N = len(lg); rows = np.arange(N)
    a = s0.copy(); wrong = np.zeros(N)
    for _ in range(60):
        live = a < end
        if not live.any():
            break
        val = lg + dlg * (a - a0)[:, None]
        k = np.argmax(val + dlg * 1e-7, 1)
        vk, dk = val[rows, k], dlg[rows, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = a[:, None] + (vk[:, None] - val) / (dlg - dk[:, None])
        c = np.where((dlg > dk[:, None]) & (c > a[:, None] + 1e-13), c, np.inf).min(1)
        b = np.minimum(end, c)
        l, u = np.maximum(a, mlo), np.minimum(b, mhi)
        wrong += np.where(live & (k != tgt) & (u > l), ndtr(u) - ndtr(np.minimum(l, u)), 0.0)
        a = np.where(live, b, a)
    return wrong


def _side(model, y, d, lg, dlg, brk, kbrk, tau, tgt, mlo, mhi, sign):
    """Returns wrong mass of the adjacent component, stop point, last piece (lg, dlg, start, end), passes."""
    N = len(y)
    end = np.minimum(brk, tau)
    start = np.zeros(N)
    wrong, stop = _env(lg, dlg, start, end, tgt, mlo, mhi)
    LG, DLG, ST, EN = lg.copy(), dlg.copy(), start.copy(), end.copy()
    passes = np.zeros(N, int); flops = np.zeros(N); K = kbrk.copy()
    for _ in range(60):
        cont = (stop >= EN - 1e-15) & (EN < tau)       # band reached the end of the walked piece
        if not cont.any():
            break
        i = np.nonzero(cont)[0]
        S = EN[i] + 1e-9
        l2, d2, up, _, ku, _ = _pass(model, y[i], sign * d[i], S)
        passes[i] += 1
        flops[i] += 2 * (T - K[i]) / T            # value + slope recomputed from the flipping step on
        K[i] = ku
        l2 = l2 - d2 * 1e-9
        e2 = np.minimum(up, tau[i])
        w2, st2 = _env(l2, d2, EN[i], e2, tgt[i], mlo[i], mhi[i])
        wrong[i] += w2; stop[i] = st2
        LG[i], DLG[i], ST[i], EN[i] = l2, d2, EN[i], e2
    return wrong, stop, (LG, DLG, ST, EN), passes, flops


def zone_line_mass(model, y, d, tau, tgt_neg, tgt_pos, mlo, mhi, rng):
    """Unbiased estimate of misclassified masked mass on |t| <= tau. Returns (estimate, passes, exact_part)."""
    N = len(y)
    lg, dlg, up, dn, kup, kdn = _pass(model, y, d, np.zeros(N))
    wr, sr, cr, pr, fr = _side(model, y, d, lg, dlg, up, kup, tau, tgt_pos, mlo, mhi, +1)
    wl, sl, cl, pl, fl = _side(model, y, d, lg, -dlg, -dn, kdn, tau, tgt_neg, mlo, mhi, -1)
    passes = 1 + pr + pl
    # rest of the last walked piece on each side is known exactly (affine logits): integrate it, no passes
    wr = wr + _env_all(cr[0], cr[1], cr[2], sr, cr[3], tgt_pos, mlo, mhi)
    wl = wl + _env_all(cl[0], cl[1], cl[2], sl, cl[3], tgt_neg, mlo, mhi)
    sr = np.maximum(sr, cr[3]); sl = np.maximum(sl, cl[3])
    k0 = np.argmax(d != 0, 1)                    # inputs before k0 are fixed along the line
    flops = (T + (T - k0)) / T + fr + fl         # first pass: value (full) + slope (from k0)
    # remaining regions (in |t|): right [max(sr,mlo), min(tau,mhi)], left [max(sl,mlo), min(tau,mhi)]
    hiM = np.minimum(tau, mhi)
    loR, loL = np.maximum(sr, mlo), np.maximum(sl, mlo)
    mR = np.where(hiM > loR, ndtr(hiM) - ndtr(np.minimum(loR, hiM)), 0.0)
    mL = np.where(hiM > loL, ndtr(hiM) - ndtr(np.minimum(loL, hiM)), 0.0)
    m = mR + mL
    est = wr + wl
    has = m > 1e-14
    if has.any():
        i = np.nonzero(has)[0]
        u = rng.random(len(i)) * m[i]
        right = u < mR[i]
        q = np.where(right, ndtr(loR[i]) + u, ndtr(loL[i]) + (u - mR[i]))
        uu = ndtri(np.clip(q, 1e-300, 1 - 1e-16))
        uu = np.where(right, np.clip(uu, loR[i], hiM[i]), np.clip(uu, loL[i], hiM[i]))
        # known from the last walked piece on that side?
        LG = np.where(right[:, None], cr[0][i], cl[0][i]); DLG = np.where(right[:, None], cr[1][i], cl[1][i])
        ST = np.where(right, cr[2][i], cl[2][i]); EN = np.where(right, cr[3][i], cl[3][i])
        known = (uu >= ST) & (uu < EN)
        out = np.argmax(LG + DLG * (uu - ST)[:, None], 1)
        tg = np.where(right, tgt_pos[i], tgt_neg[i])
        need = ~known
        if need.any():
            j = np.nonzero(need)[0]
            t = np.where(right[j], uu[j], -uu[j])
            out[j] = np.argmax(_logits(model, y[i][j] + t[:, None] * d[i][j]), 1)
            passes[i[j]] += 1
            flops[i[j]] += (T - k0[i[j]]) / T     # value only, prefix before k0 cached
        est[i] += m[i] * (out != tg)
    return est, passes, flops


def _logits(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def geom_A(model, n, rng, gA):
    y, a, b, yb, os_ = Z.surface_points("A", n, rng)
    rows = np.arange(n)
    d = np.zeros_like(y); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    tau = np.minimum(SQ2 * np.minimum(os_[:, 0] - yb, yb - os_[:, 1]), gA / SQ2)
    mlo = np.maximum(0.0, -SQ2 * yb); mhi = np.full(n, np.inf)
    return dict(y=y, d=d, tau=tau, tn=b, tp=a, mlo=mlo, mhi=mhi, ybar=yb, w=Z.NPAIR * Z.P_ACC_A,
                T_rank=SQ2 * np.minimum(os_[:, 0] - yb, yb - os_[:, 1]))


def geom_B(model, n, rng, gA, gB):
    y, a, b, yb, os_ = Z.surface_points("B", n, rng)
    rows = np.arange(n)
    d = np.zeros_like(y); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    o3 = os_[:, 0]
    tau = np.maximum(np.minimum(np.minimum(SQ2 * (yb - o3 - gA), gB / SQ2), SQ2 * yb), 0.0)
    return dict(y=y, d=d, tau=tau, tn=a, tp=b, mlo=np.zeros(n), mhi=np.full(n, np.inf), ybar=yb,
                w=Z.NPAIR * Z.P_ACC_B, T_rank=SQ2 * (yb - o3))


def term(model, G, rng):
    n = len(G["y"]); vals = np.zeros(n); pcs = np.zeros(n, int); fl = np.zeros(n)
    use = np.minimum(G["tau"], G["mhi"]) > G["mlo"]
    i = np.nonzero(use)[0]
    if len(i):
        e, p, f = zone_line_mass(model, G["y"][i], G["d"][i], G["tau"][i], G["tn"][i], G["tp"][i],
                                 G["mlo"][i], G["mhi"][i], rng)
        vals[i] = e; pcs[i] = p; fl[i] = f
    return G["w"] * vals, pcs, fl
