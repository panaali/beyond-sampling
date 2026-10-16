"""Boundary-layer integration over the tie surfaces, with model-free zones (unbiased decomposition).

Nonblind error = A + B + R, blind error = P(wrong, x_(2) <= 0):
  zone A = {x_(2) - x_(3) < gA}                         (2nd-vs-3rd boundary layer)
  zone B = {x_(1) - x_(2) < gB} minus zone A            (1st-vs-2nd boundary layer)
  zone R = everything else with x_(2) > 0; blind = x_(2) <= 0.
Zone membership depends on the input only, so the split is exact for any weights.

A and B are written as integrals over the tie surfaces x_a = x_b (8+1 dims) of the 1-D integral, along the
normal direction (e_a - e_b)/sqrt2, of the misclassified Gaussian mass inside the zone. The 1-D integral is done
EXACTLY by walking the network's linear pieces (no sampling). The surface integral is sampled (y ~ the
Gaussian on the surface restricted to the tie being at ranks 2-3 / 1-2), with the surface measure computed
by 1-D quadrature (model-free). R and blind are plain indicator sampling inside their zones.
"""
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from scipy.stats import norm

T = 10
SQ2 = np.sqrt(2.0)
NPAIR = 45
PAIRS = [(a, b) for a in range(T) for b in range(a + 1, T)]
_s = np.sqrt(0.5)
# surface measures (model-free, exact quadrature): ybar ~ N(0, 1/2), 8 other coords iid N(0,1)
P_ACC_A = quad(lambda v: norm.pdf(v, scale=_s) * 8 * norm.sf(v) * norm.cdf(v) ** 7, -12, 12, epsabs=1e-14)[0]
P_ACC_B = quad(lambda v: norm.pdf(v, scale=_s) * norm.cdf(v) ** 8, -12, 12, epsabs=1e-14)[0]


def walk_wrong(model, x0, d, tau, tgt_neg, tgt_pos, mlo, mhi, eps=1e-9, max_iter=200):
    """Exact Gaussian mass of {t in [-tau, tau], mlo <= |t| <= mhi, model wrong} on x0 + t d.
    Returns (mass, pieces)."""
    Wi, Wh, Wo = model
    N = len(x0)
    s = -tau.copy(); wrong = np.zeros(N); pieces = np.zeros(N, int)
    act = s < tau
    for _ in range(max_iter):
        if not act.any():
            break
        idx = np.nonzero(act)[0]
        X0, D, S = x0[idx], d[idx], s[idx] + eps
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
        a = S0.copy(); rows = np.arange(len(idx))
        tn, tp, lo_m, hi_m = tgt_neg[idx], tgt_pos[idx], mlo[idx], mhi[idx]
        for _ in range(60):
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
            m = np.zeros(len(idx))
            # t < 0 part: |t| in [mlo, mhi]  <=>  t in [-mhi, -mlo]
            l, u = np.maximum(a, -hi_m), np.minimum(np.minimum(b, 0.0), -lo_m)
            m += np.where(live & (k != tn) & (u > l), ndtr(u) - ndtr(np.minimum(l, u)), 0.0)
            l, u = np.maximum(np.maximum(a, 0.0), lo_m), np.minimum(b, hi_m)
            m += np.where(live & (k != tp) & (u > l), ndtr(u) - ndtr(np.minimum(l, u)), 0.0)
            wrong[idx] += m
            a = np.where(live, b, a)
        s[idx] = nxt
        act = s < tau
    return wrong, pieces


def surface_points(kind, n, rng):
    """n accepted surface points: pair (a,b), y on the hyperplane x_a = x_b, with the tie at ranks 2-3 (A)
    or 1-2 (B). Returns y, a, b, ybar, others' sorted values (desc). Model-free (rejection on inputs only)."""
    out = []; got = 0
    while got < n:
        m = 4 * n + 1000
        z = rng.standard_normal((m, T))
        pi = rng.integers(0, NPAIR, m)
        pa = np.array([p[0] for p in PAIRS])[pi]; pb = np.array([p[1] for p in PAIRS])[pi]
        rows = np.arange(m)
        yb = (z[rows, pa] + z[rows, pb]) / 2
        z[rows, pa] = yb; z[rows, pb] = yb
        oth = z.copy(); oth[rows, pa] = -np.inf; oth[rows, pb] = -np.inf
        os_ = -np.sort(-oth, 1)                         # others desc (last two are -inf)
        if kind == "A":
            ok = (yb < os_[:, 0]) & (yb > os_[:, 1])
        else:
            ok = yb > os_[:, 0]
        i = np.nonzero(ok)[0]
        out.append((z[i], pa[i], pb[i], yb[i], os_[i]))
        got += len(i)
    cat = [np.concatenate([o[j] for o in out])[:n] for j in range(5)]
    return cat


def term_A(model, n, rng, gA):
    """Unbiased estimate of P(wrong, nonblind, zone A) from n surface points. Returns (per-point values, pieces)."""
    y, a, b, yb, os_ = surface_points("A", n, rng)
    rows = np.arange(n)
    d = np.zeros_like(y); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    T_rank = SQ2 * np.minimum(os_[:, 0] - yb, yb - os_[:, 1])
    tau = np.minimum(T_rank, gA / SQ2)
    mlo = np.maximum(0.0, -SQ2 * yb)                  # nonblind: max(xa, xb) = ybar + |t|/sqrt2 > 0
    mhi = np.full(n, np.inf)
    use = tau > mlo
    vals = np.zeros(n); pcs = np.zeros(n, int)
    i = np.nonzero(use)[0]
    if len(i):
        # t > 0: x_a > x_b, so a is rank 2 (target); t < 0: b is the target
        w, p = walk_wrong(model, y[i], d[i], tau[i], b[i], a[i], mlo[i], mhi[i])
        vals[i] = w; pcs[i] = p
    return NPAIR * P_ACC_A * vals, pcs


def term_B(model, n, rng, gA, gB):
    y, a, b, yb, os_ = surface_points("B", n, rng)
    rows = np.arange(n)
    d = np.zeros_like(y); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    o3 = os_[:, 0]
    # ranks 1-2 kept and outside zone A: min(xa,xb) - o3 >= gA ; zone B: |xa-xb| < gB ; nonblind: min > 0
    tau = np.minimum(np.minimum(SQ2 * (yb - o3 - gA), gB / SQ2), SQ2 * yb)
    tau = np.maximum(tau, 0.0)
    use = tau > 0
    vals = np.zeros(n); pcs = np.zeros(n, int)
    i = np.nonzero(use)[0]
    if len(i):
        # t > 0: x_a is max, target b; t < 0: target a
        w, p = walk_wrong(model, y[i], d[i], tau[i], a[i], b[i], np.zeros(len(i)), np.full(len(i), np.inf))
        vals[i] = w; pcs[i] = p
    return NPAIR * P_ACC_B * vals, pcs


def zone_of(x, gA, gB):
    s = -np.sort(-x, 1)
    blind = s[:, 1] <= 0
    zA = (s[:, 1] - s[:, 2]) < gA
    zB = ~zA & ((s[:, 0] - s[:, 1]) < gB)
    return np.where(blind, 0, np.where(zA, 1, np.where(zB, 2, 3)))   # 0 blind, 1 A, 2 B, 3 R


def point_wrong(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1) != np.argsort(x, 1)[:, -2]


def term_zone(model, n_x, rng, gA, gB, zone):
    """Plain sampling restricted to a zone: draw n_x inputs, evaluate the model only on those in the zone.
    Returns per-input values (0 outside the zone) and number of model passes."""
    x = rng.standard_normal((n_x, T))
    z = zone_of(x, gA, gB)
    m = z == zone
    v = np.zeros(n_x)
    if m.any():
        v[m] = point_wrong(model, x[m])
    return v, int(m.sum())
