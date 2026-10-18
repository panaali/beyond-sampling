"""T223: sci_tao's exact co-area tie-slicing (m16/approaches/sci_tao/core.py, imported read-only) restricted to the
NON-blind set N = {x_(2) > 0} (the 2nd-largest input is positive). Blind set B = {x_(2) <= 0} = at most one positive
input, P(B) = (T+1)/2^T exactly (11/1024 at T = 10).

On a face segment x = y + t d (d = (e_a - e_b)/sqrt2, x_a = u + t/sqrt2, x_b = u - t/sqrt2, inside the window the
pair keeps its ranks), the 2nd-largest value is
  R23 (a, b = ranks 2, 3):  x_(2) = u + |t|/sqrt2  -> N on the segment = {|t| > -sqrt2 u} (all of it if u >= 0)
  R12 (a, b = ranks 1, 2):  x_(2) = u - |t|/sqrt2  -> N on the segment = {|t| < sqrt2 u} (empty if u <= 0)
so P(wrong, N) = Z23 E23[F_N] + Z12 E12[F_N] exactly, with F_N the misclassified phi-mass on N's part of the segment.
`walk_mass_nb` is `core.walk_mass` with that restriction (pieces clipped to N; R12 windows shrunk to |t| < sqrt2 u;
segments entirely in B are skipped at zero cost). Everything else, cost counting included, is unchanged."""
import os, sys
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sci_tao"))
import core as C  # noqa: E402

SQ2 = C.SQ2


def walk_mass_nb(model, y, a, b, tau, kind, eps=1e-10, max_cells=2000):
    Wi, Wh, Wo = model
    n, T = y.shape; H = Wh.shape[0]; rows = np.arange(n)
    u = y[rows, a]; cN = SQ2 * np.abs(u)
    tau = tau.copy()
    if kind == "12":
        tau = np.where(u > 0, np.minimum(tau, cN), 0.0)          # N = {|t| < sqrt2 u}
    else:
        tau = np.where((u < 0) & (cN >= tau), 0.0, tau)          # whole window blind
    d = np.zeros_like(y); d[rows, a] = 1 / SQ2; d[rows, b] = -1 / SQ2
    tpos = a if kind == "23" else b
    tneg = b if kind == "23" else a
    first = np.minimum(a, b)
    t = -tau.copy()
    mass = np.zeros(n); cost = np.where(tau > 0, first / T, 0.0); cells = np.zeros(n, int)
    restart = first.copy()
    act = np.nonzero(t < tau)[0]
    it = 0
    while len(act) and it < max_cells:
        it += 1
        X0, D = y[act], d[act]
        tc = t[act] + eps * (1 + np.abs(t[act]))
        h = np.zeros((len(act), H)); dh = np.zeros_like(h)
        nxt = tau[act].copy(); rstep = np.full(len(act), T)
        for s in range(T):
            z = h @ Wh.T + (X0[:, s] + tc * D[:, s])[:, None] * Wi[None]
            dz = dh @ Wh.T + D[:, s:s + 1] * Wi[None]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tc[:, None] - z / dz
            r = np.where((r > tc[:, None]) & np.isfinite(r), r, np.inf).min(1)
            upd = r < nxt
            nxt = np.where(upd, r, nxt); rstep = np.where(upd, s, rstep)
            g = z > 0
            h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        L = h @ Wo.T; dL = dh @ Wo.T
        lo = t[act]; hi = nxt
        cost[act] += 2 * (T - restart[act]) / T
        cells[act] += 1
        ua, ca = u[act], cN[act]
        for side in (0, 1):
            if side == 0:
                p_lo, p_hi, k = lo, np.minimum(hi, 0.0), tneg[act]
                if kind == "23":
                    p_hi = np.where(ua < 0, np.minimum(p_hi, -ca), p_hi)     # keep t <= -sqrt2|u|
            else:
                p_lo, p_hi, k = np.maximum(lo, 0.0), hi, tpos[act]
                if kind == "23":
                    p_lo = np.where(ua < 0, np.maximum(p_lo, ca), p_lo)      # keep t >= sqrt2|u|
            ok = p_hi > p_lo
            if not ok.any():
                continue
            rr = np.arange(len(act))
            c = (L[rr, k][:, None] - L) - (dL[rr, k][:, None] - dL) * tc[:, None]
            e = dL[rr, k][:, None] - dL
            with np.errstate(divide="ignore", invalid="ignore"):
                root = -c / e
            lb = np.where(e > 0, root, -np.inf); ub = np.where(e < 0, root, np.inf)
            jj = np.arange(L.shape[1])[None, :]
            bad0 = (e == 0) & ((c < 0) | ((np.abs(c) <= 1e-12 * (1 + np.abs(L).max(1))[:, None]) & (jj < k[:, None])))
            lb[rr, k] = -np.inf; ub[rr, k] = np.inf
            clo = np.maximum(lb.max(1), p_lo); chi = np.minimum(ub.min(1), p_hi)
            corr = np.where((chi > clo) & ~bad0.any(1), ndtr(chi) - ndtr(clo), 0.0)
            tot = ndtr(p_hi) - ndtr(p_lo)
            mass[act] += np.where(ok, tot - corr, 0.0)
        t[act] = nxt
        restart[act] = rstep
        act = act[nxt < tau[act]]
    return mass, cost, cells


def blind_points(T, U):
    """Map uniform points U (n, T+1) to B: U[:,0] picks one of the T+1 equally likely orthants (all negative, or
    exactly x_p > 0), U[:,1:] give the half-normal magnitudes. Each orthant has probability 2^-T."""
    n = len(U); o = np.minimum((U[:, 0] * (T + 1)).astype(int), T)
    x = -ndtri(np.clip(0.5 + 0.5 * U[:, 1:], 0.5, 1 - 1e-16))           # |N(0,1)|, negated
    rows = np.flatnonzero(o < T); x[rows, o[rows]] *= -1
    return x


def blind_wrong(model, x):
    return (C.forward(model, x).argmax(1) != np.argsort(x, 1)[:, -2]).astype(float)


def sobol(nd, n, seed):
    m = int(np.ceil(np.log2(max(n, 2))))
    return qmc.Sobol(nd, scramble=True, seed=seed).random_base2(m)[:n]
