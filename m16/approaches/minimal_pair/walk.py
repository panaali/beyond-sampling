"""Exact segment walk with the prefix saving IMPLEMENTED (not only counted).

Along x(t) = y + t d the network is piecewise linear in t. Within a piece, every pre-activation is
z_s(t) = z_s(tc) + (t - tc) dz_s(tc). When the walk moves to the next piece, the pre-activations of all steps before
the first step whose gate pattern changes are obtained by this linear update (O(H) per step and row), and only the
steps from there on are recomputed with the recurrent matrix (O(H^2)). The cost charged is what is computed:
    (recomputed value steps + recomputed tangent steps) / T   FPE  (no tangent before step min(a,b)),   plus the O(H) updates and root tests at their FLOP share.
The first cell of a segment is recomputed in full.

mass along the segment uses an arbitrary 1-D CDF (Gaussian for the plain co-area integrand, Student-t_{T-1} for the
radially integrated one, see hom.py). Correct-set logic (numpy first-index argmax rule) as in ht_seeds/core.py.
"""
import numpy as np
from scipy.special import ndtr


def walk(model, y, a, b, tau, kind, cdf=ndtr, eps=1e-10, max_cells=4000):
    Wi, Wh, Wo = model
    n, T = y.shape; H = Wh.shape[0]
    rows = np.arange(n)
    d = np.zeros_like(y); d[rows, a] = 1 / np.sqrt(2); d[rows, b] = -1 / np.sqrt(2)
    tpos = a if kind == "23" else b
    tneg = b if kind == "23" else a
    step_f = 2 * H * H + 2 * H                 # FLOPs of one recurrent step (one sequence), the FPE unit / T
    upd_f = 6 * H                              # FLOPs of a linear update + gate test of one step (value only)
    t = -tau.copy()
    mass = np.zeros(n); cost = np.zeros(n); cells = np.zeros(n, int)
    Z = np.zeros((n, T, H)); DZ = np.zeros((n, T, H)); TC = np.full(n, np.nan)
    act = np.nonzero(t < tau)[0]
    it = 0
    while len(act) and it < max_cells:
        it += 1
        m = len(act)
        X0, D = y[act], d[act]
        amin = np.minimum(a[act], b[act])
        tc = t[act] + eps * (1 + np.abs(t[act]))
        first = np.isnan(TC[act])
        # linear update of the stored prefix to tc; find the first step whose gates change
        Zu = Z[act] + (tc - np.where(first, tc, TC[act]))[:, None, None] * DZ[act]
        chg = (Zu > 0) != (Z[act] > 0)
        r0 = np.where(chg.any(2).any(1), chg.any(2).argmax(1), T)
        r0 = np.where(first, 0, r0)
        h = np.zeros((m, H)); dh = np.zeros((m, H))
        nxt = tau[act].copy(); rstep = np.full(m, T)
        Znew = np.empty((m, T, H)); DZnew = np.empty((m, T, H))
        for s in range(T):
            rec = r0 <= s
            z = Zu[:, s].copy(); dz = DZ[act, s].copy()
            if rec.any():
                z[rec] = h[rec] @ Wh.T + (X0[rec, s] + tc[rec] * D[rec, s])[:, None] * Wi[None]
                tg = rec & (s >= amin)            # before step min(a,b) the tangent is identically 0
                dz[tg] = dh[tg] @ Wh.T + D[tg, s:s + 1] * Wi[None]
                dz[rec & ~tg] = 0.0
            Znew[:, s] = z; DZnew[:, s] = dz
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tc[:, None] - z / dz
            r = np.where((r > tc[:, None]) & np.isfinite(r), r, np.inf).min(1)
            upd = r < nxt
            nxt = np.where(upd, r, nxt); rstep = np.where(upd, s, rstep)
            g = z > 0
            h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        Z[act] = Znew; DZ[act] = DZnew; TC[act] = tc
        L = h @ Wo.T; dL = dh @ Wo.T
        nrec = T - r0
        ntan = T - np.maximum(r0, amin)      # recomputed steps that also need the tangent
        cost[act] += ((nrec + ntan) * step_f + (T - nrec) * upd_f + T * 4 * H + 4 * T * H) / (T * step_f)
        cells[act] += 1
        lo, hi = t[act], nxt
        rr = np.arange(m)
        for side in (0, 1):
            if side == 0:
                p_lo, p_hi, k = lo, np.minimum(hi, 0.0), tneg[act]
            else:
                p_lo, p_hi, k = np.maximum(lo, 0.0), hi, tpos[act]
            ok = p_hi > p_lo
            if not ok.any():
                continue
            c = (L[rr, k][:, None] - L) - (dL[rr, k][:, None] - dL) * tc[:, None]
            e = dL[rr, k][:, None] - dL
            with np.errstate(divide="ignore", invalid="ignore"):
                root = -c / e
            lb = np.where(e > 0, root, -np.inf); ub = np.where(e < 0, root, np.inf)
            jj = np.arange(L.shape[1])[None, :]
            bad0 = (e == 0) & ((c < 0) | ((np.abs(c) <= 1e-12 * (1 + np.abs(L).max(1))[:, None]) & (jj < k[:, None])))
            lb[rr, k] = -np.inf; ub[rr, k] = np.inf
            clo = np.maximum(lb.max(1), p_lo); chi = np.minimum(ub.min(1), p_hi)
            corr = np.where((chi > clo) & ~bad0.any(1), cdf(chi) - cdf(clo), 0.0)
            tot = cdf(p_hi) - cdf(p_lo)
            mass[act] += np.where(ok, tot - corr, 0.0)
        t[act] = nxt
        act = act[nxt < tau[act]]
    return mass, cost, cells
