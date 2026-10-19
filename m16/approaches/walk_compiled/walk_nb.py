"""Compiled (numba) version of T10's frozen walk.walk (frozen/walk.py, a5063c7d).

Same algorithm, one row at a time (rows are independent in the frozen code: the global iteration cap max_cells is a
per-row cell cap). Per cell: linear update of the stored prefix, first gate-changing step r0, recompute steps >= r0
(value, and tangent from step min(a,b)), next breakpoint, logits, the two target pieces and their correct sets.
Charged cost is computed with the frozen formula (same integer arithmetic, so it matches bitwise when r0 matches).

Two CDF modes:
  repro: the walk emits the 1-D CDF arguments (p_lo, p_hi, clo, chi) of every piece; the CDF given by the caller
         (frozen: scipy stdtr via hom.tcdf) is evaluated on them in one numpy call, and the masses are accumulated
         in the frozen order (side 0 then side 1, cell by cell).
  fast:  closed-form Student-t_{T-1} CDF (integer nu) inside the compiled walk; no events.
"""
import math

import numpy as np
from numba import njit

SQ2 = math.sqrt(2.0)


@njit(cache=True, error_model="numpy", fastmath=False)
def tcdf_int(x, nu):
    """Student-t CDF with integer nu >= 1 of x (in t units), closed form (Abramowitz-Stegun 26.7.3/4)."""
    if x == np.inf:
        return 1.0
    if x == -np.inf:
        return 0.0
    if x > 1e150:
        return 1.0
    if x < -1e150:
        return 0.0
    x2 = x * x
    c2 = nu / (nu + x2)                     # cos^2 theta
    s = x / math.sqrt(nu + x2)              # sin theta
    if nu % 2 == 0:
        term = 1.0; acc = 1.0
        for k in range(1, nu // 2):
            term *= c2 * (2 * k - 1) / (2 * k)
            acc += term
        return 0.5 + 0.5 * s * acc
    th = math.atan(x / math.sqrt(nu))
    if nu == 1:
        return 0.5 + th / math.pi
    ct = math.sqrt(c2)
    term = 1.0; acc = 1.0
    for k in range(1, (nu - 1) // 2):
        term *= c2 * (2 * k) / (2 * k + 1)
        acc += term
    return 0.5 + (th + s * ct * acc) / math.pi


@njit(cache=True, error_model="numpy", fastmath=False)
def _walk(Wi, Wh, Wo, y, a, b, tau, is23, eps, max_cells, fast, nu, sq, ev, flops):
    """ev: event buffer (cap, 5) [row, p_lo, p_hi, clo, chi] with clo=chi=nan when the correct set is empty/invalid,
    row = -1 - row when the piece is empty (ok False). Returns mass, cost, cells, n_events (or -1 on overflow).
    flops[0]: FLOPs actually executed in the matmul-like work (recurrent + output), flops[1]: O(H)/O(T) work."""
    n, T = y.shape
    H = Wh.shape[0]
    O = Wo.shape[0]
    step_f = 2 * H * H + 2 * H
    upd_f = 6 * H
    mass = np.zeros(n); cost = np.zeros(n); cells = np.zeros(n, np.int64)
    Z = np.zeros((T, H)); DZ = np.zeros((T, H))
    Zu = np.zeros((T, H)); Zn = np.zeros((T, H)); DZn = np.zeros((T, H))
    h = np.zeros(H); dh = np.zeros(H); z = np.zeros(H); dz = np.zeros(H)
    L = np.zeros(O); dL = np.zeros(O)
    D = np.zeros(T)
    ne = 0
    cap = ev.shape[0]
    fl_mm = 0.0; fl_lin = 0.0
    for row in range(n):
        ai = a[row]; bi = b[row]
        for s in range(T):
            D[s] = 0.0
        D[ai] = 1 / math.sqrt(2.0); D[bi] = -1 / math.sqrt(2.0)
        if is23:
            kpos = ai; kneg = bi
        else:
            kpos = bi; kneg = ai
        amin = min(ai, bi)
        taur = tau[row]
        t = -taur
        first = True
        TC = 0.0
        it = 0
        while t < taur and it < max_cells:
            it += 1
            tc = t + eps * (1 + abs(t))
            r0 = 0
            if not first:
                dt = tc - TC
                r0 = T
                for s in range(T):
                    hit = False
                    for i in range(H):
                        zu = Z[s, i] + dt * DZ[s, i]
                        Zu[s, i] = zu
                        if (zu > 0) != (Z[s, i] > 0):
                            hit = True
                    if hit and r0 == T:
                        r0 = s
                fl_lin += T * 2 * H
            for i in range(H):
                h[i] = 0.0; dh[i] = 0.0
            nxt = taur; rstep = T
            for s in range(T):
                if r0 <= s:
                    xs = y[row, s] + tc * D[s]
                    for i in range(H):
                        acc = 0.0
                        for j in range(H):
                            acc += h[j] * Wh[i, j]
                        z[i] = acc + xs * Wi[i]
                    if s >= amin:
                        for i in range(H):
                            acc = 0.0
                            for j in range(H):
                                acc += dh[j] * Wh[i, j]
                            dz[i] = acc + D[s] * Wi[i]
                        fl_mm += 2 * (2 * H * H + 2 * H)
                    else:
                        for i in range(H):
                            dz[i] = 0.0
                        fl_mm += 2 * H * H + 2 * H
                else:
                    for i in range(H):
                        z[i] = Zu[s, i]; dz[i] = DZ[s, i]
                for i in range(H):
                    Zn[s, i] = z[i]; DZn[s, i] = dz[i]
                    r = tc - z[i] / dz[i]
                    if r > tc and r < np.inf:
                        if r < nxt:
                            nxt = r; rstep = s
                    if z[i] > 0:
                        h[i] = z[i]; dh[i] = dz[i]
                    else:
                        h[i] = 0.0; dh[i] = 0.0
                fl_lin += 5 * H
            for s in range(T):
                for i in range(H):
                    Z[s, i] = Zn[s, i]; DZ[s, i] = DZn[s, i]
            TC = tc
            first = False
            for o in range(O):
                acc = 0.0; dacc = 0.0
                for j in range(H):
                    acc += h[j] * Wo[o, j]
                    dacc += dh[j] * Wo[o, j]
                L[o] = acc; dL[o] = dacc
            fl_mm += 4 * O * H
            nrec = T - r0
            ntan = T - max(r0, amin)
            cost[row] += ((nrec + ntan) * step_f + T * upd_f + T * 4 * H + 4 * T * H) / (T * step_f)
            cells[row] += 1
            lo = t; hi = nxt
            Lmax = 0.0
            for o in range(O):
                if abs(L[o]) > Lmax:
                    Lmax = abs(L[o])
            for side in range(2):
                if side == 0:
                    p_lo = lo; p_hi = min(hi, 0.0); k = kneg
                else:
                    p_lo = max(lo, 0.0); p_hi = hi; k = kpos
                ok = p_hi > p_lo
                lbm = -np.inf; ubm = np.inf; bad = False
                for j in range(O):
                    if j == k:
                        continue
                    c = (L[k] - L[j]) - (dL[k] - dL[j]) * tc
                    e = dL[k] - dL[j]
                    if e > 0:
                        rt = -c / e
                        if rt > lbm:
                            lbm = rt
                    elif e < 0:
                        rt = -c / e
                        if rt < ubm:
                            ubm = rt
                    else:
                        if c < 0 or (abs(c) <= 1e-12 * (1 + Lmax) and j < k):
                            bad = True
                fl_lin += 8 * O
                clo = max(lbm, p_lo); chi = min(ubm, p_hi)
                valid = (chi > clo) and not bad
                if fast:
                    if ok:
                        tot = tcdf_int(p_hi * sq, nu) - tcdf_int(p_lo * sq, nu)
                        corr = 0.0
                        if valid:
                            corr = tcdf_int(chi * sq, nu) - tcdf_int(clo * sq, nu)
                        mass[row] += tot - corr
                else:
                    if ne >= cap:
                        return mass, cost, cells, -1
                    ev[ne, 0] = row if ok else -1 - row
                    ev[ne, 1] = p_lo; ev[ne, 2] = p_hi
                    if valid:
                        ev[ne, 3] = clo; ev[ne, 4] = chi
                    else:
                        ev[ne, 3] = np.nan; ev[ne, 4] = np.nan
                    ne += 1
            t = nxt
    flops[0] += fl_mm; flops[1] += fl_lin
    return mass, cost, cells, ne


@njit(cache=True)
def _accumulate(mass, rows, Fhi, Flo, Gchi, Gclo, valid):
    for q in range(len(rows)):
        r = rows[q]
        if r < 0:
            continue                       # ok False: frozen code adds 0.0
        tot = Fhi[q] - Flo[q]
        corr = Gchi[q] - Gclo[q] if valid[q] else 0.0
        mass[r] += tot - corr


FLOPS = np.zeros(2)       # running counters (executed matmul FLOPs, executed O(H) work)
MODE = {"fast": False}


def walk(model, y, a, b, tau, kind, cdf=None, eps=1e-10, max_cells=4000):
    """Drop-in for frozen walk.walk with the radial (Student-t_{T-1}) cdf of hom.tcdf(T) (fast mode) or any
    vectorised cdf (repro mode)."""
    Wi, Wh, Wo = model
    y = np.ascontiguousarray(y, dtype=np.float64)
    n, T = y.shape
    nu = T - 1
    a = np.ascontiguousarray(a, dtype=np.int64); b = np.ascontiguousarray(b, dtype=np.int64)
    tau = np.ascontiguousarray(tau, dtype=np.float64)
    fast = MODE["fast"]
    cap = 1 if fast else max(64, 2 * n * 12)
    while True:
        ev = np.empty((cap, 5))
        mass, cost, cells, ne = _walk(Wi, Wh, Wo, y, a, b, tau, kind == "23", eps, max_cells, fast, nu,
                                      math.sqrt(nu), ev, FLOPS)
        if ne >= 0:
            break
        cap *= 4
    if fast:
        return mass, cost, cells
    ev = ev[:ne]
    rows = ev[:, 0].astype(np.int64)
    valid = ~np.isnan(ev[:, 3])
    F = cdf(ev[:, 1:3])                             # cdf(p_lo), cdf(p_hi)
    G = np.where(valid[:, None], cdf(np.where(valid[:, None], ev[:, 3:5], 0.0)), 0.0)
    _accumulate(mass, rows, F[:, 1], F[:, 0], G[:, 1], G[:, 0], valid)
    return mass, cost, cells
