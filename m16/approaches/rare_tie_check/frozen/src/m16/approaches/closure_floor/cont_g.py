"""T317g (T339): T317f (cont_f.py @88453669, sha256 327dbb4a...) with ONE change, the closure's low pattern.
In T317f the 6 lows share one value mu(v4) (linearised along the walk). Here the low at absolute position j is
mu(v4) + SPREAD * (-1)^j * sigma(v4), with sigma = sd(N(0,1) | < v4) linearised along the walk the same way as mu
(model-free constants per region, computed once at import). SPREAD = 0 is T317f exactly. Everything else (G, TMAX,
screen C1, walk, merge, profile, FLOP charges) is T317f's code, unchanged. Self-contained; no Monte Carlo.
est(model, budget=None) -> (estimate, FLOPs as executed under Directive 8).

T317f's docstring follows.
T317f (T335): T317e plus the declared screen C1 on all three lines, as one self-contained file with no Monte Carlo
anywhere. T317e's code is copied verbatim; walk_c1 and est are added.

Screen C1 (the 7th declared approximation): at the walk start and at every gate flip, after the restart has recomputed
the row's tangents and dL, a row stops walking if its current cell's affine (frozen-gate) margin is positive over the
whole remaining window: min_{j != k} [m_kj(t) + (tmax - t) min(0, dm_kj)] > 0. It then gets one correct piece
[t, tmax], and skips its roots, its correct set and every later cell. This is a heuristic: gate flips later in the
window could make some of that segment wrong. Its error is measured on DEV2 (declared d vs T317e: RMS .077 pp, mean
+.045 pp, max .34 pp), not bounded.
Charges as executed: per screen try, L(t) affine 2C plus margins and min 4C + 2.
est(model, budget=None) -> (estimate, FLOPs as executed under Directive 8). Sampling-free: no random draws."""
from itertools import permutations

import numpy as np
from scipy.special import ndtr

DEDUCTIVE = True
SPREAD = 0.0                                  # T339 low pattern sign: +1, -1, or 0 (= T317f); set at freeze
H, T, C = 16, 10, 10
G = 20                                        # Gauss-Legendre nodes per v2 region
TMAX = 0.5                                    # walk window (r1w2.TMAX)
INF = 1e9                                     # marker for an interval extending to infinity (walk.INF)
EPS = 1e-10
REGIONS = (("nb", 0.0, 6.5), ("b", -6.5, 0.0))
TRI = np.array(list(permutations(range(T), 3)))                                # 720 x (p1, p2, p3)
NT = len(TRI)
FREE = np.array([[p for p in range(T) if p not in t] for t in TRI])            # (720, 7) free positions, ascending
P4 = FREE[:, -1]                                                               # declared v4 position: the latest
LOWS = FREE[:, :-1]                                                            # the other 6

# FLOP charges (identical to the T317 / T317c chain)
F_STEP = (2 * H * H + 2 * H) + H + H + (2 * H * H + 2 * H) + H + 2 * H + 2 * H + H + 2      # = 1218 (v1b2)
F_CELL_T317 = 2 * 2 * C * H + 14 * C + 6 + 4                                                # = 790 (v1b2.F_CELL)
F_TAN = 2 * H * H + 2 * H + H                 # one tangent step (matvec + input + mask) = 560
F_ROOT = 5 * H                                # roots of one step: sub/div (2H), >t and finite (2H), min (H)
F_LATER_READ_T317 = 2 * C + 2 * C * H + 146 + 4
F_CI = 229                                    # correct_interval (official count, T281 scorer 379498b6)
F_CELL = 2 * 2 * C * H + F_CI + 4             # first-cell readout + correct set + bookkeeping
F_LATER_READ = 2 * C + 2 * C * H + F_CI + 4
F_PHI = 20
_NORM_PDF_C = np.sqrt(2 * np.pi)


def _pdf(x):
    """Standard normal density, scipy.stats.norm.pdf's formula (scipy.stats is not imported: it draws random numbers
    at import to build docstring examples)."""
    return np.exp(-x ** 2 / 2.0) / _NORM_PDF_C


def mu(v3):
    return -_pdf(v3) / np.maximum(ndtr(v3), 1e-300)


def dmu(v):
    m = mu(np.atleast_1d(v))
    return -np.atleast_1d(v) * m + m * m


def gauss_v4(v3, K):
    """K-node Gauss rule for the law of v4 = max of 7 iid N(0,1) | < v3 (Stieltjes on a fine model-free grid)."""
    v = np.linspace(v3 - 8, v3, 40001)[:-1]
    w = _pdf(v) * ndtr(v) ** 6; w /= w.sum()
    a, b = np.zeros(K), np.zeros(K); p_prev = np.zeros_like(v); p = np.ones_like(v)
    for j in range(K):
        nrm = (w * p * p).sum(); a[j] = (w * v * p * p).sum() / nrm
        if j > 0:
            b[j] = nrm / nrm_prev
        p_new = (v - a[j]) * p - (b[j] if j > 0 else 0) * p_prev
        p_prev, p, nrm_prev = p, p_new, nrm
    J = np.diag(a) + np.diag(np.sqrt(b[1:]), 1) + np.diag(np.sqrt(b[1:]), -1)
    x, U = np.linalg.eigh(J)
    return x, U[0] ** 2


def sigma(v):
    """sd of N(0,1) | < v: sqrt(1 + v mu - mu^2)."""
    v = np.atleast_1d(v); m = mu(v)
    return np.sqrt(np.maximum(1 + v * m - m * m, 0.0))


def dsigma(v):
    v = np.atleast_1d(v); m = mu(v); dm = dmu(v)
    return (m + v * dm - 2 * m * dm) / (2 * sigma(v))


PAR = (-1.0) ** np.arange(10)                 # (-1)^j by absolute position j


def inputs(v1, v2, v3, v4, low, d1, d3, d4, dlow, sig=0.0, dsig=0.0, spread=0.0):
    """T317f's inputs; T339: the low at position j gets + spread (-1)^j sig (slope + spread (-1)^j dsig)."""
    n = NT; r = np.arange(n)
    x = np.zeros((n, 10)); dx = np.zeros((n, 10))
    for cols, val, dv in ((TRI[:, 0], v1, d1), (TRI[:, 1], v2, 0.0), (TRI[:, 2], v3, d3), (P4, v4, d4)):
        x[r, cols] = val; dx[r, cols] = dv
    for j in range(6):
        pj = PAR[LOWS[:, j]] * spread
        x[r, LOWS[:, j]] = low + pj * sig; dx[r, LOWS[:, j]] = dlow + pj * dsig
    return x, dx


def correct_interval(L, dL, tc, k, lo, hi):
    """Correct set of target k within [lo, hi] for logits L + (t - tc) dL (an interval; numpy first-index ties)."""
    rr = np.arange(len(L))
    c = (L[rr, k][:, None] - L) - (dL[rr, k][:, None] - dL) * tc[:, None]
    e = dL[rr, k][:, None] - dL
    with np.errstate(divide="ignore", invalid="ignore"):
        root = -c / e
    lb = np.where(e > 0, root, -np.inf); ub = np.where(e < 0, root, np.inf)
    jj = np.arange(L.shape[1])[None]
    bad0 = (e == 0) & ((c < 0) | ((np.abs(c) <= 1e-12 * (1 + np.abs(L).max(1))[:, None]) & (jj < k[:, None])))
    lb[rr, k] = -np.inf; ub[rr, k] = np.inf
    clo = np.maximum(lb.max(1), lo); chi = np.minimum(ub.min(1), hi)
    ok = (chi > clo) & ~bad0.any(1)
    return np.where(ok, clo, 0.0), np.where(ok, chi, 0.0)


def _gl(lo, hi, G):
    x, w = np.polynomial.legendre.leggauss(G)
    return lo + (hi - lo) * (x + 1) / 2, (hi - lo) / 2 * w


def first_cell(model, x0, d):
    """Prefix-tree first cell at tc = EPS. Returns per row Z (n,T,H) at tc, DZ, G, per-step min roots R (n,T) and their
    neuron J (n,T), plus the node count (executed step-evaluations)."""
    Wi, Wh, Wo = model; n = len(x0); tc = EPS
    inv_prev = np.zeros(n, int); h_prev = np.zeros((1, H)); dh_prev = np.zeros((1, H)); nodes = 0
    Z = np.zeros((n, T, H)); DZ = np.zeros((n, T, H)); G = np.zeros((n, T, H), bool)
    R = np.full((n, T), np.inf); J = np.zeros((n, T), int)
    for s in range(T):
        key = np.round(np.column_stack([inv_prev, x0[:, s], d[:, s]]), 12)
        _, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
        par = inv_prev[rep]; nodes += len(rep)
        z = h_prev[par] @ Wh.T + (x0[rep, s] + tc * d[rep, s])[:, None] * Wi[None]
        dz = dh_prev[par] @ Wh.T + d[rep, s][:, None] * Wi[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = tc - z / dz
        r = np.where((r > tc) & np.isfinite(r), r, np.inf)
        g = z > 0
        Z[:, s] = z[inv]; DZ[:, s] = dz[inv]; G[:, s] = g[inv]; R[:, s] = r.min(1)[inv]; J[:, s] = r.argmin(1)[inv]
        h_prev, dh_prev, inv_prev = np.where(g, z, 0.0), np.where(g, dz, 0.0), inv
    return Z, DZ, G, R, J, nodes


def walk_cont(model, x0, d, k, tmax, max_cells=400):
    Wi, Wh, Wo = model; n = len(x0); pieces = []
    Z, DZ, G, R, J, nodes = first_cell(model, x0, d)
    flops = nodes * F_STEP + n * F_CELL_T317
    tref = np.full(n, EPS)                                  # parameter at which Z holds the values
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)
        pieces.append((act.copy(), lo, hi))
        t[act] = nxt
        go = act[nxt < tmax]; ss = sstar[nxt < tmax]
        if not len(go):
            break
        tn = t[go]
        # values at t* from the affine form; tref -> t*; L(t*) affine
        Z[go] += (tn - tref[go])[:, None, None] * DZ[go]; L[go] += (tn - tref[go])[:, None] * dL[go]; tref[go] = tn
        jj = J[go, ss]; G[go, ss, jj] = ~G[go, ss, jj]
        # tangents of steps > s*, roots of steps >= s* (flipped neuron excluded at its own step)
        for idx in range(len(go)):
            row, s0 = go[idx], ss[idx]
            dh = np.where(G[row, s0], DZ[row, s0], 0.0)
            for s in range(s0 + 1, T):
                DZ[row, s] = Wh @ dh + d[row, s] * Wi
                dh = np.where(G[row, s], DZ[row, s], 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tn[idx] - Z[row, s0:] / DZ[row, s0:]
            r = np.where((r > tn[idx] + 1e-12 * (1 + abs(tn[idx]))) & np.isfinite(r), r, np.inf)
            r[0, jj[idx]] = np.inf
            R[row, s0:] = r.min(1); J[row, s0:] = r.argmin(1)
            dL[row] = Wo @ dh
            flops += 2 * T * H + 1 + H + (T - 1 - s0) * F_TAN + (T - s0) * F_ROOT + F_LATER_READ_T317
        act = go
    return pieces, flops


def walk_counted(model, x0, d, k, tmax):
    """walk_cont (T317 charges) with the honest correct_interval count (same executed arithmetic; FLOPs recounted)."""
    pieces, f_old = walk_cont(model, x0, d, k, tmax)
    n_first = len(x0); n_later = sum(len(p[0]) for p in pieces) - n_first
    f = f_old + n_first * (F_CELL - F_CELL_T317) + n_later * (F_LATER_READ - F_LATER_READ_T317)
    return pieces, f


def merge(pieces, n, tmax):
    """Per row, merge touching non-empty correct pieces in walk order; a piece ending at tmax merges with the model-free
    tail. Returns (row, a, b) arrays with b = INF for merged tails, and the comparison count."""
    cur_a = np.full(n, np.nan); cur_b = np.full(n, np.nan); out_r, out_a, out_b = [], [], []; ops = 0
    for rows, lo, hi in pieces:
        m = hi > lo; r, a, b = rows[m], lo[m], hi[m]; ops += 2 * len(rows)
        touch = cur_b[r] == a
        cur_b[r[touch]] = b[touch]
        new = ~touch
        old = r[new][~np.isnan(cur_a[r[new]])]
        out_r.append(old); out_a.append(cur_a[old]); out_b.append(cur_b[old])
        cur_a[r[new]] = a[new]; cur_b[r[new]] = b[new]
    open_ = np.flatnonzero(~np.isnan(cur_a))
    out_r.append(open_); out_a.append(cur_a[open_]); out_b.append(cur_b[open_])
    R = np.concatenate(out_r); A = np.concatenate(out_a); B = np.concatenate(out_b)
    reach = B >= tmax                                     # merges with the tail [tmax, inf)
    B = np.where(reach, INF, B)
    tail_rows = np.setdiff1d(np.arange(n), R[reach])      # rows whose last piece does not reach tmax keep the tail alone
    R = np.concatenate([R, tail_rows]); A = np.concatenate([A, np.full(len(tail_rows), tmax)])
    B = np.concatenate([B, np.full(len(tail_rows), INF)])
    return R, A, B, ops


def profile(R, A, B, n, nodes, kind, tmax, phi0):
    """Sum over merged intervals; Phi at model-free arguments (a = 0 -> phi0 table; b = inf; a = tmax) is not charged."""
    out = np.zeros((n, len(nodes))); f = 0.0
    a0 = A == 0.0; binf = B >= INF - 1e-12; atm = A == tmax
    # evaluate Phi only where the argument is weight-dependent (masked BEFORE the ndtr call: savings as executed)
    Pa = np.empty((len(R), len(nodes))); Pb = np.empty((len(R), len(nodes)))
    ia = np.flatnonzero(~a0); ib = np.flatnonzero(~binf)
    if kind == "12":
        Pa[a0] = phi0["p"][None]; Pa[ia] = ndtr(nodes[None] + A[ia, None])
        Pb[binf] = 1.0; Pb[ib] = ndtr(nodes[None] + B[ib, None])
        val = Pb - Pa
    else:
        Pa[a0] = phi0["m8"][None]; Pa[ia] = ndtr(nodes[None] - A[ia, None]) ** 8
        Pb[binf] = 0.0; Pb[ib] = ndtr(nodes[None] - B[ib, None]) ** 8
        val = Pa - Pb
    np.add.at(out, R, val)
    nphi = len(ia) + len(ib)              # Phi calls actually made (a = tmax tail starts are evaluated, so charged)
    f += len(nodes) * (nphi * (F_PHI + (3 if kind == "23" else 0)) + 2 * len(R))
    return out, f


def exact_points(lo, hi, n=400_001):
    """E[v2], E[v1 - v2], E[v2 - v3] for the top-3 of 10 iid N(0,1) given v2 in (lo, hi], by deterministic quadrature.
    Density of v2: 90 phi (1 - Phi) Phi^8. Given v2: E[v1 - v2] = (phi - v2 (1 - Phi)) / (1 - Phi) (closed form);
    E[v2 - v3] = int_{-inf}^{v2} (Phi(v) / Phi(v2))^8 dv (computed as a cumulative integral on a fine grid)."""
    v = np.concatenate([np.linspace(-14.0, lo, n), np.linspace(lo, hi, n)[1:]])      # lo is a grid node
    P = ndtr(v); p = _pdf(v); Q = ndtr(-v)
    P8 = P ** 8; dv = np.diff(v)
    cum8 = np.concatenate([[0.0], np.cumsum((P8[1:] + P8[:-1]) * dv / 2)])        # int_{-14}^{v} Phi^8 (tail < 1e-40)
    f = 90 * p * Q * P8
    w = np.zeros(len(v)); w[n - 1:-1] += dv[n - 1:] / 2; w[n:] += dv[n - 1:] / 2    # trapezoid weights on [lo, hi]
    Z = (w * f).sum()
    v2 = (w * f * v).sum() / Z
    g12 = (w * 90 * p * P8 * (p - v * Q)).sum() / Z
    g23 = (w * 90 * p * Q * cum8).sum() / Z
    return float(v2), float(g12), float(g23)


def _tables(spread=None):
    spread = SPREAD if spread is None else spread
    out = []
    for name, lo, hi in REGIONS:
        v2s, g12s, g23s = exact_points(lo, hi)
        nodes, wts = _gl(lo, hi, G)
        v3s = v2s - g23s; v4s = float(gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
        m4 = float(mu(np.array([v4s]))[0]); dm4 = float(dmu(v4s)[0])
        s4 = float(sigma(v4s)[0]); ds4 = float(dsigma(v4s)[0])
        t = dict(name=name, nodes=nodes, wts=wts, phi0=dict(p=ndtr(nodes), m8=ndtr(nodes) ** 8),
                 blind12=1.0 - ndtr(nodes), dens=90 * _pdf(nodes), point=(v2s, g12s, g23s), v4=v4s, m4=m4, dm4=dm4)
        t["line23"] = inputs(v2s + g12s, v2s, v2s, v2s - delta, m4 + dm4 * (v2s - delta - v4s), 0.0, -1.0, -1.0, -dm4,
                             sig=s4 + ds4 * (v2s - delta - v4s), dsig=-ds4, spread=spread)
        if name.startswith("nb"):
            t["line12"] = inputs(v2s, v2s, v3s, v4s, m4, 1.0, 0.0, 0.0, 0.0, sig=s4, dsig=0.0, spread=spread)
        out.append(t)
    return out


TABLES = _tables()                                        # model-free, computed once at import, uncharged


def table_sizes(tables=None):
    rows = []
    for t in tables or TABLES:
        for k, v in t.items():
            if k == "name":
                continue
            if isinstance(v, dict):
                rows += [(t["name"], f"{k}.{kk}", np.size(vv)) for kk, vv in v.items()]
            elif isinstance(v, tuple) and k.startswith("line"):
                rows += [(t["name"], f"{k}.x", v[0].size), (t["name"], f"{k}.dx", v[1].size)]
            else:
                rows.append((t["name"], k, int(np.size(v))))
    return rows


F_TRY = 2 * C + 4 * C + 2                     # L(t) affine + margins / slope / min + compare


def walk_c1(model, x0, d, k, tmax, max_cells=400, stats=None, tag=""):
    """walk_cont (walk_counted's charges) with screen C1 at the walk start and at every flip."""
    Wi, Wh, Wo = model; n = len(x0); pieces = []
    Z, DZ, G, R, J, nodes = first_cell(model, x0, d)
    if stats is not None:                     # T339 diagnostic only: prefix-tree node count per line
        stats[tag] = nodes
    flops = nodes * F_STEP + n * F_CELL
    tref = np.full(n, EPS)
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0

    def safe(row):
        nonlocal flops
        delta = tmax - t[row]; kk = k[row]
        Lt = L[row] + (t[row] - tref[row]) * dL[row]; flops += 2 * C
        m = Lt[kk] - Lt; dm = dL[row, kk] - dL[row]; m[kk] = np.inf; dm[kk] = 0.0
        flops += 4 * C + 2
        return bool((m + delta * np.minimum(0.0, dm)).min() > 0)
    s0 = np.array([safe(r_) for r_ in act])
    if s0.any():
        sr = act[s0]; pieces.append((sr, t[sr].copy(), np.full(len(sr), tmax))); act = act[~s0]
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)
        pieces.append((act.copy(), lo, hi))
        t[act] = nxt
        keep = nxt < tmax; go = act[keep]; ss = sstar[keep]
        if not len(go):
            break
        tn = t[go]
        Z[go] += (tn - tref[go])[:, None, None] * DZ[go]; L[go] += (tn - tref[go])[:, None] * dL[go]; tref[go] = tn
        jj = J[go, ss]; G[go, ss, jj] = ~G[go, ss, jj]
        alive = np.ones(len(go), bool)
        for idx in range(len(go)):
            row, s0_ = go[idx], ss[idx]
            dh = np.where(G[row, s0_], DZ[row, s0_], 0.0)
            for s in range(s0_ + 1, T):
                DZ[row, s] = Wh @ dh + d[row, s] * Wi
                dh = np.where(G[row, s], DZ[row, s], 0.0)
            dL[row] = Wo @ dh
            flops += 2 * T * H + 1 + H + (T - 1 - s0_) * F_TAN + 2 * C * H
            if safe(row):
                alive[idx] = False; continue
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tn[idx] - Z[row, s0_:] / DZ[row, s0_:]
            r = np.where((r > tn[idx] + 1e-12 * (1 + abs(tn[idx]))) & np.isfinite(r), r, np.inf)
            r[0, jj[idx]] = np.inf
            R[row, s0_:] = r.min(1); J[row, s0_:] = r.argmin(1)
            flops += (T - s0_) * F_ROOT + F_LATER_READ - 2 * C * H
        if (~alive).any():
            sr = go[~alive]; pieces.append((sr, t[sr].copy(), np.full(len(sr), tmax)))
        go = go[alive]
        act = go
    return pieces, flops


def est(model, budget=None, tables=None, stats=None):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = TRI[:, 1]; flops = 0.0; tot = np.zeros(NT)
    for t in (TABLES if tables is None else tables):
        nodes = t["nodes"]; I = {}
        for kind in ("23", "12"):
            if f"line{kind}" not in t:
                continue
            x, dx = t[f"line{kind}"]
            p, f = walk_c1(model, x, dx, k, TMAX, stats=stats, tag=f"{t['name']}-{kind}"); flops += f
            R, A, B, ops = merge(p, NT, TMAX); flops += ops
            I[kind], f = profile(R, A, B, NT, nodes, kind, TMAX, t["phi0"]); flops += f
        I12 = I["12"] if "12" in I else np.repeat(t["blind12"][None], NT, 0)    # blind: model-free (V1)
        tot += (I12 * I["23"] * t["dens"][None]) @ t["wts"]
        flops += NT * len(nodes) * 4
    flops += NT
    return float(tot.mean()), float(flops)
