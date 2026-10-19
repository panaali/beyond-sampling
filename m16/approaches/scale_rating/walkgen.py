"""T427: the frozen graded-g12 estimator (estimator_t366, claude/t218-tie-tail-labels @5b4a1250) generalised to any (H, T).
The frozen chain is t366.est_e1 -> t354.walk_seg -> walkfix.walk3_fixed (replacing rare.walk3) -> cont_g (first_cell,
correct_interval, inputs, TABLES) with rare.mass / rare._cond_g23 and t340c.gauss_g12 (claude/t286-cheap-frontier
@620eb47b). Those files hard-code H = 16, T = 10. This file copies each used function with ONLY these changes:
  - H, T, C = T, TRI / FREE / P4 / LOWS / PAR built from T; the T-4 lows (was 6) in inputs, x of width T (was 10);
  - order-statistic law of T iid N(0,1): density T(T-1) phi (1 - Phi) Phi^(T-2) (was 90 ... Phi^8), g23 | v2 law
    Phi^(T-2) (was ^8, in mass, _cond_g23, exact_points), v4 = max of T-3 lows (gauss_v4 weight Phi^(T-4), was ^6);
  - est_e1 runs rows in chunks of ROW_BUDGET / (T H) rows (T407's declared deviation: 4 v2 nodes at (16, 10)).
FLOP charges are the frozen formulas in H, C, T. At (16, 10) this must reproduce the frozen T407 wrapper's estimates and
flag shares (check_equal.py). Deductive: no random draws."""
from itertools import permutations

import numpy as np
from scipy.special import ndtr

EPS = 1e-10
G_NODES = 20
REGIONS = (("nb", 0.0, 6.5), ("b", -6.5, 0.0))
TBIG = 6.0
F_CI = 229
F_PHI = 20
_NORM_PDF_C = np.sqrt(2 * np.pi)
PANELS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.07), (0.07, 0.15), (0.15, 0.4)]
A_TAIL, K_TAIL = 0.4, 3
NN = 2 * len(PANELS) + K_TAIL
_gx, _gw = np.polynomial.legendre.leggauss(2)
ROW_BUDGET = 4 * NN * 720 * 10 * 16          # T407's chunk (4 v2 nodes at H 16, T 10) in rows x T x H


def _pdf(x):
    return np.exp(-x ** 2 / 2.0) / _NORM_PDF_C


def mu(v3):
    return -_pdf(v3) / np.maximum(ndtr(v3), 1e-300)


def dmu(v):
    m = mu(np.atleast_1d(v))
    return -np.atleast_1d(v) * m + m * m


def _gl(lo, hi, G):
    x, w = np.polynomial.legendre.leggauss(G)
    return lo + (hi - lo) * (x + 1) / 2, (hi - lo) / 2 * w


def correct_interval(L, dL, tc, k, lo, hi):
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


def gauss_g12(v2, a, K, n=40001):
    g = np.linspace(a, a + 12.0, n); w = _pdf(v2 + g); w = w / w.sum()
    al, be = np.zeros(K), np.zeros(K); p_prev = np.zeros_like(g); p = np.ones_like(g); nrm_prev = 1.0
    for j in range(K):
        nrm = (w * p * p).sum(); al[j] = (w * g * p * p).sum() / nrm
        if j > 0:
            be[j] = nrm / nrm_prev
        p_new = (g - al[j]) * p - (be[j] if j > 0 else 0) * p_prev
        p_prev, p, nrm_prev = p, p_new, nrm
    Jm = np.diag(al) + np.diag(np.sqrt(be[1:]), 1) + np.diag(np.sqrt(be[1:]), -1)
    x, U = np.linalg.eigh(Jm)
    return x, U[0] ** 2


def graded_g12(v2):
    norm_ = ndtr(-v2); xs, ws = [], []
    for a, b in PANELS:
        g = a + (b - a) * (_gx + 1) / 2; xs += list(g); ws += list((b - a) / 2 * _gw * _pdf(v2 + g) / norm_)
    gn, gw = gauss_g12(v2, A_TAIL, K_TAIL); tail = ndtr(-(v2 + A_TAIL)) / norm_
    xs += list(gn); ws += list(gw * tail)
    return np.array(xs), np.array(ws)


class Walker:
    """All T- and H-dependent pieces of the frozen chain for one (H, T)."""

    def __init__(self, H, T):
        self.H, self.T, self.C = H, T, T
        C = T
        self.TRI = np.array(list(permutations(range(T), 3))); self.NT = len(self.TRI)
        FREE = np.array([[p for p in range(T) if p not in t] for t in self.TRI])
        self.P4 = FREE[:, -1]; self.LOWS = FREE[:, :-1]
        self.PAR = (-1.0) ** np.arange(T)
        self.F_STEP = (2 * H * H + 2 * H) + H + H + (2 * H * H + 2 * H) + H + 2 * H + 2 * H + H + 2
        self.F_TAN = 2 * H * H + 2 * H + H
        self.F_ROOT = 5 * H
        self.F_CELL = 2 * 2 * C * H + F_CI + 4
        self.F_LATER_READ = 2 * C + 2 * C * H + F_CI + 4
        self.TABLES = self._tables()
        self.LINES = self._node_lines_graded()

    # --- cont_g ---
    def gauss_v4(self, v3, K):
        v = np.linspace(v3 - 8, v3, 40001)[:-1]
        w = _pdf(v) * ndtr(v) ** (self.T - 4); w /= w.sum()
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

    def inputs(self, v1, v2, v3, v4, low, d1, d3, d4, dlow):
        """cont_g.inputs with spread = 0 (T317f; the frozen chain only uses spread 0)."""
        n = self.NT; r = np.arange(n); TRI = self.TRI
        x = np.zeros((n, self.T)); dx = np.zeros((n, self.T))
        for cols, val, dv in ((TRI[:, 0], v1, d1), (TRI[:, 1], v2, 0.0), (TRI[:, 2], v3, d3), (self.P4, v4, d4)):
            x[r, cols] = val; dx[r, cols] = dv
        for j in range(self.T - 4):
            pj = self.PAR[self.LOWS[:, j]] * 0.0
            x[r, self.LOWS[:, j]] = low + pj * 0.0; dx[r, self.LOWS[:, j]] = dlow + pj * 0.0
        return x, dx

    def first_cell(self, model, x0, d):
        Wi, Wh, Wo = model; n = len(x0); tc = EPS; T, H = self.T, self.H
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

    def exact_points(self, lo, hi, n=400_001):
        T = self.T; c2 = T * (T - 1)
        v = np.concatenate([np.linspace(-14.0, lo, n), np.linspace(lo, hi, n)[1:]])
        P = ndtr(v); p = _pdf(v); Q = ndtr(-v)
        P8 = P ** (T - 2); dv = np.diff(v)
        cum8 = np.concatenate([[0.0], np.cumsum((P8[1:] + P8[:-1]) * dv / 2)])
        f = c2 * p * Q * P8
        w = np.zeros(len(v)); w[n - 1:-1] += dv[n - 1:] / 2; w[n:] += dv[n - 1:] / 2
        Z = (w * f).sum()
        v2 = (w * f * v).sum() / Z
        g12 = (w * c2 * p * P8 * (p - v * Q)).sum() / Z
        g23 = (w * c2 * p * Q * cum8).sum() / Z
        return float(v2), float(g12), float(g23)

    def _tables(self):
        out = []; c2 = self.T * (self.T - 1)
        for name, lo, hi in REGIONS:
            nodes, wts = _gl(lo, hi, G_NODES)
            out.append(dict(name=name, nodes=nodes, wts=wts, blind12=1.0 - ndtr(nodes), dens=c2 * _pdf(nodes)))
        return out

    # --- rare ---
    def _cond_g23(self, v2, n=20001):
        v = np.linspace(v2 - 12, v2, n); f = (ndtr(v) / ndtr(v2)) ** (self.T - 2)
        return float(((f[1:] + f[:-1]) / 2 * np.diff(v)).sum())

    def mass(self, pieces, nrow, node_of_row, nodes, t0):
        out = np.zeros((nrow, len(nodes))); f = 0.0; e = self.T - 2
        for rows, lo, hi in pieces:
            a = np.maximum(lo, t0); m = hi > a
            if not m.any():
                continue
            r, a, b = rows[m], a[m], hi[m]
            j = node_of_row[r]; nd = nodes[j]
            val = ndtr(nd - a) ** e - ndtr(nd - b) ** e
            np.add.at(out, (r, j), val); f += len(r) * (2 * (F_PHI + 3) + 2)
        return out, f

    # --- walkfix (pieces_c dropped: E1 only uses pieces; its charges are kept as in the frozen code) ---
    def walk3_fixed(self, model, x0, d, k, k3, tmax, max_cells=5000):
        Wi, Wh, Wo = model; n = len(x0); pieces = []; T, H = self.T, self.H
        Z, DZ, Gt, R, J, nodes = self.first_cell(model, x0, d)
        flops = nodes * self.F_STEP + n * self.F_CELL
        tref = np.full(n, EPS)
        hT = np.where(Gt[:, -1], Z[:, -1], 0.0); dhT = np.where(Gt[:, -1], DZ[:, -1], 0.0)
        L = hT @ Wo.T; dL = dhT @ Wo.T
        t = np.zeros(n); act = np.arange(n); it = 0; ncell = 0
        while len(act) and it < max_cells:
            it += 1
            sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
            lo, hi = correct_interval(L[act], dL[act], tref[act], k3[act], t[act], nxt)
            flops += len(act) * F_CI; ncell += len(act)
            pieces.append((act.copy(), lo, hi))
            t[act] = nxt
            go = act[nxt < tmax]; ss = sstar[nxt < tmax]
            if not len(go):
                break
            tn = t[go]
            Z[go] += (tn - tref[go])[:, None, None] * DZ[go]; L[go] += (tn - tref[go])[:, None] * dL[go]; tref[go] = tn
            for idx in range(len(go)):
                row, s0 = go[idx], ss[idx]
                dz = DZ[row, s0]
                for s in range(s0, T):
                    if s > s0:
                        dz = Wh @ dh + d[row, s] * Wi
                    z = Z[row, s]; tol = 1e-9 * (1.0 + np.abs(z).max())
                    g = np.where(np.abs(z) > tol, z > 0, dz > 0)
                    Gt[row, s] = g; DZ[row, s] = dz; dh = np.where(g, dz, 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = tn[idx] - Z[row, s0:] / DZ[row, s0:]
                ztol = 1e-9 * (1.0 + np.abs(Z[row, s0:]).max(1, keepdims=True))
                r = np.where((r > tn[idx] + 1e-12 * (1 + abs(tn[idx]))) & np.isfinite(r) & (np.abs(Z[row, s0:]) > ztol), r, np.inf)
                R[row, s0:] = r.min(1); J[row, s0:] = r.argmin(1)
                dL[row] = Wo @ dh
                flops += 2 * T * H + 1 + H + (T - 1 - s0) * self.F_TAN + (T - s0) * (self.F_ROOT + 2 * H) + self.F_LATER_READ
            act = go
        return pieces, flops, it, ncell

    # --- t354 ---
    def walk_seg(self, model, x, dx, k, k3, t_start, tmax):
        p, f, it, nc = self.walk3_fixed(model, x + t_start * dx, dx, k, k3, tmax - t_start)
        return [(r, lo + t_start, hi + t_start) for r, lo, hi in p], f, it

    # --- t366 ---
    def _node_lines_graded(self):
        out = []
        for t in self.TABLES:
            xs, dxs, om = [], [], []
            for v2 in t["nodes"]:
                v3s = v2 - self._cond_g23(v2); v4s = float(self.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
                m4 = float(mu(np.array([v4s]))[0]); dm4 = float(dmu(v4s)[0])
                gn, gw = graded_g12(v2)
                for g12, w in zip(gn, gw):
                    x, dx = self.inputs(v2 + g12, v2, v2, v2 - delta, m4 + dm4 * (v2 - delta - v4s), 0.0, -1.0, -1.0, -dm4)
                    xs.append(x); dxs.append(dx); om.append(np.full(self.NT, w))
            out.append((np.concatenate(xs), np.concatenate(dxs), np.concatenate(om)))
        return out

    def est_with_flag(self, model, t0s, tmax=TBIG, log=None):
        """T407's est_with_flag: E1(t0) = P(pick = p3, g23 > t0) for each t0, the share of it from g12 nodes < .1
        (deterministic trust flag), FLOPs as executed, and the max walk iterations (cap 5000)."""
        model = tuple(np.asarray(w, np.float64) for w in model); NT = self.NT; TRI = self.TRI
        res = {t: 0.0 for t in t0s}; low = {t: 0.0 for t in t0s}; flops = 0.0; t_start = min(t0s); itmax = 0
        step = max(NT, int(ROW_BUDGET // (self.T * self.H)) // NT * NT)
        for ri, tab in enumerate(self.TABLES):
            nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = self.LINES[ri]; nrow = len(x)
            k = np.tile(TRI[:, 1], nrow // NT); k3 = np.tile(TRI[:, 2], nrow // NT)
            node_of_row = np.repeat(np.arange(len(nodes)), NN * NT)
            g12_row = np.concatenate([np.repeat(graded_g12(v2)[0], NT) for v2 in nodes])
            is_low = (g12_row < 0.1).astype(float)
            wgt = tab["dens"] * tab["blind12"]
            for a in range(0, nrow, step):
                sl = slice(a, min(a + step, nrow)); n = sl.stop - sl.start
                pieces, f, it = self.walk_seg(model, x[sl], dx[sl], k[sl], k3[sl], t_start, tmax); flops += f
                itmax = max(itmax, it)
                for t0 in t0s:
                    W, f = self.mass(pieces, n, node_of_row[sl], nodes, t0); flops += f
                    res[t0] += float((W * om[sl, None]).sum(0) * wgt @ wts) / NT
                    low[t0] += float((W * (om[sl] * is_low[sl])[:, None]).sum(0) * wgt @ wts) / NT
                    flops += n * len(nodes) * 3
                if log:
                    log(f"region {ri} rows {sl.stop}/{nrow} itmax {itmax}")
        share = {t: (low[t] / res[t] if res[t] > 0 else None) for t in t0s}
        return res, share, flops, itmax

    # --- exact condition probability (T401's p_gap, generalised) ---
    def p_gap(self, t0, grid=None):
        g = np.linspace(-7.0, 10.0, 2_000_001) if grid is None else grid; T = self.T
        f = T * (T - 1) * np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi) * ndtr(-g) * ndtr(g - t0) ** (T - 2)
        return float(((f[1:] + f[:-1]) / 2 * np.diff(g)).sum())
