"""T424: the frozen rare-event estimators generalised to any (H, T). DEDUCTIVE: no random inputs.

Source (frozen, read-only; sha256 asserted by freeze.py): claude/t218-tie-tail-labels @5b4a1250 r1_cont/{t340c, t354,
walkfix, t366, estimator_t366}.py and claude/t286-cheap-frontier @620eb47b rare_events/rare.py, closure_floor/cont_g.py.
This file re-states those functions with every hard-coded (H, T) = (16, 10) constant made a parameter. Nothing else is
changed: same lines, same Gauss rules, same walk (with the T354 simultaneous-flip fix), same closure, same charges.
What depends on T (10 -> T):
  C = T logits; triples TRI = permutations(range(T), 3) (NT = T(T-1)(T-2)); the declared v4 position = the latest free
  one, the other T-4 free positions are the lows; v2 density 90 phi -> T(T-1) phi; the g23 | v2 profile Phi^8 ->
  Phi^(T-2) (rare.mass, rare._cond_g23); v4 = max of T-3 iid N | < v3 (gauss_v4: Phi^6 -> Phi^(T-4)).
What depends on H: only the FLOP charges (F_STEP, F_CELL, F_TAN, F_ROOT, F_LATER_READ, the restart charge) and the
first-cell state width, both read from the model.
Rows are walked in chunks (as rare_noise/t388_chunked.py) to stay under 2 GB; lines are independent rows, so the
result equals the unchunked run up to summation order (checked at (16, 10) by validate.py).
  est_k3(model, t0s)      -> ({t0: est}, FLOPs)   H-R1 / T364 K3: T340c lines (3 exact-law g12 nodes), T_hi = 6
  est_graded(model, t0s)  -> ({t0: est}, FLOPs)   estimator_t366: graded g12 rule (10 GL nodes on [0, .4] + 3 beyond)"""
import os
from itertools import permutations

import numpy as np
from scipy.special import ndtr

DEDUCTIVE = True
G_NODES = 20
EPS = 1e-10
TBIG = 6.0
REGIONS = (("nb", 0.0, 6.5), ("b", -6.5, 0.0))
F_CI = 229
F_PHI = 20
_NORM_PDF_C = np.sqrt(2 * np.pi)
PANELS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.07), (0.07, 0.15), (0.15, 0.4)]
A_TAIL, K_TAIL = 0.4, 3
_gx, _gw = np.polynomial.legendre.leggauss(2)


def _pdf(x):
    return np.exp(-x ** 2 / 2.0) / _NORM_PDF_C


def mu(v3):
    return -_pdf(v3) / np.maximum(ndtr(v3), 1e-300)


def dmu(v):
    m = mu(np.atleast_1d(v))
    return -np.atleast_1d(v) * m + m * m


def charges(H, C):
    F_STEP = (2 * H * H + 2 * H) + H + H + (2 * H * H + 2 * H) + H + 2 * H + 2 * H + H + 2
    return dict(F_STEP=F_STEP, F_TAN=2 * H * H + 2 * H + H, F_ROOT=5 * H, F_CELL=2 * 2 * C * H + F_CI + 4,
                F_LATER_READ=2 * C + 2 * C * H + F_CI + 4)


def fpe(H, T):
    """Directive 7/8 FPE: 5948 FLOPs x flops(H, T) / flops(16, 10), flops = T(2H^2 + 2H) + 2HT (m16/leaderboard.py)."""
    f = lambda h, t: t * (2 * h * h + 2 * h) + 2 * h * t
    return 5948.0 * f(H, T) / f(16, 10)


def correct_interval(L, dL, tc, k, lo, hi):
    """cont_g.correct_interval, unchanged."""
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


def first_cell(model, x0, d, T, H):
    """cont_g.first_cell with (T, H) as parameters."""
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


def walk3_fixed(model, x0, d, k, k3, tmax, max_cells=5000):
    """walkfix.walk3_fixed with (T, H) from the model. Also returns whether max_cells truncated any row (diagnostic)."""
    Wi, Wh, Wo = model; n = len(x0); pieces = []; T, H = Wo.shape[0], Wh.shape[0]; F = charges(H, T)
    Z, DZ, Gt, R, J, nodes = first_cell(model, x0, d, T, H)
    flops = nodes * F["F_STEP"] + n * F["F_CELL"]
    tref = np.full(n, EPS)
    hT = np.where(Gt[:, -1], Z[:, -1], 0.0); dhT = np.where(Gt[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = correct_interval(L[act], dL[act], tref[act], k3[act], t[act], nxt)
        flops += len(act) * F_CI
        pieces.append((act.copy(), lo, hi))
        # (the frozen walk also computes the target-k interval here for E2; E1 does not read it, so it is skipped;
        #  its cost was not charged separately there either: it sits in F_CELL / F_LATER_READ, which are kept)
        t[act] = nxt
        go = act[nxt < tmax]; ss = sstar[nxt < tmax]
        if not len(go):
            act = go
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
            flops += 2 * T * H + 1 + H + (T - 1 - s0) * F["F_TAN"] + (T - s0) * (F["F_ROOT"] + 2 * H) + F["F_LATER_READ"]
        act = go
    return pieces, flops, len(act)


def walk_seg(model, x, dx, k, k3, t_start, tmax):
    """t354.walk_seg: fresh first cell at x + t_start dx, pieces shifted back to absolute t."""
    p, f, trunc = walk3_fixed(model, x + t_start * dx, dx, k, k3, tmax - t_start)
    return [(r, lo + t_start, hi + t_start) for r, lo, hi in p], f, trunc


class Law:
    """Model-free tables and lines for sequence length T (computed once)."""

    def __init__(self, T):
        self.T = T; self.C = T
        self.TRI = np.array(list(permutations(range(T), 3))); self.NT = len(self.TRI)
        FREE = np.array([[p for p in range(T) if p not in t] for t in self.TRI])
        self.P4 = FREE[:, -1]; self.LOWS = FREE[:, :-1]
        self.TABLES = []
        for name, lo, hi in REGIONS:
            x, w = np.polynomial.legendre.leggauss(G_NODES)
            nodes, wts = lo + (hi - lo) * (x + 1) / 2, (hi - lo) / 2 * w
            self.TABLES.append(dict(name=name, nodes=nodes, wts=wts, blind12=1.0 - ndtr(nodes), dens=T * (T - 1) * _pdf(nodes)))
        self._lines = {}

    # --- model-free laws (T-dependent exponents) ---
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
        Jm = np.diag(a) + np.diag(np.sqrt(b[1:]), 1) + np.diag(np.sqrt(b[1:]), -1)
        xx, U = np.linalg.eigh(Jm)
        return xx, U[0] ** 2

    def cond_g23(self, v2, n=20001):
        v = np.linspace(v2 - 12, v2, n); f = (ndtr(v) / ndtr(v2)) ** (self.T - 2)
        return float(((f[1:] + f[:-1]) / 2 * np.diff(v)).sum())

    def inputs(self, v1, v2, v3, v4, low, d1, d3, d4, dlow):
        n = self.NT; r = np.arange(n); TRI = self.TRI
        x = np.zeros((n, self.T)); dx = np.zeros((n, self.T))
        for cols, val, dv in ((TRI[:, 0], v1, d1), (TRI[:, 1], v2, 0.0), (TRI[:, 2], v3, d3), (self.P4, v4, d4)):
            x[r, cols] = val; dx[r, cols] = dv
        for j in range(self.T - 4):
            x[r, self.LOWS[:, j]] = low; dx[r, self.LOWS[:, j]] = dlow
        return x, dx

    def mass(self, pieces, nrow, node_of_row, nodes, t0):
        e = self.T - 2; out = np.zeros((nrow, len(nodes))); f = 0.0
        for rows, lo, hi in pieces:
            a = np.maximum(lo, t0); m = hi > a
            if not m.any():
                continue
            r, a, b = rows[m], a[m], hi[m]
            j = node_of_row[r]; nd = nodes[j]
            val = ndtr(nd - a) ** e - ndtr(nd - b) ** e
            np.add.at(out, (r, j), val); f += len(r) * (2 * (F_PHI + 3) + 2)
        return out, f

    # --- g12 rules ---
    @staticmethod
    def gauss_g12(v2, a, K, n=40001):
        """t340c.gauss_g12, unchanged (law of g12 | v2 does not depend on T)."""
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

    @classmethod
    def graded_g12(cls, v2):
        """t366.graded_g12, unchanged."""
        norm_ = ndtr(-v2); xs, ws = [], []
        for a, b in PANELS:
            g = a + (b - a) * (_gx + 1) / 2; xs += list(g); ws += list((b - a) / 2 * _gw * _pdf(v2 + g) / norm_)
        gn, gw = cls.gauss_g12(v2, A_TAIL, K_TAIL); tail = ndtr(-(v2 + A_TAIL)) / norm_
        xs += list(gn); ws += list(gw * tail)
        return np.array(xs), np.array(ws)

    def lines(self, rule):
        """Per region: line23 inputs for every (v2 node, g12 node, triple) and row weights (t340c.node_lines with a = 0,
        or t366.node_lines_graded)."""
        if rule in self._lines:
            return self._lines[rule]
        out = []
        for tab in self.TABLES:
            xs, dxs, om = [], [], []
            for v2 in tab["nodes"]:
                v3s = v2 - self.cond_g23(v2); v4s = float(self.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
                m4 = float(mu(np.array([v4s]))[0]); dm4 = float(dmu(v4s)[0])
                gn, gw = self.gauss_g12(v2, 0.0, 3) if rule == "k3" else self.graded_g12(v2)
                for g12, w in zip(gn, gw):
                    x, dx = self.inputs(v2 + g12, v2, v2, v2 - delta, m4 + dm4 * (v2 - delta - v4s), 0.0, -1.0, -1.0, -dm4)
                    xs.append(x); dxs.append(dx); om.append(np.full(self.NT, w))
            out.append((np.concatenate(xs), np.concatenate(dxs), np.concatenate(om), len(gn)))
        self._lines[rule] = out
        return out

    def est_e1(self, model, t0s, rule, tmax=TBIG, ch=None, stats=None):
        """t366.est_e1 (rule 'graded') / t354.est_seg's E1 branch at K = 3 (rule 'k3'), rows walked in chunks."""
        model = tuple(np.asarray(w, np.float64) for w in model); H = model[1].shape[0]; T = self.T
        assert model[2].shape == (T, H)
        ch = ch or int(os.environ.get("T424_CH", 0)) or int(min(40_000, max(2_000, 4e8 / (T * H * 20))))   # row chunk (memory only)
        res = {t0: 0.0 for t0 in t0s}; flops = 0.0; t_start = min(t0s); trunc = 0
        for ri, tab in enumerate(self.TABLES):
            nodes, wts = tab["nodes"], tab["wts"]; x, dx, om, nn = self.lines(rule)[ri]; nrow = len(x)
            k = np.tile(self.TRI[:, 1], nrow // self.NT); k3 = np.tile(self.TRI[:, 2], nrow // self.NT)
            node_of_row = np.repeat(np.arange(len(nodes)), nn * self.NT)
            wgt = tab["dens"] * tab["blind12"]; acc = {t0: np.zeros(len(nodes)) for t0 in t0s}
            for a in range(0, nrow, ch):
                sl = slice(a, min(a + ch, nrow)); n = sl.stop - sl.start
                pieces, f, tr = walk_seg(model, x[sl], dx[sl], k[sl], k3[sl], t_start, tmax); flops += f; trunc += tr
                for t0 in t0s:
                    W, f = self.mass(pieces, n, node_of_row[sl], nodes, t0); flops += f
                    acc[t0] += (W * om[sl][:, None]).sum(0)
            for t0 in t0s:
                res[t0] += float(acc[t0] * wgt @ wts) / self.NT; flops += nrow * len(nodes) * 3
        if stats is not None:
            stats["truncated_rows"] = trunc
        return res, flops


_LAWS = {}


def law(T):
    if T not in _LAWS:
        _LAWS[T] = Law(T)
    return _LAWS[T]


def est_k3(model, t0s, **kw):
    return law(np.asarray(model[2]).shape[0]).est_e1(model, tuple(t0s), "k3", **kw)


def est_graded(model, t0s, **kw):
    return law(np.asarray(model[2]).shape[0]).est_e1(model, tuple(t0s), "graded", **kw)
