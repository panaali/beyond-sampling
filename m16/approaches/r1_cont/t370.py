"""T370: replace the R1W closure by an exact-law cubature over (v4 position, v4 value | v3, 6 lows | v4), one walk per
node, on T366's E1 estimator (graded g12) and T340c's E2 estimator (K3 restricted g12) (README, T370 pre-registration).
Nodes: the first M = 2^k points of the unscrambled 8-D Sobol set, half-cell shifted (model-free, equal weights). Node values are linearised along each
line around the model-free anchor t_a = t_start + E[g23 - t_start | g23 > t_start, v2]. closure="R1W" rebuilds T366's
closure lines through the same loop (validation). Deductive: no random draws.
est_e1(model, t0s, M) / est_e2(model, t, M) -> (estimate, FLOPs as executed)."""
import os
import sys

import numpy as np
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t366 as X  # noqa: E402  (graded g12 rule; patches the fixed walk into T340's rare.walk3)

TC = X.TC; G = X.G; rare = X.rare; S = X.S; NT = X.NT; TRI = X.TRI
CHUNK = 50_000
import cont_e as CE  # noqa: E402  (TRI / P4 / LOWS are the same model-free tables)
FREE = np.sort(np.concatenate([CE.P4[:, None], CE.LOWS], 1), 1)                       # (720, 7) free positions
pdf = lambda v: np.exp(-v * v / 2) / np.sqrt(2 * np.pi)


def sobol_nodes(M):
    """First M = 2^k points of the unscrambled 8-D Sobol set, shifted by half a cell: (u + .5 / M) mod 1, so no
    coordinate is 0 or 1 and the balance property is kept. Deterministic, model-free."""
    from scipy.stats import qmc
    k = int(np.log2(M)); assert 2 ** k == M
    return (qmc.Sobol(d=8, scramble=False).random_base2(k) + 0.5 / M) % 1.0


def excess_g23(v2, ts, n=4001):
    """E[g23 - ts | g23 > ts, v2], with P(g23 > g | v2) = (Phi(v2 - g) / Phi(v2))^8."""
    g = np.linspace(ts, ts + 12, n); s = (ndtr(v2 - g) / ndtr(v2 - ts)) ** 8
    return float(((s[1:] + s[:-1]) / 2 * np.diff(g)).sum())


def node_rows(v2, g12s, g12w, t_start, U, closure=None):
    """Lines (x0, d) and row weights for one v2 node: rows ordered (g12 node, cubature node, triple)."""
    ta = t_start + excess_g23(v2, t_start); v3a = v2 - ta; xs, ds, om = [], [], []
    r = np.arange(NT)
    for g12, wg in zip(g12s, g12w):
        if closure == "R1W":                                        # T366's closure line (validation)
            v3s = v2 - rare._cond_g23(v2); v4s = float(G.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
            m4 = float(G.mu(np.array([v4s]))[0]); dm4 = float(G.dmu(v4s)[0])
            x, dx = G.inputs(v2 + g12, v2, v2, v2 - delta, m4 + dm4 * (v2 - delta - v4s), 0.0, -1.0, -1.0, -dm4)
            xs.append(x); ds.append(dx); om.append(np.full(NT, wg)); continue
        for u in U:
            a = u[1] ** (1 / 7); v4a = ndtri(ndtr(v3a) * a); dv4 = -(pdf(v3a) * a / pdf(v4a))          # dv4/dt
            lows_a = ndtri(ndtr(v4a) * u[2:]); dl = pdf(v4a) * u[2:] / pdf(lows_a) * dv4               # dlow/dt
            p = min(int(u[0] * 7), 6)
            xa = np.zeros((NT, 10)); d = np.zeros((NT, 10))
            xa[r, TRI[:, 0]] = v2 + g12; xa[r, TRI[:, 1]] = v2; xa[r, TRI[:, 2]] = v3a; d[r, TRI[:, 2]] = -1.0
            xa[r, FREE[:, p]] = v4a; d[r, FREE[:, p]] = dv4
            lowpos = np.delete(FREE, p, axis=1)
            for i in range(6):
                xa[r, lowpos[:, i]] = lows_a[i]; d[r, lowpos[:, i]] = dl[i]
            xs.append(xa - ta * d); ds.append(d); om.append(np.full(NT, wg / len(U)))
    return np.concatenate(xs), np.concatenate(ds), np.concatenate(om)


def _walk_chunks(model, x, dx, k, k3, t_start, tmax):
    pieces, pieces_c, flops = [], [], 0.0
    for c0 in range(0, len(x), CHUNK):
        sl = slice(c0, c0 + CHUNK)
        p, pc, f = S.walk_seg(model, x[sl], dx[sl], k[sl], k3[sl], t_start, tmax); flops += f
        pieces += [(rr + c0, lo, hi) for rr, lo, hi in p]; pieces_c += [(rr + c0, lo, hi) for rr, lo, hi in pc]
    return pieces, pieces_c, flops


def est_e1(model, t0s, M, closure=None, tmax=6.0):
    model = tuple(np.asarray(w, np.float64) for w in model); U = sobol_nodes(M) if closure is None else None
    res = {t0: 0.0 for t0 in t0s}; flops = 0.0; ts = min(t0s)
    for tab in G.TABLES:
        for j, v2 in enumerate(tab["nodes"]):
            gn, gw = X.graded_g12(v2); x, dx, om = node_rows(v2, gn, gw, ts, U, closure); n = len(x)
            k = np.tile(TRI[:, 1], n // NT); k3 = np.tile(TRI[:, 2], n // NT)
            pieces, _, f = _walk_chunks(model, x, dx, k, k3, ts, tmax); flops += f
            for t0 in t0s:
                W, f = rare.mass(pieces, n, np.zeros(n, int), np.array([v2]), t0); flops += f
                res[t0] += float((W[:, 0] * om).sum() * tab["dens"][j] * tab["blind12"][j] * tab["wts"][j]) / NT
                flops += n * 3
    return res, flops


def est_e2(model, t, M, closure=None, tmax=4.0):
    model = tuple(np.asarray(w, np.float64) for w in model); U = sobol_nodes(M) if closure is None else None
    res = 0.0; flops = 0.0
    for tab in G.TABLES:
        for j, v2 in enumerate(tab["nodes"]):
            gn, gw = TC.gauss_g12(v2, t, 3); x, dx, om = node_rows(v2, gn, gw, t, U, closure); n = len(x)
            k = np.tile(TRI[:, 1], n // NT); k3 = np.tile(TRI[:, 2], n // NT)
            _, pieces_c, f = _walk_chunks(model, x, dx, k, k3, t, tmax); flops += f
            Cm, f = rare.mass(pieces_c, n, np.zeros(n, int), np.array([v2]), t); flops += f
            tot = ndtr(v2 - t) ** 8 - ndtr(v2 - tmax) ** 8
            res += float(((tot - Cm[:, 0]) * om).sum() * tab["dens"][j] * ndtr(-(v2 + t)) * tab["wts"][j]) / NT
            flops += n * 4
    return res, flops
