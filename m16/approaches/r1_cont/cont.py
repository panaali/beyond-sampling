"""T317: R1W V1 with (a) continuity restarts and (b) an executed B2 prefix tree; cost only (estimates = V1).

(a) A gate flip is continuous in the walk parameter. At the flip t* = (step s*, neuron j*), every pre-activation value
z_t(t*) follows from the current cell's affine form, exactly one gate toggles (s*, j*), and only the tangents of steps
> s* change. Per later cell, as executed:
  values at t*: 2 T H | gate toggle: 1 | tangents t > s*: (2H^2 + 2H + H) each, plus the mask at s*: H |
  roots for steps >= s* (steps < s* are unchanged affine pieces whose cached minima remain valid): 5H each |
  readout L(t*) affine 2C, dL = Wo dh_T 2CH, correct-set solve 146, bookkeeping 4.
(b) First cell: a prefix tree over (input, tangent) prefixes, executed (each distinct prefix evaluated once): F_STEP
  per node, then per row readout F_CELL.
est(model, budget=None, G=48) -> (acc, FLOPs). With G=48 the estimate equals frozen V1 (r1w2.est) up to rounding."""
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("r1_v1b2", "r1_walk", "r0_walk", "r0_integral"):
    sys.path.insert(0, os.path.join(HERE, "..", p))
import r0  # noqa: E402
import walk as W  # noqa: E402
import r1w  # noqa: E402
import r1w2  # noqa: E402
import v1b2  # noqa: E402

DEDUCTIVE = True
H, T, C = 16, 10, 10
TMAX = r1w2.TMAX
F_STEP, F_CELL = v1b2.F_STEP, v1b2.F_CELL
F_TAN = 2 * H * H + 2 * H + H            # one tangent step (matvec + input + mask) = 560
F_ROOT = 5 * H                           # roots of one step: sub/div (2H), >t and finite (2H), min (H)
F_LATER_READ = 2 * C + 2 * C * H + 146 + 4
EPS = 1e-10


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
    flops = nodes * F_STEP + n * F_CELL
    tref = np.full(n, EPS)                                  # parameter at which Z holds the values
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = W.correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)
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
            flops += 2 * T * H + 1 + H + (T - 1 - s0) * F_TAN + (T - s0) * F_ROOT + F_LATER_READ
        act = go
    return pieces, flops


def _gl(lo, hi, G):
    x, w = np.polynomial.legendre.leggauss(G)
    return lo + (hi - lo) * (x + 1) / 2, (hi - lo) / 2 * w


def est(model, budget=None, G=48, return_cost=True):
    model = tuple(np.asarray(w, np.float64) for w in model)
    TRI = r1w.TRI; NT = len(TRI); k = TRI[:, 1]; flops = 0.0; tot = np.zeros(NT)
    full = [(np.arange(NT), np.zeros(NT), np.full(NT, W.INF))]
    for name, lo, hi, (v2s, g12s, g23s) in r1w2.REG[False]:
        nodes, wts = _gl(lo, hi, G)
        v3s = v2s - g23s; v4s = float(r1w.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
        m4 = float(r0.mu(np.array([v4s]))[0]); dm4 = float(r1w.dmu(v4s)[0])
        tail = (np.arange(NT), np.full(NT, TMAX), np.full(NT, W.INF))
        x, dx = r1w.inputs(v2s + g12s, v2s, v2s, v2s - delta, m4 + dm4 * (v2s - delta - v4s), 0.0, -1.0, -1.0, -dm4)
        p23, f = walk_cont(model, x, dx, k, TMAX); flops += f
        I23, _ = W._profile(p23 + [tail], NT, nodes, "23"); flops += v1b2.phi_flops(p23 + [tail], nodes, "23", TMAX)
        if name.startswith("nb"):
            x, dx = r1w.inputs(v2s, v2s, v3s, v4s, m4, 1.0, 0.0, 0.0, 0.0)
            p12, f = walk_cont(model, x, dx, k, TMAX); flops += f
            I12, _ = W._profile(p12 + [tail], NT, nodes, "12"); flops += v1b2.phi_flops(p12 + [tail], nodes, "12", TMAX)
        else:
            I12, _ = W._profile(full, NT, nodes, "12")            # blind: g12 assumed correct (V1), model-free
        tot += (I12 * I23 * (90 * norm.pdf(nodes))[None]) @ wts
        flops += NT * len(nodes) * 4
    flops += NT
    return float(tot.mean()), float(flops)
