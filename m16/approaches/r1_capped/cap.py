"""T315 estimator CAP-c (README): R1W V1 closure/points/masses, all four lines walked, the walk capped at c gate cells
with the last cell's exact affine margin extended to tmax (declared), first cell evaluated as a shared prefix tree
(B2, executed). est(model, budget=None, c=2) -> (acc, charged FLOPs)."""
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
H, T = 16, 10
TMAX = r1w2.TMAX
F_STEP, F_CELL = v1b2.F_STEP, v1b2.F_CELL
EPS = 1e-10


def first_cell_tree(model, x0, d):
    """First cell at t = 0 (tc = EPS for every row) as a prefix tree: each distinct (input, tangent) prefix is
    evaluated once. Returns per row L, dL, nxt (first gate flip), rstep, and the step-evaluation count."""
    Wi, Wh, Wo = model; n = len(x0); tc = EPS
    inv_prev = np.zeros(n, int); h_prev = np.zeros((1, H)); dh_prev = np.zeros((1, H))
    best_prev = np.full(1, np.inf); bstep_prev = np.full(1, T); nsteps = 0
    for s in range(T):
        key = np.round(np.column_stack([inv_prev, x0[:, s], d[:, s]]), 12)
        _, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel(); par = inv_prev[rep]; nsteps += len(rep)
        z = h_prev[par] @ Wh.T + (x0[rep, s] + tc * d[rep, s])[:, None] * Wi[None]
        dz = dh_prev[par] @ Wh.T + d[rep, s][:, None] * Wi[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = tc - z / dz
        r = np.where((r > tc) & np.isfinite(r), r, np.inf).min(1)
        bp = best_prev[par]; upd = r < bp
        best = np.where(upd, r, bp); bstep = np.where(upd, s, bstep_prev[par])
        g = z > 0
        h_prev, dh_prev = np.where(g, z, 0.0), np.where(g, dz, 0.0)
        best_prev, bstep_prev, inv_prev = best, bstep, inv
    L = h_prev[inv_prev] @ Wo.T; dL = dh_prev[inv_prev] @ Wo.T
    return L, dL, best_prev[inv_prev], bstep_prev[inv_prev], nsteps


def walk_capped(model, x0, d, k, tmax, c):
    """Exact walk through the first c cells (first one as a prefix tree), then the last cell's affine margin extended
    to tmax. Returns correct pieces and charged FLOPs."""
    Wi, Wh, Wo = model; n = len(x0); pieces = []
    L, dL, nxt, rstep, nsteps = first_cell_tree(model, x0, d)
    nxt = np.minimum(nxt, tmax); tc = np.full(n, EPS); t = np.zeros(n)
    flops = nsteps * F_STEP + n * F_CELL
    lo, hi = W.correct_interval(L, dL, tc, k, t, nxt); pieces.append((np.arange(n), lo, hi))
    t = nxt.copy(); restart = rstep.copy(); act = np.flatnonzero(t < tmax); cells = 1
    Llast, dLlast, tclast = L.copy(), dL.copy(), tc.copy()
    while len(act) and cells < c:
        cells += 1
        X0, D = x0[act], d[act]; tca = t[act] + EPS * (1 + np.abs(t[act]))
        h = np.zeros((len(act), H)); dh = np.zeros_like(h); nx = np.full(len(act), tmax); rs = np.full(len(act), T)
        for s in range(T):
            z = h @ Wh.T + (X0[:, s] + tca * D[:, s])[:, None] * Wi[None]
            dz = dh @ Wh.T + D[:, s:s + 1] * Wi[None]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tca[:, None] - z / dz
            r = np.where((r > tca[:, None]) & np.isfinite(r), r, np.inf).min(1)
            upd = r < nx; nx = np.where(upd, r, nx); rs = np.where(upd, s, rs)
            g = z > 0; h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        La = h @ Wo.T; dLa = dh @ Wo.T
        lo, hi = W.correct_interval(La, dLa, tca, k[act], t[act], nx); pieces.append((act.copy(), lo, hi))
        flops += (T - restart[act]).sum() * F_STEP + len(act) * F_CELL
        Llast[act], dLlast[act], tclast[act] = La, dLa, tca
        t[act] = nx; restart[act] = rs; act = act[nx < tmax]
    if len(act):                                       # declared: no further gate change up to tmax
        lo, hi = W.correct_interval(Llast[act], dLlast[act], tclast[act], k[act], t[act], np.full(len(act), tmax))
        pieces.append((act.copy(), lo, hi)); flops += len(act) * F_CELL
    return pieces, flops


def est(model, budget=None, c=2, parts=False):
    model = tuple(np.asarray(w, np.float64) for w in model)
    TRI = r1w.TRI; NT = len(TRI); k = TRI[:, 1]; flops = 0.0; tot = np.zeros(NT); split = {}
    for name, lo, hi, (v2s, g12s, g23s) in r1w2.REG[False]:
        nodes, wts = W._gl(lo, hi)
        v3s = v2s - g23s; v4s = float(r1w.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
        m4 = float(r0.mu(np.array([v4s]))[0]); dm4 = float(r1w.dmu(v4s)[0])
        tail = (np.arange(NT), np.full(NT, TMAX), np.full(NT, W.INF))
        x, dx = r1w.inputs(v2s + g12s, v2s, v2s, v2s - delta, m4 + dm4 * (v2s - delta - v4s), 0.0, -1.0, -1.0, -dm4)
        p23, f = walk_capped(model, x, dx, k, TMAX, c); flops += f
        I23, _ = W._profile(p23 + [tail], NT, nodes, "23"); flops += v1b2.phi_flops(p23 + [tail], nodes, "23", TMAX)
        x, dx = r1w.inputs(v2s, v2s, v3s, v4s, m4, 1.0, 0.0, 0.0, 0.0)
        p12, f = walk_capped(model, x, dx, k, TMAX, c); flops += f
        I12, _ = W._profile(p12 + [tail], NT, nodes, "12"); flops += v1b2.phi_flops(p12 + [tail], nodes, "12", TMAX)
        part = (I12 * I23 * (90 * norm.pdf(nodes))[None]) @ wts; tot += part; split[name] = float(part.mean())
        flops += NT * len(nodes) * 4
    flops += NT
    acc = float(tot.mean())
    return (acc, flops, split) if parts else (acc, flops)
