"""T330: deductive "safely correct to tmax" screen on T317e's walks (README, T330 pre-registration).
Tier A: per model and line, a row-independent box on the tangent over all gate patterns, b_s = |Wh| b_{s-1} + delta_s|Wi|
(delta_s = max_rows |d_s|), gives |d(L_k - L_j)/dt| <= beta_kj = |Wo_k - Wo_j| . b_T for every t (the logits are
continuous and piecewise linear, so this holds across flips). A row is stopped at t (the walk start, or a gate flip t*)
iff min_{j != k} [L_k(t) - L_j(t) - (tmax - t) beta_kj] > 0: its whole remaining segment is then correct whatever
flips happen, and it gets one correct piece [t, tmax]. At a flip, the check needs only L(t*) (affine, 2C); a stopped row
skips the restart (values, tangents, roots, dL, readout). est(model, lines=...) -> (estimate, FLOPs as executed);
lines=() reproduces cont_e.est exactly."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
from cont_e import H, T, C, NT, TMAX  # noqa: E402

F_CHECK = 2 * C + 4 * C + 2             # L(t) affine (2C) + margins, Delta*beta, subtract, min (4C) + compare (2)


def tier_a(model, d):
    """beta (C, C) for this line's directions d (NT, T); and its charged FLOPs."""
    Wi, Wh, Wo = model
    aWh, aWi = np.abs(Wh), np.abs(Wi); b = np.zeros(H)
    delta = np.abs(d).max(0)
    for s in range(T):
        b = aWh @ b + delta[s] * aWi
    beta = np.abs(Wo[:, None, :] - Wo[None, :, :]) @ b
    f = H * H + H + T * (2 * H * H + 2 * H) + 2 * C * C * H + 2 * C * C * H
    return beta, float(f)


def walk_screen(model, x0, d, k, tmax, beta, max_cells=400):
    """cont_e.walk_cont (with walk_counted's charges) plus the tier-A screen; beta=None -> no screen (== cont_e)."""
    Wi, Wh, Wo = model; n = len(x0); pieces = []; rr = np.arange(n)
    Z, DZ, G, R, J, nodes = E.first_cell(model, x0, d)
    flops = nodes * E.F_STEP + n * E.F_CELL; stopped = 0
    tref = np.full(n, E.EPS)
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0
    bk = None if beta is None else beta[k]                   # (n, C): beta_{k_row, j}

    def safe_rows(rows):
        Lt = L[rows] + (t[rows] - tref[rows])[:, None] * dL[rows]
        m = Lt[np.arange(len(rows)), k[rows]][:, None] - Lt - (tmax - t[rows])[:, None] * bk[rows]
        m[np.arange(len(rows)), k[rows]] = np.inf
        return m.min(1) > 0
    if beta is not None:                                     # check at the walk start
        s0 = safe_rows(act); flops += n * F_CHECK
        if s0.any():
            sr = act[s0]; pieces.append((sr, t[sr].copy(), np.full(len(sr), tmax))); act = act[~s0]; stopped += len(sr)
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = E.correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)
        pieces.append((act.copy(), lo, hi))
        t[act] = nxt
        keep = nxt < tmax; go = act[keep]; ss = sstar[keep]
        if beta is not None and len(go):
            sf = safe_rows(go); flops += len(go) * F_CHECK
            if sf.any():
                sr = go[sf]; pieces.append((sr, t[sr].copy(), np.full(len(sr), tmax))); stopped += len(sr)
                go = go[~sf]; ss = ss[~sf]
        if not len(go):
            break
        tn = t[go]
        Z[go] += (tn - tref[go])[:, None, None] * DZ[go]; L[go] += (tn - tref[go])[:, None] * dL[go]; tref[go] = tn
        jj = J[go, ss]; G[go, ss, jj] = ~G[go, ss, jj]
        for idx in range(len(go)):
            row, s0_ = go[idx], ss[idx]
            dh = np.where(G[row, s0_], DZ[row, s0_], 0.0)
            for s in range(s0_ + 1, T):
                DZ[row, s] = Wh @ dh + d[row, s] * Wi
                dh = np.where(G[row, s], DZ[row, s], 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tn[idx] - Z[row, s0_:] / DZ[row, s0_:]
            r = np.where((r > tn[idx] + 1e-12 * (1 + abs(tn[idx]))) & np.isfinite(r), r, np.inf)
            r[0, jj[idx]] = np.inf
            R[row, s0_:] = r.min(1); J[row, s0_:] = r.argmin(1)
            dL[row] = Wo @ dh
            flops += 2 * T * H + 1 + H + (T - 1 - s0_) * E.F_TAN + (T - s0_) * E.F_ROOT + E.F_LATER_READ
        act = go
    return pieces, flops, stopped


def est(model, budget=None, lines=("nb-23", "nb-12"), parts=False):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = E.TRI[:, 1]; flops = 0.0; tot = np.zeros(NT); info = {}; prof = {}
    for tb in E.TABLES:
        nodes = tb["nodes"]; I = {}
        for kind in ("23", "12"):
            if f"line{kind}" not in tb:
                continue
            x, dx = tb[f"line{kind}"]; line = f"{tb['name']}-{kind}"
            beta = None
            if line in lines:
                beta, f = tier_a(model, dx); flops += f
            p, f, stp = walk_screen(model, x, dx, k, TMAX, beta); flops += f
            R, A, B, ops = E.merge(p, NT, TMAX); flops += ops
            I[kind], f = E.profile(R, A, B, NT, nodes, kind, TMAX, tb["phi0"]); flops += f
            info[line] = stp; prof[line] = I[kind]
        I12 = I["12"] if "12" in I else np.repeat(tb["blind12"][None], NT, 0)
        tot += (I12 * I["23"] * tb["dens"][None]) @ tb["wts"]
        flops += NT * len(nodes) * 4
    flops += NT
    if parts:
        return float(tot.mean()), float(flops), info, prof
    return float(tot.mean()), float(flops)
