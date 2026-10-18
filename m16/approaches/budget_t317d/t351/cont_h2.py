"""T353: arm A3 of T351 (cont_h.est_h(spread=-1, tilt=True)) with the crossing slopes taken from the walk's own
crossing-cell state instead of a fresh value pass. walk_c1_cap is cont_g.walk_c1 verbatim (same arithmetic and charges)
plus a capture of the cell state (gates, dL, L at theta*) where each row's first non-empty correct piece starts.
Charges for the reuse (per wall row): L(theta*) affine 2C, margins + argmin 3C, alpha 1, cross tangent T F_TAN + 2CH,
beta + tilt 5; g-rule as A3 (cont_h.F_D_ROW). The capture is a copy (no arithmetic; uncharged, declared)."""
import os
import sys

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cont_g as CG  # noqa: E402
import cont_h as CH  # noqa: E402

H, T, C, TMAX, NT, TRI = CG.H, CG.T, CG.C, CG.TMAX, CG.NT, CG.TRI
F_REUSE = 2 * C + 3 * C + 1 + T * CG.F_TAN + 2 * C * H + 5            # = 5,976 per wall row


def walk_c1_cap(model, x0, d, k, tmax, max_cells=400):
    """cont_g.walk_c1 + capture (marked CAP)."""
    Wi, Wh, Wo = model; n = len(x0); pieces = []
    Z, DZ, G, R, J, nodes = CG.first_cell(model, x0, d)
    flops = nodes * CG.F_STEP + n * CG.F_CELL
    tref = np.full(n, CG.EPS)
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0
    cap = dict(done=np.zeros(n, bool), th=np.full(n, tmax), G=np.zeros((n, T, H), bool), dL=np.zeros((n, C)), L=np.zeros((n, C)))  # CAP

    def capture(rows, lo):                                                                      # CAP
        new = rows[~cap["done"][rows]]; lo_ = lo[~cap["done"][rows]]
        if len(new):
            cap["done"][new] = True; cap["th"][new] = lo_; cap["G"][new] = G[new]; cap["dL"][new] = dL[new]
            cap["L"][new] = L[new] + (lo_ - tref[new])[:, None] * dL[new]                          # value copy at theta*

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
        capture(sr, t[sr].copy())                                                               # CAP
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = CG.correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)
        pieces.append((act.copy(), lo, hi))
        ne = hi > lo; capture(act[ne], lo[ne])                                                  # CAP
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
            flops += 2 * T * H + 1 + H + (T - 1 - s0_) * CG.F_TAN + 2 * C * H
            if safe(row):
                alive[idx] = False; continue
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tn[idx] - Z[row, s0_:] / DZ[row, s0_:]
            r = np.where((r > tn[idx] + 1e-12 * (1 + abs(tn[idx]))) & np.isfinite(r), r, np.inf)
            r[0, jj[idx]] = np.inf
            R[row, s0_:] = r.min(1); J[row, s0_:] = r.argmin(1)
            flops += (T - s0_) * CG.F_ROOT + CG.F_LATER_READ - 2 * C * H
        if (~alive).any():
            sr = go[~alive]; pieces.append((sr, t[sr].copy(), np.full(len(sr), tmax)))
            capture(sr, t[sr].copy())                                                           # CAP
        go = go[alive]
        act = go
    return pieces, flops, cap


def _slopes_reuse(model, cap, rows, cross, k):
    Wi, Wh, Wo = model; Gc = cap["G"][rows]; n = len(rows); db = np.zeros((n, H))
    for s in range(T):
        db = np.where(Gc[:, s], db @ Wh.T + cross[:, s:s + 1] * Wi[None], 0.0)
    Lb = db @ Wo.T; L = cap["L"][rows]; dL = cap["dL"][rows]
    rr = np.arange(n); M = L[rr, k][:, None] - L; M[rr, k] = np.inf; riv = M.argmin(1)
    return dL[rr, k] - dL[rr, riv], Lb[rr, k] - Lb[rr, riv]


def est(model, budget=None, spread=-1.0, parts=False):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = TRI[:, 1]; flops = 0.0; tot = np.zeros(NT); keep = {}
    tabs = CH.TABLES[spread] if spread in CH.TABLES else CG._tables(spread)
    for t in tabs:
        nodes = t["nodes"]; I = {}
        for kind in ("23", "12"):
            if f"line{kind}" not in t:
                continue
            x, dx = t[f"line{kind}"]
            if t["name"] == "nb":
                p, f, cap = walk_c1_cap(model, x, dx, k, TMAX)
            else:
                p, f = CG.walk_c1(model, x, dx, k, TMAX)
            flops += f
            R, A, B, ops = CG.merge(p, NT, TMAX); flops += ops
            I[kind], f = CG.profile(R, A, B, NT, nodes, kind, TMAX, t["phi0"]); flops += f
            if t["name"] == "nb":
                keep[kind] = (R, A, cap)
        I12 = I["12"] if "12" in I else np.repeat(t["blind12"][None], NT, 0)
        tot += (I12 * I["23"] * t["dens"][None]) @ t["wts"]
        flops += NT * len(nodes) * 4
    flops += NT
    acc = float(tot.mean())
    nb = tabs[0]; (x23, d23), (x12, d12) = nb["line23"], nb["line12"]; v2s, g12s, g23s = nb["point"]
    V = CH.V; Dt = np.zeros((NT, len(V))); info = {}
    for kind, cross, gs, W, Psi in (
            ("23", d12, g12s, CH.W23, lambda c: CH.P8V[None, :, None] - ndtr(V[None, :, None] - c) ** 8),
            ("12", d23, g23s, CH.W12, lambda c: ndtr(V[None, :, None] + c) - CH.PV[None, :, None])):
        R, A, cap = keep[kind]; th = CH._first_start(R, A, NT); flops += 2 * len(R)
        w = np.flatnonzero((th > 0) & (th < TMAX))
        if not len(w):
            continue
        a, b = _slopes_reuse(model, cap, w, cross[w], k[w]); flops += len(w) * F_REUSE
        s = np.where(np.abs(a) > 1e-12, -b / np.where(np.abs(a) > 1e-12, a, 1.0), 0.0)
        thg = np.clip(th[w, None, None] + s[:, None, None] * (CH.GQ[None, None] - gs), 0, TMAX)
        Dt[w] += ((W[None] * (Psi(thg) - Psi(th[w, None, None]))) @ CH.WQ)
        flops += len(w) * CH.F_D_ROW; info[f"walls{kind}"] = len(w)
        info[f"th_mismatch{kind}"] = int((np.abs(cap["th"][w] - th[w]) > 0).sum())
    corr = (Dt * nb["dens"][None]) @ nb["wts"]; flops += NT * len(V) * 3
    D = float(corr.mean()); acc -= D
    return (acc, flops, dict(D=D, **info)) if parts else (acc, flops)
