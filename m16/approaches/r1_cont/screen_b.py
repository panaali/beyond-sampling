"""T330 tier B: row-specific "safely correct to tmax" screen (README, T330 pre-registration; tier A fired on 0 rows:
its gate-agnostic slope bound is ~1e8).
At t (walk start, or a gate flip t* after the values update), with Delta = tmax - t, the row's exact z_s(t) and current
tangents, propagate an interval box for dz_s / dh_s valid for EVERY point of [t, tmax] and every flip pattern there:
  dz_s in Wh [dh_{s-1}] + d_s Wi (centre/radius; exact while the radius is 0);
  reachable z_s,i over the window lies in [z + Delta min(0, lo dz), z + Delta max(0, hi dz)]; if that cannot reach the
  other side of 0 (an off gate: stays <= 0; an on gate: stays >= 0), the gate is FIXED and dh_s,i = gate * dz_s,i,
  else dh_s,i in hull(0, dz_s,i).
Sound by induction over s: each box holds for all t in the window because the boxes of earlier steps do.
Then d(L_k - L_j)/dt in (Wo_k - Wo_j)[dh_T] and the row is safe iff for all j != k:
  (L_k - L_j)(t) + Delta * min(0, lo d(L_k - L_j)) > 0.
A cheap prefilter decides when to TRY (frozen-gate margin at the exact tangent > 0); it does not affect soundness.
The gate test at the flipped neuron itself (z = 0) uses its new gate: it stays fixed if dz keeps it on its new side.
Charges as executed: prefilter 2C + 4C + 2; a full check: per step with radius 0, 4H; with radius > 0, 4H^2 + 10H;
readout box 2CH + 2CH + 4C."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
from cont_e import H, T, C, NT, TMAX  # noqa: E402

F_PRE = 2 * C + 4 * C + 2
F_STEP0, F_STEPR = 4 * H, 4 * H * H + 10 * H
F_READBOX = 2 * C * H + 2 * C * H + 4 * C


def check_row(model, aWh, z, DZ, g, d_row, Lt, k, delta, aWoD, s_exact):
    """z, g: (T, H) at t; DZ[s] is exact for s <= s_exact (gates g). Returns (safe, FLOPs)."""
    Wi, Wh, Wo = model
    c = np.zeros(H); r = np.zeros(H); f = 0.0; exact = True
    for s in range(T):
        if exact and s <= s_exact:
            cz = DZ[s]; rz = np.zeros(H); f += F_STEP0
        elif not r.any():
            cz = Wh @ c + d_row[s] * Wi; rz = np.zeros(H); f += 2 * H * H + 2 * H + F_STEP0
        else:
            cz = Wh @ c + d_row[s] * Wi; rz = aWh @ r; f += F_STEPR
        lo, hi = cz - rz, cz + rz
        zlo = z[s] + delta * np.minimum(0.0, lo); zhi = z[s] + delta * np.maximum(0.0, hi)
        on = g[s]
        fixed = np.where(on, zlo >= 0.0, zhi <= 0.0)
        exact = exact and bool(fixed.all())
        hlo = np.where(fixed, np.where(on, lo, 0.0), np.minimum(lo, 0.0))
        hhi = np.where(fixed, np.where(on, hi, 0.0), np.maximum(hi, 0.0))
        c = (hlo + hhi) / 2; r = (hhi - hlo) / 2
    dm_c = (Wo[k] - Wo) @ c; dm_r = aWoD[k] @ r
    m = Lt[k] - Lt + delta * np.minimum(0.0, dm_c - dm_r); m[k] = np.inf; f += F_READBOX
    return bool(m.min() > 0), f


def walk_screen(model, x0, d, k, tmax, use, max_cells=400):
    Wi, Wh, Wo = model; n = len(x0); pieces = []
    aWh = np.abs(Wh); aWoD = np.abs(Wo[:, None, :] - Wo[None, :, :])
    Z, DZ, G, R, J, nodes = E.first_cell(model, x0, d)
    flops = nodes * E.F_STEP + n * E.F_CELL + (H * H + C * C * H * 2 if use else 0); st = dict(tried=0, stopped=0)
    tref = np.full(n, E.EPS)
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0

    def screen(rows):
        nonlocal flops
        ok = np.zeros(len(rows), bool)
        for i, row in enumerate(rows):
            delta = tmax - t[row]; Lt = L[row] + (t[row] - tref[row]) * dL[row]
            pre = Lt[k[row]] - Lt + delta * np.minimum(0.0, dL[row, k[row]] - dL[row]); pre[k[row]] = np.inf
            flops += F_PRE
            if pre.min() <= 0:
                continue
            st["tried"] += 1
            zt = Z[row] + (t[row] - tref[row]) * DZ[row]
            s_, f = check_row(model, aWh, zt, DZ[row], G[row], d[row], Lt, k[row], delta, aWoD, T)
            flops += f + 2 * T * H; ok[i] = s_
        return ok
    if use:
        s0 = screen(act)
        if s0.any():
            sr = act[s0]; pieces.append((sr, t[sr].copy(), np.full(len(sr), tmax))); act = act[~s0]; st["stopped"] += len(sr)
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = E.correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)
        pieces.append((act.copy(), lo, hi))
        t[act] = nxt
        keep = nxt < tmax; go = act[keep]; ss = sstar[keep]
        if not len(go):
            break
        tn = t[go]
        Z[go] += (tn - tref[go])[:, None, None] * DZ[go]; L[go] += (tn - tref[go])[:, None] * dL[go]; tref[go] = tn
        jj = J[go, ss]; G[go, ss, jj] = ~G[go, ss, jj]
        if use:
            # after the toggle: the flipped neuron's new tangent is needed for the box; recompute the row's tangents
            # lazily inside the check from step s* (charged as the regular restart would)
            sf = np.zeros(len(go), bool)
            for i in range(len(go)):
                row, s0_ = go[i], ss[i]
                delta = tmax - tn[i]
                pre = L[row, k[row]] - L[row] + delta * np.minimum(0.0, dL[row, k[row]] - dL[row]); pre[k[row]] = np.inf
                flops += F_PRE
                if pre.min() <= 0:
                    continue
                st["tried"] += 1
                # box from the flip step: steps < s* keep their exact tangents; the check propagates the rest
                zt = Z[row].copy(); zt[s0_, jj[i]] = 0.0            # the flipped neuron is exactly at its root
                s_, f = check_row(model, aWh, zt, DZ[row], G[row], d[row], L[row], k[row], delta, aWoD, s0_)
                flops += f; sf[i] = s_
            if sf.any():
                sr = go[sf]; pieces.append((sr, t[sr].copy(), np.full(len(sr), tmax))); st["stopped"] += len(sr)
                go = go[~sf]; ss = ss[~sf]; jj = jj[~sf]; tn = tn[~sf]
            if not len(go):
                break
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
    return pieces, flops, st


def est(model, budget=None, lines=("nb-23", "nb-12"), parts=False):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = E.TRI[:, 1]; flops = 0.0; tot = np.zeros(NT); info = {}; prof = {}
    for tb in E.TABLES:
        nodes = tb["nodes"]; I = {}
        for kind in ("23", "12"):
            if f"line{kind}" not in tb:
                continue
            x, dx = tb[f"line{kind}"]; line = f"{tb['name']}-{kind}"
            p, f, st = walk_screen(model, x, dx, k, TMAX, line in lines); flops += f
            R, A, B, ops = E.merge(p, NT, TMAX); flops += ops
            I[kind], f = E.profile(R, A, B, NT, nodes, kind, TMAX, tb["phi0"]); flops += f
            info[line] = st; prof[line] = I[kind]
        I12 = I["12"] if "12" in I else np.repeat(tb["blind12"][None], NT, 0)
        tot += (I12 * I["23"] * tb["dens"][None]) @ tb["wts"]
        flops += NT * len(nodes) * 4
    flops += NT
    if parts:
        return float(tot.mean()), float(flops), info, prof
    return float(tot.mean()), float(flops)
