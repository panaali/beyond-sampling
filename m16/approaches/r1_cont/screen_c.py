"""T330 tier B, v2 ("radius-only"): row-specific "safely correct to tmax" screen on T317e's walks.
At t (the walk start, or a gate flip t* AFTER the restart has recomputed the exact tangents dz*_s for the current gates
g and dL*), with Delta = tmax - t, a radius e_s bounds |dh_s(t') - dh*_s| for EVERY t' in [t, tmax] and every flip
pattern there:
  ez_s = |Wh| e_{s-1}  (bounds |dz_s(t') - dz*_s|);
  the reachable z_s,i over the window lies in z_s,i(t) + Delta [dz* - ez, dz* + ez] (taking min/max with 0);
  if it cannot reach the other side of 0 the gate is FIXED: e_s,i = ez_s,i; else FREE: e_s,i = |dz*_s,i| + ez_s,i
  (dh is 0 or dz there).
This is sound by induction over the steps. The row is safe iff for every j != k:
  m_kj(t) + Delta * min(0, dm*_kj - |Wo_k - Wo_j| . e_T) > 0, with dm* = dL*_k - dL*_j.
A safe row gets one correct piece [t, tmax] and skips its roots, its correct set and all later cells.
Prefilter: m_kj(t) + Delta min(0, dm*_kj) > 0, a NECESSARY condition (the box contains dm*), so it never loses a
stop. Charges as executed: prefilter 4C + 2; check: 4H per step, plus 2H^2 + 2H once e != 0; readout 2CH + 2C.
est(model, lines=()) reproduces cont_e.est exactly."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
import t331 as T331  # noqa: E402
from cont_e import H, T, C, NT, TMAX  # noqa: E402

F_PRE = 4 * C + 2
F_TEST, F_PROP = 4 * H, 2 * H * H + 2 * H
F_READ = 2 * C * H + 2 * C


def check(aWh, aWoDk, z, dz, g, m, dm, delta, s_root=None):
    """z, dz, g: (T, H) at t with exact tangents; m, dm: (C,) margins L_k - L_j and their exact slopes (inf at k).
    s_root = (s*, j*): the flipped neuron, exactly at its root. Returns (safe, FLOPs)."""
    e = np.zeros(H); f = 0.0; nz = False
    for s in range(T):
        ez = aWh @ e if nz else e; f += F_TEST + (F_PROP if nz else 0)
        zs = z[s]
        if s_root is not None and s == s_root[0]:
            zs = zs.copy(); zs[s_root[1]] = 0.0
        lo, hi = dz[s] - ez, dz[s] + ez
        zlo = zs + delta * np.minimum(0.0, lo); zhi = zs + delta * np.maximum(0.0, hi)
        fixed = np.where(g[s], zlo >= 0.0, zhi <= 0.0)
        e = np.where(fixed, ez, np.abs(dz[s]) + ez); nz = nz or bool(e.any())
    bound = m + delta * np.minimum(0.0, dm - aWoDk @ e); f += F_READ
    return bool(bound.min() > 0), f


def walk_screen(model, x0, d, k, tmax, use, max_cells=400, dmax=np.inf, maxtry=np.inf, mode="B", ext=False):
    """mode B: rigorous tier B; C1/C2: declared heuristic tier C (stop on the frozen-gate margin alone; C2 only after
    the row's first wrong piece has ended)."""
    Wi, Wh, Wo = model; n = len(x0); pieces = []
    Z, DZ, G, R, J, nodes = E.first_cell(model, x0, d)
    flops = nodes * E.F_STEP + n * E.F_CELL; st = dict(tried=0, stopped=0, prefiltered=0, restarts=0)
    if use:
        aWh = np.abs(Wh); aWoD = np.abs(Wo[:, None, :] - Wo[None, :, :]); flops += H * H + 2 * C * C * H
    tref = np.full(n, E.EPS)
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0; ntry = np.zeros(n); seen_wrong = np.zeros(n, bool)

    def try_row(row, s_root):
        nonlocal flops
        delta = tmax - t[row]; kk = k[row]
        if delta > dmax or ntry[row] >= maxtry:
            return False
        Lt = L[row] + (t[row] - tref[row]) * dL[row]; flops += 2 * C
        m = Lt[kk] - Lt; dm = dL[row, kk] - dL[row]; m[kk] = np.inf; dm[kk] = 0.0
        flops += F_PRE
        if (m + delta * np.minimum(0.0, dm)).min() <= 0:
            st["prefiltered"] += 1; return False
        if mode == "C1" or (mode == "C2" and seen_wrong[row]):
            st["tried"] += 1; return True
        if mode == "C2":
            return False
        st["tried"] += 1; ntry[row] += 1
        zt = Z[row] + (t[row] - tref[row]) * DZ[row]; flops += 2 * T * H
        s_, f = check(aWh, aWoD[kk], zt, DZ[row], G[row], m, dm, delta, s_root); flops += f
        return s_
    def stop_piece(sr):
        nonlocal flops
        if not ext:
            return (sr, t[sr].copy(), np.full(len(sr), tmax))
        lo_, hi_ = E.correct_interval(L[sr], dL[sr], tref[sr], k[sr], t[sr], np.full(len(sr), E.INF))
        flops += len(sr) * E.F_CI
        return (sr, lo_, hi_)
    if use:
        s0 = np.array([try_row(r_, None) for r_ in act])
        if s0.any():
            sr = act[s0]; pieces.append(stop_piece(sr)); act = act[~s0]; st["stopped"] += len(sr)
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = E.correct_interval(L[act], dL[act], tref[act], k[act], t[act],
                                    np.where(nxt >= tmax, E.INF, nxt) if ext else nxt)
        pieces.append((act.copy(), lo, hi))
        seen_wrong[act[(hi - lo) < (nxt - t[act]) * (1 - 1e-12) - 1e-15]] = True
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
            flops += 2 * T * H + 1 + H + (T - 1 - s0_) * E.F_TAN + 2 * C * H
            if use and try_row(row, (s0_, jj[idx])):
                alive[idx] = False; continue
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tn[idx] - Z[row, s0_:] / DZ[row, s0_:]
            r = np.where((r > tn[idx] + 1e-12 * (1 + abs(tn[idx]))) & np.isfinite(r), r, np.inf)
            r[0, jj[idx]] = np.inf
            R[row, s0_:] = r.min(1); J[row, s0_:] = r.argmin(1)
            flops += (T - s0_) * E.F_ROOT + E.F_LATER_READ - 2 * C * H; st["restarts"] += 1
        if (~alive).any():
            sr = go[~alive]; pieces.append(stop_piece(sr)); st["stopped"] += len(sr)
        go = go[alive]
        act = go
    return pieces, flops, st


def est(model, budget=None, lines=("nb-23", "nb-12"), parts=False, dmax=np.inf, maxtry=np.inf, mode="B", tmax=TMAX,
        ext=False):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = E.TRI[:, 1]; flops = 0.0; tot = np.zeros(NT); info = {}; prof = {}
    for tb in E.TABLES:
        nodes = tb["nodes"]; I = {}
        for kind in ("23", "12"):
            if f"line{kind}" not in tb:
                continue
            x, dx = tb[f"line{kind}"]; line = f"{tb['name']}-{kind}"
            p, f, st = walk_screen(model, x, dx, k, tmax, line in lines, dmax=dmax, maxtry=maxtry, mode=mode, ext=ext)
            flops += f
            R, A, B, ops = T331.merge_ext(p, NT) if ext else E.merge(p, NT, tmax); flops += ops
            I[kind], f = E.profile(R, A, B, NT, nodes, kind, tmax, tb["phi0"]); flops += f
            info[line] = st; prof[line] = I[kind]
        I12 = I["12"] if "12" in I else np.repeat(tb["blind12"][None], NT, 0)
        tot += (I12 * I["23"] * tb["dens"][None]) @ tb["wts"]
        flops += NT * len(nodes) * 4
    flops += NT
    if parts:
        return float(tot.mean()), float(flops), info, prof
    return float(tot.mean()), float(flops)
