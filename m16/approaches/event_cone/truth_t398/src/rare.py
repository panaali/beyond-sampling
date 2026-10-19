"""T340 deductive rare-event estimator: P(network picks the rank-3 position AND g23 > t0), from T317f's exact line23
gate-cell walks (cont_g.py with SPREAD = 0 = T317f's closure), with no tmax cut (walk to TBIG) and no screen C1; in each
cell the exact interval where the rank-3 position is the argmax; its mass on t >= t0 under the model-free law of g23 | v2
(Phi^8 profile); v2 integrated with T317's G = 20 Gauss-Legendre nodes per region; full g12 mass (1 - Phi(v2)).
Variant A: T317's transverse point per region (one walk per triple per region).
Variant B: one walk per triple per GL node, node-specific transverse points (E[g12|v2], E[g23|v2], v4 | v3*).
est(model, t0s, variant) -> ({t0: estimate}, FLOPs as executed). Deductive: no random numbers."""
import os
import sys

import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "closure_floor"))
import cont_g as G  # noqa: E402

TBIG = 6.0
T, H, C, NT, TRI = G.T, G.H, G.C, G.NT, G.TRI


def walk3(model, x0, d, k, k3, tmax, max_cells=5000):
    """G.walk_cont's loop (T317 continuity restarts, prefix-tree first cell), collecting per cell the interval where k3
    is the argmax. Charges: walk_counted's (honest correct_interval count) + one extra correct_interval per cell."""
    Wi, Wh, Wo = model; n = len(x0); pieces = []; pieces_c = []
    Z, DZ, Gt, R, J, nodes = G.first_cell(model, x0, d)
    flops = nodes * G.F_STEP + n * G.F_CELL
    tref = np.full(n, G.EPS)
    hT = np.where(Gt[:, -1], Z[:, -1], 0.0); dhT = np.where(Gt[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0; ncell = 0
    tm = np.broadcast_to(np.asarray(tmax, float), (n,))         # per-row window end (T362); scalar for A/B
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tm[act])
        lo, hi = G.correct_interval(L[act], dL[act], tref[act], k3[act], t[act], nxt)
        flops += len(act) * G.F_CI; ncell += len(act)
        pieces.append((act.copy(), lo, hi))
        lo, hi = G.correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)   # target: charged in F_CELL / F_LATER_READ
        pieces_c.append((act.copy(), lo, hi))
        t[act] = nxt
        go = act[nxt < tm[act]]; ss = sstar[nxt < tm[act]]
        if not len(go):
            break
        tn = t[go]
        Z[go] += (tn - tref[go])[:, None, None] * DZ[go]; L[go] += (tn - tref[go])[:, None] * dL[go]; tref[go] = tn
        for idx in range(len(go)):
            # T340 fix (bug found by the T218 child): several pre-activations can reach 0 at the same t (e.g. every
            # step-0 neuron when an input crosses 0). Re-gate every step >= s* from its value, with the sign of its
            # new slope for the ones at 0, instead of toggling only the argmin gate.
            row, s0 = go[idx], ss[idx]
            dh = None
            for s in range(s0, T):
                if s > s0:
                    DZ[row, s] = Wh @ dh + d[row, s] * Wi
                z = Z[row, s]; dz = DZ[row, s]
                near = np.abs(z) <= 1e-9 * (1.0 + np.abs(z).max())
                Gt[row, s] = np.where(near, dz > 0, z > 0)
                dh = np.where(Gt[row, s], dz, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tn[idx] - Z[row, s0:] / DZ[row, s0:]
            nearall = np.abs(Z[row, s0:]) <= 1e-9 * (1.0 + np.abs(Z[row, s0:]).max(1, keepdims=True))
            r = np.where((r > tn[idx] + 1e-12 * (1 + abs(tn[idx]))) & np.isfinite(r) & ~nearall, r, np.inf)
            R[row, s0:] = r.min(1); J[row, s0:] = r.argmin(1)
            dL[row] = Wo @ dh
            flops += (2 * T * H + 1 + H + (T - 1 - s0) * G.F_TAN + (T - s0) * G.F_ROOT + G.F_LATER_READ
                      + (T - s0) * 4 * H)                        # re-gating: |z| test, max, select, compare per step
        act = go
    # charges: the target's correct_interval per cell is inside F_CELL / F_LATER_READ (T317 count); the k3 interval
    # is one extra F_CI per cell.
    return pieces, pieces_c, flops, it, ncell


def mass(pieces, nrow, node_of_row, nodes, t0):
    """Sum over pieces of P(g23 in [max(lo, t0), hi] | v2 = node) x Phi(v2)^8 normaliser, i.e. Phi(v2-a)^8 - Phi(v2-b)^8."""
    out = np.zeros((nrow, len(nodes))); f = 0.0
    for rows, lo, hi in pieces:
        a = np.maximum(lo, t0); m = hi > a
        if not m.any():
            continue
        r, a, b = rows[m], a[m], hi[m]
        if node_of_row is None:
            val = ndtr(nodes[None] - a[:, None]) ** 8 - ndtr(nodes[None] - b[:, None]) ** 8
            np.add.at(out, r, val); f += len(r) * len(nodes) * (2 * (G.F_PHI + 3) + 2)
        else:
            j = node_of_row[r]; nd = nodes[j]
            val = ndtr(nd - a) ** 8 - ndtr(nd - b) ** 8
            np.add.at(out, (r, j), val); f += len(r) * (2 * (G.F_PHI + 3) + 2)
    return out, f


def _cond_g23(v2, n=20001):
    v = np.linspace(v2 - 12, v2, n); f = (ndtr(v) / ndtr(v2)) ** 8
    return float(((f[1:] + f[:-1]) / 2 * np.diff(v)).sum())


def _node_lines():
    """Variant B: per region, line23 inputs for every (triple, GL node), node-specific transverse points (model-free)."""
    out = []
    for t in G.TABLES:
        xs, dxs = [], []
        for v2 in t["nodes"]:
            g12 = float(G._pdf(v2) / max(ndtr(-v2), 1e-300) - v2)
            v3s = v2 - _cond_g23(v2); v4s = float(G.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
            m4 = float(G.mu(np.array([v4s]))[0]); dm4 = float(G.dmu(v4s)[0])
            x, dx = G.inputs(v2 + g12, v2, v2, v2 - delta, m4 + dm4 * (v2 - delta - v4s), 0.0, -1.0, -1.0, -dm4)
            xs.append(x); dxs.append(dx)
        # row order: node-major (node j, triple i) -> row j * NT + i
        out.append((np.concatenate(xs), np.concatenate(dxs)))
    return out


NODE_LINES = _node_lines()                    # model-free, once at import


def est(model, t0s=(0.0, 1.0, 1.25, 1.5), variant="A", tmax=TBIG, t2s=(0.5, 0.75)):
    """-> ({t0: E1 estimate}, {t: E2 estimate}, FLOPs, info)"""
    model = tuple(np.asarray(w, np.float64) for w in model)
    res = {t0: 0.0 for t0 in t0s}; res2 = {t: 0.0 for t in t2s}; flops = 0.0; info = {}
    for ri, tab in enumerate(G.TABLES):
        nodes, wts = tab["nodes"], tab["wts"]; wgt = tab["dens"] * tab["blind12"]          # 90 phi(v2) (1 - Phi(v2))
        if variant == "A":
            x, dx = tab["line23"]; k, k3 = TRI[:, 1], TRI[:, 2]; nrow = NT; node_of_row = None
        else:
            x, dx = NODE_LINES[ri]; k = np.tile(TRI[:, 1], len(nodes)); k3 = np.tile(TRI[:, 2], len(nodes))
            nrow = len(x); node_of_row = np.repeat(np.arange(len(nodes)), NT)
        pieces, pieces_c, f, it, ncell = walk3(model, x, dx, k, k3, tmax); flops += f
        info[tab["name"]] = dict(iters=it, cells_per_row=ncell / nrow)
        for t0 in t0s:
            W, f = mass(pieces, nrow, node_of_row, nodes, t0); flops += f
            res[t0] += float((W * wgt[None]).sum(0) @ wts) / NT if variant == "A" else float(W.sum(0) * wgt @ wts) / NT
            flops += nrow * len(nodes) * 2
        for t2 in t2s:            # E2: any-wrong on line23 beyond t2 (complement of the correct set), g12 mass beyond t2
            Cm, f = mass(pieces_c, nrow, node_of_row, nodes, t2); flops += f
            tot = ndtr(nodes - t2) ** 8 - ndtr(nodes - tmax) ** 8            # P(g23 in [t2, tmax]) x Phi(v2)^8, model-free
            W = (tot[None] - Cm) if node_of_row is None else None
            if node_of_row is not None:
                W = np.zeros_like(Cm); jj = np.repeat(np.arange(len(nodes)), NT); W[np.arange(nrow), jj] = tot[jj] - Cm[np.arange(nrow), jj]
            w2 = tab["dens"] * ndtr(-(nodes + t2))                                 # 90 phi(v2) (1 - Phi(v2 + t2))
            res2[t2] += float((W * w2[None]).sum(0) @ wts) / NT
            flops += nrow * len(nodes) * 3
    return res, res2, flops, info
