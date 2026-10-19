"""T354: the continuity walk with a fix for SIMULTANEOUS gate flips (README, T354 bug note).
Bug in walk_cont / T340's walk3: at a flip t* only the argmin gate (s*, j*) is toggled. When several pre-activations hit
0 at the same t* (e.g. every step-0 neuron, z = x_0 Wi, when the input coordinate x_0 crosses 0), the others have roots
exactly at t*, the strict "root > t*" filter drops them, and their gates are never flipped.
Fix: at each restart, re-gate every step s >= s* from the values: gate = z > 0 if |z| > tol, else dz > 0 (the side the
line moves into); tangents are recomputed from s* with the new gates and roots from s* as before. Charged: + 2H per
re-gated step. walk3_fixed has rare.walk3's signature and returns."""
import numpy as np


def walk3_fixed(G, model, x0, d, k, k3, tmax, max_cells=5000):
    Wi, Wh, Wo = model; n = len(x0); pieces = []; pieces_c = []; T, H = G.T, G.H
    Z, DZ, Gt, R, J, nodes = G.first_cell(model, x0, d)
    flops = nodes * G.F_STEP + n * G.F_CELL
    tref = np.full(n, G.EPS)
    hT = np.where(Gt[:, -1], Z[:, -1], 0.0); dhT = np.where(Gt[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0; ncell = 0; nsim = 0
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = G.correct_interval(L[act], dL[act], tref[act], k3[act], t[act], nxt)
        flops += len(act) * G.F_CI; ncell += len(act)
        pieces.append((act.copy(), lo, hi))
        lo, hi = G.correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)
        pieces_c.append((act.copy(), lo, hi))
        t[act] = nxt
        go = act[nxt < tmax]; ss = sstar[nxt < tmax]
        if not len(go):
            break
        tn = t[go]
        Z[go] += (tn - tref[go])[:, None, None] * DZ[go]; L[go] += (tn - tref[go])[:, None] * dL[go]; tref[go] = tn
        for idx in range(len(go)):
            row, s0 = go[idx], ss[idx]
            dz = DZ[row, s0]                                     # steps < s* unchanged, so dz at s* is current
            for s in range(s0, T):
                if s > s0:
                    dz = Wh @ dh + d[row, s] * Wi
                z = Z[row, s]; tol = 1e-9 * (1.0 + np.abs(z).max())
                g = np.where(np.abs(z) > tol, z > 0, dz > 0)
                nsim += int((g != Gt[row, s]).sum()) - (1 if s == s0 else 0)
                Gt[row, s] = g; DZ[row, s] = dz; dh = np.where(g, dz, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tn[idx] - Z[row, s0:] / DZ[row, s0:]
            ztol = 1e-9 * (1.0 + np.abs(Z[row, s0:]).max(1, keepdims=True))
            r = np.where((r > tn[idx] + 1e-12 * (1 + abs(tn[idx]))) & np.isfinite(r) & (np.abs(Z[row, s0:]) > ztol), r, np.inf)
            R[row, s0:] = r.min(1); J[row, s0:] = r.argmin(1)
            dL[row] = Wo @ dh
            flops += 2 * T * H + 1 + H + (T - 1 - s0) * G.F_TAN + (T - s0) * (G.F_ROOT + 2 * H) + G.F_LATER_READ
        act = go
    return pieces, pieces_c, flops, it, ncell, nsim
