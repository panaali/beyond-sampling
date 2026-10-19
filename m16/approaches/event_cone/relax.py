"""T390: prune a prefix cell when an LP relaxation of the remaining steps proves the far-gap event infeasible on it.
Cell = (G, A): the cone {G x >= 0} on which the first k steps have fixed gates, so h_k = A x exactly.
Relaxation R_k(cell): x in [-1, 1]^10, G x >= 0, g23 = x_p2 - x_p3 >= DELTA (homogeneous normalisation, README);
for steps k+1..10: z = Wh h_prev + Wi x_t, h = relu(z) replaced by its triangle relaxation with bounds [l, u] on z
computed by LP over the relaxation built so far (stable neurons exact). Event: Wo[p3].h10 >= Wo[j].h10 - EPS for
all j != p3. The cell is PRUNED iff that LP is infeasible."""
import numpy as np
from scipy.optimize import linprog

import cone_count as cc

DELTA = 0.1
EPS = 1e-7
Wi, Wh, Wo = cc.model()
H = 16; D = 10


def lp(c, A_ub, b_ub, A_eq, b_eq, bounds):
    return linprog(c, A_ub=A_ub if len(A_ub) else None, b_ub=b_ub if len(b_ub) else None,
                   A_eq=A_eq if len(A_eq) else None, b_eq=b_eq if len(b_eq) else None, bounds=bounds, method="highs")


def relax_infeasible(G, A, k, tri, bound_lp=True, return_stats=False):
    p1, p2, p3 = tri; nb = D - k; nv = D + H * nb
    ub, bu, eq, be = [], [], [], []
    for g in G:                                                    # G x >= 0  ->  -G x <= 0
        r = np.zeros(nv); r[:D] = -g; ub.append(r); bu.append(0.)
    r = np.zeros(nv); r[p2] = -1; r[p3] = 1; ub.append(r); bu.append(-DELTA)
    bounds = [(-1, 1)] * D + [(0, 0)] * (H * nb)
    prev = np.zeros((H, nv)); prev[:, :D] = A                      # h_k as rows over v
    nstab = 0
    for s in range(nb):
        t = k + s
        Z = Wh @ prev; Z[:, t] += Wi
        off = D + H * s
        if bound_lp:
            lo = np.empty(H); hi = np.empty(H)
            for j in range(H):
                r1 = lp(Z[j], np.array(ub), np.array(bu), np.array(eq), np.array(be), bounds)
                r2 = lp(-Z[j], np.array(ub), np.array(bu), np.array(eq), np.array(be), bounds)
                if r1.status == 2 or r2.status == 2:
                    return (True, dict(stage=s, nstab=nstab)) if return_stats else True
                lo[j] = r1.fun; hi[j] = -r2.fun
        for j in range(H):
            v = off + j; l, u = lo[j], hi[j]
            if u <= 0:
                bounds[v] = (0, 0); nstab += 1
            elif l >= 0:
                r = -Z[j].copy(); r[v] += 1; eq.append(r); be.append(0.); bounds[v] = (0, None); nstab += 1
            else:
                bounds[v] = (0, u)
                r = Z[j].copy(); r[v] -= 1; ub.append(r); bu.append(0.)            # h >= z
                r = -u / (u - l) * Z[j]; r[v] += 1; ub.append(r); bu.append(-u * l / (u - l))   # h <= u(z-l)/(u-l)
        prev = np.zeros((H, nv)); prev[:, off:off + H] = np.eye(H)
    L = Wo @ prev                                                  # logits rows over v
    for j in range(D):
        if j != p3:
            ub.append(L[j] - L[p3]); bu.append(EPS)
    r = lp(np.zeros(nv), np.array(ub), np.array(bu), np.array(eq), np.array(be), bounds)
    inf = r.status == 2
    return (inf, dict(stage=nb, nstab=nstab, status=int(r.status))) if return_stats else inf
