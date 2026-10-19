"""T398: tighter pruning of a prefix cell (README T398). Stages, each sound (it only removes non-event points):
 0 norm   : the cell with x in [-1,1]^10, g23 >= DELTA is empty (as T390)
 1 base   : T390's relaxation (triangle, LP bounds computed forward without the event) + event is infeasible
 2 event  : event-aware bound tightening: every unstable neuron's pre-activation bounds are recomputed by LP over
            the FULL stage-1 relaxation INCLUDING the event constraint (one Jacobi pass); then re-check
 3 split  : ReLU splitting of up to DEPTH unstable neurons (earliest step first, then largest triangle area);
            pruned iff every leaf LP (event included) is infeasible
CROWN-style back-substitution bounds are a relaxation of the same LP, so the LP bounds used here are at least as tight;
CROWN would only be faster."""
import numpy as np
from scipy.optimize import linprog

import cone_count as cc

DELTA = 0.1
EPS = 1e-7
DEPTH = 6
Wi, Wh, Wo = cc.model()
H = 16; D = 10


def solve(c, P, bounds):
    ub, bu, eq, be = P
    return linprog(c, A_ub=np.array(ub) if ub else None, b_ub=np.array(bu) if bu else None,
                   A_eq=np.array(eq) if eq else None, b_eq=np.array(be) if be else None, bounds=bounds, method="highs")


class Relax:
    def __init__(self, G, A, k, tri):
        self.k, self.tri = k, tri; self.nb = D - k; self.nv = D + H * self.nb
        p1, p2, p3 = tri
        self.base_ub, self.base_bu = [], []
        for g in G:
            r = np.zeros(self.nv); r[:D] = -g; self.base_ub.append(r); self.base_bu.append(0.)
        r = np.zeros(self.nv); r[p2] = -1; r[p3] = 1; self.base_ub.append(r); self.base_bu.append(-DELTA)
        self.Z = []                                                # pre-activation rows per remaining step
        prev = np.zeros((H, self.nv)); prev[:, :D] = A
        for s in range(self.nb):
            Zs = Wh @ prev; Zs[:, k + s] += Wi; self.Z.append(Zs)
            prev = np.zeros((H, self.nv)); prev[:, D + H * s:D + H * (s + 1)] = np.eye(H)
        L = Wo @ prev
        self.ev_ub = [L[j] - L[p3] for j in range(D) if j != p3]; self.ev_bu = [EPS] * (D - 1)
        self.lo = np.full((self.nb, H), -np.inf); self.hi = np.full((self.nb, H), np.inf)

    def var(self, s, j):
        return D + H * s + j

    def program(self, upto, event, fix=None):
        """Constraints for steps < upto with current bounds; fix: {(s, j): 0/1} split decisions."""
        ub = list(self.base_ub); bu = list(self.base_bu); eq, be = [], []
        bounds = [(-1, 1)] * D + [(0, 0)] * (H * self.nb)
        fix = fix or {}
        for s in range(upto):
            for j in range(H):
                v = self.var(s, j); z = self.Z[s][j]; l, u = self.lo[s, j], self.hi[s, j]; f = fix.get((s, j))
                if u <= 0 or f == 0:
                    bounds[v] = (0, 0)
                    if f == 0:
                        ub.append(z.copy()); bu.append(0.)          # z <= 0
                elif l >= 0 or f == 1:
                    r = -z.copy(); r[v] += 1; eq.append(r); be.append(0.); bounds[v] = (0, None)
                    if f == 1:
                        ub.append(-z.copy()); bu.append(0.)         # z >= 0
                else:
                    bounds[v] = (0, u)
                    r = z.copy(); r[v] -= 1; ub.append(r); bu.append(0.)
                    r = -u / (u - l) * z; r[v] += 1; ub.append(r); bu.append(-u * l / (u - l))
        if event:
            ub += self.ev_ub; bu += self.ev_bu
        return (ub, bu, eq, be), bounds

    def forward_bounds(self):
        for s in range(self.nb):
            P, bounds = self.program(s, False)
            for j in range(H):
                r1 = solve(self.Z[s][j], P, bounds); r2 = solve(-self.Z[s][j], P, bounds)
                if r1.status == 2 or r2.status == 2:
                    return False
                self.lo[s, j] = r1.fun; self.hi[s, j] = -r2.fun
        return True

    def feasible(self, fix=None):
        P, bounds = self.program(self.nb, True, fix)
        return solve(np.zeros(self.nv), P, bounds).status != 2

    def event_tighten(self):
        P, bounds = self.program(self.nb, True)
        lo, hi = self.lo.copy(), self.hi.copy()
        for s in range(self.nb):
            for j in range(H):
                if self.lo[s, j] >= 0 or self.hi[s, j] <= 0:
                    continue
                r1 = solve(self.Z[s][j], P, bounds); r2 = solve(-self.Z[s][j], P, bounds)
                if r1.status == 2 or r2.status == 2:
                    return False
                lo[s, j] = max(lo[s, j], r1.fun); hi[s, j] = min(hi[s, j], -r2.fun)
        self.lo, self.hi = lo, hi
        return True

    def split_order(self):
        c = [(s, -(u * -l) / (u - l), j) for s in range(self.nb) for j in range(H)
             for l, u in [(self.lo[s, j], self.hi[s, j])] if l < 0 < u]
        return [(s, j) for s, _, j in sorted(c)]

    def split_prune(self, order, depth):
        self.leaves = []
        def rec(fix, i):
            if not self.feasible(fix):
                self.leaves.append(dict(fix)); return True
            if i >= min(depth, len(order)):
                return False
            sj = order[i]
            return rec({**fix, sj: 1}, i + 1) and rec({**fix, sj: 0}, i + 1)
        return rec({}, 0)


def prune_stage(G, A, k, tri, depth=DEPTH, event_aware=True, cert=None):
    """Returns the stage that prunes the cell (0..3) or -1 if it survives. If cert is a list, appends the leaves
    (list of fix dicts {(s, j): 0/1}, s = remaining-step index) of a pruned cell."""
    R = Relax(G, A, k, tri)
    if not R.forward_bounds():
        st = 0
    elif not R.feasible():
        st = 1
    elif event_aware and (not R.event_tighten() or not R.feasible()):
        st = 2
    elif depth > 0 and R.split_prune(R.split_order(), depth):
        st = 3
    else:
        return -1
    if cert is not None:
        cert.extend(R.leaves if st == 3 else [{}])
    return st


def max_margin(R, fix=None):
    """Upper bound from the relaxation on max_x (logit_p3 - max_{j != p3} logit_j) (event constraint dropped)."""
    P, bounds = R.program(R.nb, False, fix)
    ub, bu, eq, be = P; nv = R.nv
    ub = [np.r_[r, 0.] for r in ub]; eq = [np.r_[r, 0.] for r in eq]
    for r in R.ev_ub:                                              # L_j - L_p3 + t <= 0
        ub.append(np.r_[r, 1.]); bu.append(0.)
    c = np.zeros(nv + 1); c[-1] = -1
    res = solve(c, (ub, bu, eq, be), bounds + [(None, None)])
    return -res.fun if res.status == 0 else np.nan
