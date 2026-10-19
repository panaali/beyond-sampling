"""Dominating points of E1 for M_16,10 by MIQP (SCIP) over the unrolled net (10 steps x 16 ReLUs = 160 binaries).

Scale invariance: the net has no biases, so {argmax logits = pos x_(3)} is a cone K and E1(t0) = K ∩ {g23 >= t0};
dominating points of E1(t0) are t0 times those of E1(1). We therefore solve at t0 = 1 on the box ||x||_inf <= B.

Point i: x_i = argmin ||x||^2 s.t. x in K ∩ {g23 >= 1} ∩ box, and x_j^T (x - x_j) <= -EPS ||x_j||^2 for all earlier j
(Bai, Huang, Lam & Zhao 2020: the union of halfspaces {x_j^T (x - x_j) >= 0} then covers the event).
Triple-free encoding: binaries z_c (c = predicted class = position of x_(3)), y_j (j one of the two inputs above it):
  y_j = 1, z_c = 1 -> x_j - x_c >= 1 ; y_j = 0, z_c = 1 -> x_j <= x_c ; z_c = 1 -> l_c >= l_j for all j.
ReLU h = max(a, 0) with LP-tightened pre-activation bounds [L, U] (OBBT on the triangle relaxation over the box)."""
import json, os, sys, time
import numpy as np
from scipy.optimize import linprog
from net import Wi, Wh, Wo, H, T

B = 4.0          # box at t0 = 1 (= 10 at t0 = 2.5)
EPS = 1e-6


def bounds_obbt(B=B):
    """Pre-activation bounds per step (T, H) by LP over the triangle relaxation of earlier steps."""
    L = np.zeros((T, H)); U = np.zeros((T, H))
    # variables: x (10), then for each step t < current: a_t (16), h_t (16)
    for t in range(T):
        nv = 10 + 32 * t
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        bnds = [(-B, B)] * 10
        for s in range(t):
            ia, ih = 10 + 32 * s, 10 + 32 * s + 16
            for k in range(H):
                row = np.zeros(nv); row[ia + k] = -1; row[s] = Wi[k]
                if s > 0:
                    row[10 + 32 * (s - 1) + 16: 10 + 32 * (s - 1) + 32] = Wh[k]
                A_eq.append(row); b_eq.append(0.0)                       # a = Wh h_{s-1} + Wi x_s
                l, u = L[s, k], U[s, k]
                if u <= 0:
                    r = np.zeros(nv); r[ih + k] = 1; A_eq.append(r); b_eq.append(0.0)
                elif l >= 0:
                    r = np.zeros(nv); r[ih + k] = 1; r[ia + k] = -1; A_eq.append(r); b_eq.append(0.0)
                else:
                    r = np.zeros(nv); r[ih + k] = -1; r[ia + k] = 1; A_ub.append(r); b_ub.append(0.0)   # h >= a
                    r = np.zeros(nv); r[ih + k] = 1; r[ia + k] = -u / (u - l); A_ub.append(r); b_ub.append(-u * l / (u - l))
            bnds += [(L[s, k], U[s, k]) for k in range(H)] + [(0, max(U[s, k], 0)) for k in range(H)]
        for k in range(H):
            c = np.zeros(nv); c[t] = Wi[k]
            if t > 0:
                c[10 + 32 * (t - 1) + 16: 10 + 32 * t] = Wh[k]
            vals = []
            for sgn in (1, -1):
                r = linprog(sgn * c, A_ub=np.array(A_ub) if A_ub else None, b_ub=b_ub or None,
                            A_eq=np.array(A_eq) if A_eq else None, b_eq=b_eq or None, bounds=bnds, method="highs")
                vals.append(sgn * r.fun)
            L[t, k], U[t, k] = vals[0] - 1e-7, vals[1] + 1e-7
    return L, U


def build(L, U, excl, time_limit=600, B=B):
    from pyscipopt import Model, quicksum
    m = Model(); m.hideOutput()
    m.setParam("limits/time", time_limit)
    x = [m.addVar(lb=-B, ub=B, name=f"x{j}") for j in range(10)]
    hp = [0.0] * H
    for t in range(T):
        hn = []
        for k in range(H):
            a = quicksum(Wh[k, i] * hp[i] for i in range(H) if t > 0) + Wi[k] * x[t]
            l, u = L[t, k], U[t, k]
            if u <= 0:
                hn.append(0.0)
            elif l >= 0:
                v = m.addVar(lb=0, ub=u); m.addCons(v == a); hn.append(v)
            else:
                v = m.addVar(lb=0, ub=u); d = m.addVar(vtype="B")
                m.addCons(v >= a); m.addCons(v <= a - l * (1 - d)); m.addCons(v <= u * d); hn.append(v)
        hp = hn
    lg = [quicksum(Wo[c, i] * hp[i] for i in range(H)) for c in range(10)]
    # logit bounds for big-M
    lo = np.array([sum(min(Wo[c, i] * 0, Wo[c, i] * max(U[-1, i], 0)) for i in range(H)) for c in range(10)])
    hi = np.array([sum(max(Wo[c, i] * 0, Wo[c, i] * max(U[-1, i], 0)) for i in range(H)) for c in range(10)])
    z = [m.addVar(vtype="B") for _ in range(10)]; y = [m.addVar(vtype="B") for _ in range(10)]
    m.addCons(quicksum(z) == 1); m.addCons(quicksum(y) == 2)
    Mx = 2 * B + 1
    for j in range(10):
        m.addCons(y[j] + z[j] <= 1)
    for c in range(10):
        for j in range(10):
            if j == c:
                continue
            m.addCons(x[j] - x[c] >= 1 - Mx * (2 - z[c] - y[j]))
            m.addCons(x[j] - x[c] <= Mx * (1 - z[c] + y[j]))
            m.addCons(lg[c] - lg[j] >= -(hi[j] - lo[c]) * (1 - z[c]))
    for e in excl:
        e = np.asarray(e); m.addCons(quicksum(float(e[j]) * x[j] for j in range(10)) <= (1 - EPS) * float(e @ e))
    s = m.addVar(lb=0); m.addCons(s >= quicksum(x[j] * x[j] for j in range(10)))
    m.setObjective(s, "minimize")
    return m, x


def solve(L, U, excl, time_limit=600):
    m, x = build(L, U, excl, time_limit)
    t = time.time(); m.optimize(); st = m.getStatus()
    out = dict(status=st, sec=time.time() - t, gap=None, x=None, r2=None, dual=None)
    if m.getNSols() > 0:
        xv = np.array([m.getVal(v) for v in x]); out.update(x=xv.tolist(), r2=float(xv @ xv), gap=m.getGap(), dual=m.getDualbound())
    elif st == "infeasible":
        pass
    return out


if __name__ == "__main__":
    t = time.time(); L, U = bounds_obbt(); print("OBBT %.0fs; max |bound| per step:" % (time.time() - t), np.abs(np.r_[L, U]).reshape(2, T, H).max((0, 2)).round(1), flush=True)
    np.savez("bounds.npz", L=L, U=U)
    r = solve(L, U, [], time_limit=float(sys.argv[1]) if len(sys.argv) > 1 else 600)
    print(r, flush=True)
