"""Optional (not scored) check: per-triple MIQP min ||x||^2 over the (9,3,2) part of K ∩ {g23 >= 1}, with order
constraints (tighter OBBT), warm-started with the best descent point. Reports SCIP's dual bound: if it equals the
incumbent, no lower-norm point exists in this triple on the box. usage: python miqp_check.py [time_limit]"""
import json, sys, time
import numpy as np
from scipy.optimize import linprog
from net import Wi, Wh, Wo, H, T

TRI = (9, 3, 2); B = 4.0


def order_rows():
    a, b, c = TRI; e = np.eye(10); R = [e[b] - e[a], e[c] - e[b]]; rhs = [0.0, -1.0]        # R x <= rhs
    R += [e[k] - e[c] for k in range(10) if k not in TRI]; rhs += [0.0] * 7
    return np.array(R), np.array(rhs)


def obbt():
    Ro, ro = order_rows(); L = np.zeros((T, H)); U = np.zeros((T, H))
    for t in range(T):
        nv = 10 + 32 * t; A_ub = [np.r_[r, np.zeros(nv - 10)] for r in Ro]; b_ub = list(ro); A_eq, b_eq = [], []
        bnds = [(-B, B)] * 10
        for s in range(t):
            ia, ih = 10 + 32 * s, 10 + 32 * s + 16
            for k in range(H):
                row = np.zeros(nv); row[ia + k] = -1; row[s] = Wi[k]
                if s > 0:
                    row[10 + 32 * (s - 1) + 16: 10 + 32 * s] = Wh[k]
                A_eq.append(row); b_eq.append(0.0); l, u = L[s, k], U[s, k]
                if u <= 0:
                    r = np.zeros(nv); r[ih + k] = 1; A_eq.append(r); b_eq.append(0.0)
                elif l >= 0:
                    r = np.zeros(nv); r[ih + k] = 1; r[ia + k] = -1; A_eq.append(r); b_eq.append(0.0)
                else:
                    r = np.zeros(nv); r[ih + k] = -1; r[ia + k] = 1; A_ub.append(r); b_ub.append(0.0)
                    r = np.zeros(nv); r[ih + k] = 1; r[ia + k] = -u / (u - l); A_ub.append(r); b_ub.append(-u * l / (u - l))
            bnds += [(L[s, k], U[s, k]) for k in range(H)] + [(0, max(U[s, k], 0)) for k in range(H)]
        for k in range(H):
            c = np.zeros(nv); c[t] = Wi[k]
            if t > 0:
                c[10 + 32 * (t - 1) + 16: 10 + 32 * t] = Wh[k]
            v = [s_ * linprog(s_ * c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq) if A_eq else None,
                               b_eq=b_eq or None, bounds=bnds, method="highs").fun for s_ in (1, -1)]
            L[t, k], U[t, k] = v[0] - 1e-7, v[1] + 1e-7
    return L, U


def main(tl):
    from pyscipopt import Model, quicksum
    t = time.time(); L, U = obbt(); tb = time.time() - t
    P = json.load(open("points.json"))["points"]; x0 = np.array([p for p in P if tuple(p["tri"]) == TRI][0]["x"])
    m = Model(); m.hideOutput(); m.setParam("limits/time", tl)
    x = [m.addVar(lb=-B, ub=B) for _ in range(10)]; hp = [0.0] * H; nb = 0
    for t_ in range(T):
        hn = []
        for k in range(H):
            a = quicksum(Wh[k, i] * hp[i] for i in range(H) if t_ > 0) + Wi[k] * x[t_]; l, u = L[t_, k], U[t_, k]
            if u <= 0:
                hn.append(0.0)
            elif l >= 0:
                v = m.addVar(lb=0, ub=u); m.addCons(v == a); hn.append(v)
            else:
                v = m.addVar(lb=0, ub=u); d = m.addVar(vtype="B"); nb += 1
                m.addCons(v >= a); m.addCons(v <= a - l * (1 - d)); m.addCons(v <= u * d); hn.append(v)
        hp = hn
    lg = [quicksum(Wo[c, i] * hp[i] for i in range(H)) for c in range(10)]
    Ro, ro = order_rows()
    for r, b in zip(Ro, ro):
        m.addCons(quicksum(float(r[j]) * x[j] for j in range(10)) <= float(b))
    for j in range(10):
        if j != TRI[2]:
            m.addCons(lg[TRI[2]] >= lg[j])
    s = m.addVar(lb=0); m.addCons(s >= quicksum(x[j] * x[j] for j in range(10))); m.setObjective(s, "minimize")
    m.setParam("limits/objectivestop", -1e20) if False else None
    t = time.time(); m.optimize(); st = m.getStatus()
    out = dict(triple=TRI, obbt_sec=tb, binaries=nb, max_bound=float(np.abs(np.r_[L, U]).max()), status=st,
               sec=time.time() - t, incumbent_descent=float(x0 @ x0),
               primal=m.getPrimalbound() if m.getNSols() else None, dual=m.getDualbound())
    json.dump(out, open("miqp_check.json", "w"), indent=1); print(out, flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1800)
