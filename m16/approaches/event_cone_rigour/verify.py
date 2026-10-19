"""T400 part (b): rigorous verification that a prefix cell (or split leaf) carries no event point.

verify_prune(tri, gates, k, event_aware=False) -> bool   (or (bool, info) with return_info=True)
  gates: int (10, 16); rows < k are the cell's exact gate history (1 on / 0 off); rows >= k may hold ReLU split
  decisions (1: z >= 0 and h = z; 0: z <= 0 and h = 0) or -1 (free). Proves, if it can,
     {x in [-1,1]^10, x_p2 - x_p3 >= DELTA} ∩ cell ∩ splits ∩ {L_p3 >= L_j for all j} = ∅
  by rebuilding T390/T398's triangle relaxation (relax.py / relax2.py stages 0-3; a stage-3 prune is verified
  leaf by leaf with verify_leaves). Every pre-activation bound is a Neumaier–Shcherbina safe dual bound: float LP
  duals from HiGHS, residual absorbed by the variable box, every data and rounding error bounded explicitly.
  The last step proves min over the relaxation of max_j (L_j - L_p3) > 0 the same way (phase 1 with a boxed
  tau). A float LP status never decides anything by itself.

Error model (IEEE double, round to nearest, u = 2^-53): a length-n float dot product has
|fl - exact| <= gamma_n * sum |terms|, gamma_n = n u / (1 - n u). Each error bound is computed in float and
multiplied by SAFE = 2, which covers the rounding of the bound computation itself (relative error <= gamma_N << 1)."""
import os
import sys

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import common  # noqa: E402

Wi, Wh, Wo = common.Wi, common.Wh, common.Wo
H, D = 16, 10
DELTA = 0.1          # the float 0.1, used as the exact rational 0.1000000000000000055... (README part c)
U = 2.0 ** -53
SAFE = 2.0
TBOX = 1e3           # box of the phase-1 variable


def gam(n):
    return n * U / (1 - n * U)


def cell_rows(gates, k):
    """Float rows G (true cell: G_true x >= 0, |G_true - G| <= EG) and h_k = A x (|A_true - A| <= EA)."""
    A = np.zeros((H, D)); EA = np.zeros((H, D)); G = []; EG = []
    aWh = np.abs(Wh)
    for t in range(k):
        Z = Wh @ A; Z[:, t] += Wi
        EZ = SAFE * (aWh @ EA + gam(H + 1) * (aWh @ np.abs(A) + np.abs(Z)))
        g = gates[t]
        for j in range(H):
            if g[j] == 1:
                G.append(Z[j]); EG.append(EZ[j])
            elif g[j] == 0:
                G.append(-Z[j]); EG.append(EZ[j])
            else:
                raise ValueError("free gate inside the prefix")
        m = (g == 1).astype(float)[:, None]
        A = m * Z; EA = m * EZ
    return np.array(G).reshape(-1, D), np.array(EG).reshape(-1, D), A, EA


def safe_lb(c, Ec, A, EA, b, Eb, lo, hi, y):
    """Rigorous lower bound on min c.v s.t. A_true v >= b_true, lo <= v <= hi, for any y >= 0.
    Weak duality: c.v = y.(A v) + r.v >= y.b + min_box r.v with r = c - A'y; data errors (Ec, EA, Eb) and float
    rounding enter as an explicit error term."""
    y = np.maximum(y, 0.0)
    m, n = A.shape
    r = c - A.T @ y
    Er = Ec + EA.T @ y + gam(m + 1) * (np.abs(c) + np.abs(A).T @ y)
    yb = y @ b
    Eyb = y @ Eb + gam(m) * (y @ np.abs(b))
    tl = np.minimum(r * lo, r * hi)
    M = np.maximum(np.abs(lo), np.abs(hi))
    err = Eyb + Er @ M + gam(n + 2) * (np.abs(tl).sum() + abs(yb) + Er @ M)
    return yb + tl.sum() - SAFE * err


def _lp(c, A, b, lo, hi):
    return linprog(c, A_ub=-A, b_ub=-b, bounds=list(zip(lo, hi)), method="highs")


class Relax:
    """v = (x, h_{k+1}, ..., h_10); rows A v >= b with error bounds; box lo <= v <= hi."""

    def __init__(self, tri, gates, k, delta=DELTA):
        self.tri, self.k, self.gates = tri, k, np.asarray(gates)
        self.nb = D - k; self.nv = D + H * self.nb
        G, EG, A, EA = cell_rows(self.gates, k)
        p2, p3 = tri[1], tri[2]
        pad = lambda M: np.hstack([M, np.zeros((len(M), self.nv - D))])  # noqa: E731
        self.base = [pad(G)]; self.ebase = [pad(EG)]
        p1 = tri[0]; e = np.eye(self.nv)
        order = [e[p1] - e[p2]] + [e[p3] - e[l] for l in range(D) if l not in tri]   # ordering cone (exact)
        self.base.append(np.array(order)); self.ebase.append(np.zeros((len(order), self.nv)))
        r = np.zeros(self.nv); r[p2], r[p3] = 1, -1
        self.base.append(r[None]); self.ebase.append(np.zeros((1, self.nv)))
        self.bbase = np.r_[np.zeros(len(G) + len(order)), float(delta)]
        self.Z, self.EZ = [], []
        prev = np.zeros((H, self.nv)); prev[:, :D] = A
        eprev = np.zeros((H, self.nv)); eprev[:, :D] = EA
        for s in range(self.nb):
            Z = Wh @ prev
            EZ = SAFE * (np.abs(Wh) @ eprev + gam(H + 1) * (np.abs(Wh) @ np.abs(prev) + np.abs(Z)))
            Z[:, k + s] += Wi
            self.Z.append(Z); self.EZ.append(EZ)
            prev = np.zeros((H, self.nv)); prev[:, D + H * s:D + H * (s + 1)] = np.eye(H)
            eprev = np.zeros((H, self.nv))
        L = Wo @ prev
        self.ev = np.array([L[p3] - L[j] for j in range(D) if j != p3])        # event: ev v >= 0
        self.eev = np.array([SAFE * (gam(H) * (np.abs(Wo[p3]) + np.abs(Wo[j])) @ np.abs(prev) + U * np.abs(L[p3] - L[j]))
                             for j in range(D) if j != p3])
        self.lo = np.full((self.nb, H), -np.inf); self.hi = np.full((self.nb, H), np.inf)

    def program(self, upto, event):
        rows = list(self.base); errs = list(self.ebase); bs = [self.bbase]
        lo = np.r_[-np.ones(D), np.zeros(H * self.nb)]; hi = np.r_[np.ones(D), np.zeros(H * self.nb)]
        for s in range(upto):
            for j in range(H):
                v = D + H * s + j; z = self.Z[s][j]; ez = self.EZ[s][j]
                l, u = self.lo[s, j], self.hi[s, j]; f = self.gates[self.k + s][j]
                e = np.zeros(self.nv); e[v] = 1
                if f == 0 or u <= 0:
                    if f == 0:
                        rows.append(-z[None]); errs.append(ez[None]); bs.append(np.zeros(1))
                    continue                                     # h = 0
                if f == 1 or l >= 0:                             # h = z
                    rows += [(e - z)[None], (z - e)[None]]; errs += [ez[None], ez[None]]; bs.append(np.zeros(2))
                    lo[v], hi[v] = max(l, 0.0), u
                    if f == 1:
                        rows.append(z[None]); errs.append(ez[None]); bs.append(np.zeros(1))
                    continue
                hi[v] = u                                        # triangle
                alpha = u / (u - l); beta = -alpha * l
                # chord validity (alpha l + beta >= 0, alpha u + beta >= u) made safe by raising beta
                beta += SAFE * gam(4) * (abs(alpha * l) + abs(alpha * u) + abs(beta) + abs(u))
                rows += [(e - z)[None], (alpha * z - e)[None]]
                errs += [ez[None], (alpha * ez + gam(2) * np.abs(alpha * z))[None]]
                bs.append(np.array([0.0, -beta]))
        if event:
            rows.append(self.ev); errs.append(self.eev); bs.append(np.zeros(len(self.ev)))
        return np.vstack(rows), np.vstack(errs), np.concatenate(bs), np.zeros(sum(len(x) for x in bs)), lo, hi


def bound(c, Ec, A, EA, b, Eb, lo, hi):
    """Safe lower bound of min c.v; None if the float LP says infeasible."""
    r = _lp(c, A, b, lo, hi)
    if r.status == 2:
        return None
    if r.status != 0:
        return -np.inf
    return safe_lb(c, Ec, A, EA, b, Eb, lo, hi, -r.ineqlin.marginals)


def margin_lb(A, EA, b, Eb, lo, hi, extra=None, eextra=None):
    """Safe lower bound of min tau s.t. A v + tau >= b (extra rows: E v + tau >= 0), tau in [-TBOX, TBOX]."""
    m, n = A.shape
    if extra is None:
        A1 = np.hstack([A, np.ones((m, 1))]); E1 = np.hstack([EA, np.zeros((m, 1))]); b1 = b; Eb1 = Eb
    else:
        A1 = np.vstack([np.hstack([A, np.zeros((m, 1))]), np.hstack([extra, np.ones((len(extra), 1))])])
        E1 = np.vstack([np.hstack([EA, np.zeros((m, 1))]), np.hstack([eextra, np.zeros((len(extra), 1))])])
        b1 = np.r_[b, np.zeros(len(extra))]; Eb1 = np.r_[Eb, np.zeros(len(extra))]
    c = np.zeros(n + 1); c[-1] = 1
    lo1, hi1 = np.r_[lo, -TBOX], np.r_[hi, TBOX]
    r = _lp(c, A1, b1, lo1, hi1)
    if r.status != 0:
        return -np.inf
    return safe_lb(c, np.zeros(n + 1), A1, E1, b1, Eb1, lo1, hi1, -r.ineqlin.marginals)


def verify_prune(tri, gates, k, event_aware=False, return_info=False, delta=DELTA):
    R = Relax(tri, gates, k, delta)
    info = dict(stage=None, lps=0)

    def done(ok, stage):
        info["stage"] = stage
        return (ok, info) if return_info else ok

    for s in range(R.nb):                                        # forward bounds (event dropped)
        P = R.program(s, False)
        for j in range(H):
            info["lps"] += 2
            a = bound(R.Z[s][j], R.EZ[s][j], *P)
            c = bound(-R.Z[s][j], R.EZ[s][j], *P) if a is not None else None
            if a is None or c is None:
                return done(bool(margin_lb(*P) > 0), 0 if s == 0 else 1)
            R.lo[s, j], R.hi[s, j] = a, -c
    if event_aware:                                              # one Jacobi pass with the event rows
        P = R.program(R.nb, True)
        lo, hi = R.lo.copy(), R.hi.copy()
        for s in range(R.nb):
            for j in range(H):
                if R.lo[s, j] >= 0 or R.hi[s, j] <= 0 or R.gates[k + s][j] >= 0:
                    continue
                info["lps"] += 2
                a = bound(R.Z[s][j], R.EZ[s][j], *P)
                c = bound(-R.Z[s][j], R.EZ[s][j], *P) if a is not None else None
                if a is None or c is None:
                    return done(bool(margin_lb(*P) > 0), 2)
                lo[s, j] = max(lo[s, j], a); hi[s, j] = min(hi[s, j], -c)
        R.lo, R.hi = lo, hi
    P = R.program(R.nb, False)                                   # final: min max_j (L_j - L_p3) > 0
    info["lps"] += 1
    m = margin_lb(*P, extra=R.ev, eextra=R.eev)          # rows tau + (L_p3 - L_j) >= 0
    info["margin_lb"] = float(m)
    return done(bool(m > 0), 3 if event_aware else 1)


def verify_leaves(tri, leaves, k, event_aware=True, delta=DELTA):
    """A stage-3 (split) prune holds iff every leaf (gates with split rows) is verified."""
    return all(verify_prune(tri, g, k, event_aware=event_aware, delta=delta) for g in leaves)


def verify_certfile(path, return_times=False):
    """Verify every cell of a T398 certificate file (format v1: tri, k, stage (N,), start (N,), leaves, delta).
    delta is used as the exact rational value of its float64. Returns the boolean array (and seconds per cell)."""
    import time
    c = np.load(path, allow_pickle=True)
    tri = tuple(int(v) for v in c["tri"]); k = int(c["k"]); delta = float(c["delta"])
    st, start, leaves = c["stage"], c["start"], c["leaves"]
    ends = list(start[1:]) + [len(leaves)]
    ok = np.zeros(len(st), bool); ts = np.zeros(len(st))
    for i in range(len(st)):
        t = time.time()
        L = leaves[start[i]:ends[i]].astype(int)
        ok[i] = verify_leaves(tri, L, k, event_aware=st[i] >= 2, delta=delta)
        ts[i] = time.time() - t
    return (ok, ts) if return_times else ok
