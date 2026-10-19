"""T400 part (b): rigorous verification that a prefix cell carries no event point (README, pre-registration).

verify_prune(tri, gates, k) -> (pruned: bool, info)
  Cell = (tri, first k rows of gates). Proves, if it can, that
     cell ∩ {x in [-1,1]^10, x_p2 - x_p3 >= DELTA} ∩ {argmax of the step-10 logits = p3 (closed)} = ∅
  by rebuilding T390's triangle relaxation (relax.py) step by step. Every neuron bound l <= z <= u is a
  Neumaier–Shcherbina safe dual bound (float LP duals from HiGHS; residual absorbed by the variable box; all
  data and rounding errors bounded explicitly). The final step proves min over the relaxation of
  max_j (L_j - L_p3) > 0 the same way. A float LP never decides anything by itself.

Error model (IEEE double, round to nearest, u = 2^-53): for a dot product of length n computed in float,
|fl - exact| <= gamma_n * sum |terms|, gamma_n = n u / (1 - n u). Every error bound below is computed in float and
then multiplied by SAFE = 2 to cover the rounding of the bound computation itself (its relative error is
<= gamma_N << 1)."""
import os
import sys

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import common  # noqa: E402

Wi, Wh, Wo = common.Wi, common.Wh, common.Wo
H, D = 16, 10
DELTA = 0.1          # the float 0.1 (exact rational 0.1000000000000000055...); README part (c)
U = 2.0 ** -53
SAFE = 2.0


def gam(n):
    return n * U / (1 - n * U)


def cell_rows(gates, k):
    """Float cell rows G (G x >= 0 for the true cell) with elementwise error bounds EG, and h_k = A x with
    error bound EA (|A_true - A| <= EA), for the first k steps of gates."""
    A = np.zeros((H, D)); EA = np.zeros((H, D)); G = []; EG = []
    aWh = np.abs(Wh)
    for t in range(k):
        Z = Wh @ A; Z[:, t] += Wi
        EZ = aWh @ EA + gam(H + 1) * (aWh @ np.abs(A) + np.abs(Z))
        EZ *= SAFE
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
    """Rigorous lower bound on min c.v s.t. A v >= b (true data within the error bounds), lo <= v <= hi,
    for any y >= 0 (float duals). c, A, b are float; Ec, EA, Eb elementwise bounds on |true - float|."""
    y = np.maximum(y, 0.0)
    m, n = A.shape
    r = c - A.T @ y
    Er = Ec + EA.T @ y + gam(m + 1) * (np.abs(c) + np.abs(A).T @ y)
    yb = y @ b
    Eyb = y @ Eb + gam(m) * (y @ np.abs(b))
    tl = np.minimum(r * lo, r * hi)
    M = np.maximum(np.abs(lo), np.abs(hi))
    s = tl.sum()
    err = Eyb + Er @ M + gam(n + 2) * (np.abs(tl).sum() + abs(yb)) + gam(n) * (Er @ M)
    return yb + s - SAFE * err


class Relax:
    """The relaxation as rows over v = (x, h_{k+1}, ..., h_s) with A v >= b, plus box lo <= v <= hi."""

    def __init__(self, tri, gates, k):
        self.tri, self.k = tri, k
        G, EG, A, EA = cell_rows(gates, k)
        p2, p3 = tri[1], tri[2]
        self.rows = [G]; self.err = [EG]; self.b = [np.zeros(len(G))]; self.eb = [np.zeros(len(G))]
        r = np.zeros(D); r[p2], r[p3] = 1, -1
        self.rows.append(r[None]); self.err.append(np.zeros((1, D))); self.b.append(np.array([DELTA])); self.eb.append(np.zeros(1))
        self.lo = [-np.ones(D)]; self.hi = [np.ones(D)]
        self.n = D
        self.prevA, self.prevE = A, EA      # h_prev = prevA v (cols over current v), with error prevE

    def mat(self):
        n = self.n
        pad = lambda M: np.hstack([M, np.zeros((len(M), n - M.shape[1]))])  # noqa: E731
        return (np.vstack([pad(M) for M in self.rows]), np.vstack([pad(M) for M in self.err]),
                np.concatenate(self.b), np.concatenate(self.eb), np.concatenate(self.lo), np.concatenate(self.hi))

    def z_rows(self, t):
        """Pre-activation rows of step t (0-based input index) over v, with error bounds."""
        Z = Wh @ self.prevA
        EZ = np.abs(Wh) @ self.prevE + gam(H + 1) * (np.abs(Wh) @ np.abs(self.prevA) + np.abs(Z))
        Z[:, t] += Wi
        return Z, SAFE * EZ


def _lp(c, A, b, lo, hi):
    return linprog(c, A_ub=-A, b_ub=-b, bounds=list(zip(lo, hi)), method="highs")


def bound(c, Ec, A, EA, b, Eb, lo, hi):
    """Safe lower bound of min c.v; None if the float LP is infeasible (caller then tries phase 1)."""
    r = _lp(c, A, b, lo, hi)
    if r.status == 2:
        return None
    if r.status != 0:
        return -np.inf
    return safe_lb(c, Ec, A, EA, b, Eb, lo, hi, -r.ineqlin.marginals)


def phase1_empty(A, EA, b, Eb, lo, hi, T=1e3):
    """Prove {A v >= b} ∩ box = ∅: min tau s.t. A v + tau >= b, tau in [-T, T] has a positive safe bound."""
    m, n = A.shape
    A1 = np.hstack([A, np.ones((m, 1))]); E1 = np.hstack([EA, np.zeros((m, 1))])
    c = np.zeros(n + 1); c[-1] = 1
    lo1, hi1 = np.r_[lo, -T], np.r_[hi, T]
    r = _lp(c, A1, b, lo1, hi1)
    if r.status != 0:
        return False
    return safe_lb(c, np.zeros(n + 1), A1, E1, b, Eb, lo1, hi1, -r.ineqlin.marginals) > 0


def verify_prune(tri, gates, k, return_info=False):
    gates = np.asarray(gates)
    p3 = tri[2]
    R = Relax(tri, gates, k)
    info = dict(stage=None, lps=0)

    def done(ok, stage):
        info["stage"] = stage
        return (ok, info) if return_info else ok

    A, EA, b, Eb, lo, hi = R.mat()
    if phase1_empty(A, EA, b, Eb, lo, hi):
        return done(True, 0)
    for s in range(D - k):
        t = k + s
        Z, EZ = R.z_rows(t)
        A, EA, b, Eb, lo, hi = R.mat()
        l = np.empty(H); u = np.empty(H)
        for j in range(H):
            info["lps"] += 2
            a = bound(Z[j], EZ[j], A, EA, b, Eb, lo, hi)
            if a is None:
                return done(phase1_empty(A, EA, b, Eb, lo, hi), s + 1)
            c = bound(-Z[j], EZ[j], A, EA, b, Eb, lo, hi)
            if c is None:
                return done(phase1_empty(A, EA, b, Eb, lo, hi), s + 1)
            l[j], u[j] = a, -c
        # append h_t variables
        n0 = R.n; R.n += H
        newlo = np.zeros(H); newhi = np.zeros(H)
        rows, errs, bs, ebs = [], [], [], []
        for j in range(H):
            zr = np.r_[Z[j], np.zeros(H)]; ze = np.r_[EZ[j], np.zeros(H)]
            e = np.zeros(n0 + H); e[n0 + j] = 1
            if u[j] <= 0:
                continue                                   # h = 0 (box [0, 0])
            if l[j] >= 0:                                  # h = z: two rows
                newlo[j], newhi[j] = l[j], u[j]
                rows += [e - zr, zr - e]; errs += [ze, ze]; bs += [0, 0]; ebs += [0, 0]
            else:                                          # triangle: h >= z, h <= alpha z + beta
                newhi[j] = u[j]
                alpha = u[j] / (u[j] - l[j]); beta = -alpha * l[j]
                # validity of the chord: alpha l + beta >= 0 and alpha u + beta >= u, made safe by raising beta
                beta = beta + SAFE * gam(4) * (abs(alpha * l[j]) + abs(alpha * u[j]) + abs(beta) + abs(u[j])) + 4 * U * abs(u[j])
                rows += [e - zr, alpha * zr - e]
                errs += [ze, alpha * ze + gam(2) * (np.abs(alpha * zr))]
                bs += [0, -beta]; ebs += [0, 0]
        R.rows = [np.hstack([M, np.zeros((len(M), H))]) for M in R.rows]
        R.err = [np.hstack([M, np.zeros((len(M), H))]) for M in R.err]
        if rows:
            R.rows.append(np.array(rows)); R.err.append(np.array(errs)); R.b.append(np.array(bs, float)); R.eb.append(np.array(ebs, float))
        R.lo.append(newlo); R.hi.append(newhi)
        R.prevA = np.zeros((H, R.n)); R.prevA[:, n0:] = np.eye(H); R.prevE = np.zeros((H, R.n))
    # final: min tau s.t. L_j - L_p3 <= tau for all j != p3, i.e. tau - (L_j - L_p3) >= 0
    A, EA, b, Eb, lo, hi = R.mat()
    L = Wo @ R.prevA
    m, n = A.shape
    T = 1e3
    ev = np.array([np.r_[L[p3] - L[j], 1.0] for j in range(D) if j != p3])
    eE = np.array([np.r_[gam(H) * (np.abs(Wo[p3]) + np.abs(Wo[j])) @ np.abs(R.prevA) * SAFE + 2 * U * np.abs(L[p3] - L[j]), 0.0]
                   for j in range(D) if j != p3])
    A1 = np.vstack([np.hstack([A, np.zeros((m, 1))]), ev]); E1 = np.vstack([np.hstack([EA, np.zeros((m, 1))]), eE])
    b1 = np.r_[b, np.zeros(len(ev))]; Eb1 = np.r_[Eb, np.zeros(len(ev))]
    lo1, hi1 = np.r_[lo, -T], np.r_[hi, T]
    c = np.zeros(n + 1); c[-1] = 1
    info["lps"] += 1
    r = _lp(c, A1, b1, lo1, hi1)
    if r.status == 2:
        return done(phase1_empty(A, EA, b, Eb, lo, hi), D - k + 1)
    if r.status != 0:
        return done(False, -1)
    lb = safe_lb(c, np.zeros(n + 1), A1, E1, b1, Eb1, lo1, hi1, -r.ineqlin.marginals)
    info["margin_lb"] = float(lb); info["margin_float"] = float(r.fun)
    return done(bool(lb > 0), D - k + 1)
