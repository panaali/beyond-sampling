"""minimal_pair leaderboard entry: sign-stratified, radially integrated tie-face quadrature (deterministic).

est(model, budget, rng) -> (estimate, cost). rng is ignored: nodes are fixed scrambled-Sobol rules (seeds fixed in
advance, identical for every model). No random model inputs.

Identity (sci_tao co-area, exact): P(wrong) = sum_kind Z_kind E_face,kind[F].
Changes relative to sci_tao / ht_seeds (see README):
 1. Radial integration (hom.py): the model has no biases, so the error set is a cone and the radius of the face point
    integrates out in closed form: F_hom(w) = int 1[err(w + s d)] Student-t_{T-1}(s sqrt(T-1)) ds sqrt(T-1).
 2. Sign strata (model-free): the face law is split by the sign pattern of the face point,
      R23: S0 = {u<0, o1<0} (all inputs negative), S1 = {u<0, o1>0} (one positive), S2 = {u>0};
      R12: S0 = {u<0} (all negative), S2 = {u>0}.
    The <=1-positive strata carry most of the new T=5 error variance (README, D1c/D2).
 3. Deterministic two-stage allocation: pilot = equal power-of-2 Sobol prefixes per stratum until >= PILOT of the
    budget is spent, then Neyman (n_h ~ M_h sd_h / sqrt(c_h)) for the rest. All pilot work is charged and reused.
 4. Cost as implemented (walk.py): the unchanged prefix of the sequence is updated linearly, only steps from the
    first changed gate are recomputed; O(H) updates and root tests are charged at their FLOP share.
"""
import os
import sys

import numpy as np
from scipy.special import ndtr, ndtri

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C  # noqa: E402
import hom  # noqa: E402
import walk as W  # noqa: E402

DEDUCTIVE = True
FIXED_COST = False
SUPPORTS = [(16, T) for T in (3, 4, 5, 6, 8, 10)]
PRECOMPUTE_FLOPS = 0          # face CDF tables and Sobol points: model-free, < 1e8 FLOPs, not charged
RULE_SEED = 31_415
PILOT = 0.25


def strata(T):
    out = [("23", 0), ("23", 1), ("23", 2), ("12", 0), ("12", 2)] if T >= 3 else [("12", 0), ("12", 2)]
    return out


def _q0(T, kind):
    tab = C.tables(T)[0][kind][1]
    return float(np.interp(0.0, C._g, tab))


def face_points_strat(T, kind, h, U):
    """Face points of stratum h from uniform U (n, T). Returns y, a, b, tau, wt; the stratum integral is
    M_h * mean(wt * F) with M_h = Z_kind * (q0 if h<2 else 1-q0)."""
    tab = C.tables(T)[0][kind][1]
    cfg = C.configs(T, kind)
    n = len(U); rows = np.arange(n)
    ci = np.minimum((U[:, 0] * len(cfg)).astype(int), len(cfg) - 1)
    a, b, p1 = cfg[ci, 0], cfg[ci, 1], cfg[ci, 2]
    q0 = _q0(T, kind)
    q = U[:, 1] * q0 if h < 2 else q0 + U[:, 1] * (1 - q0)
    u = np.interp(q, tab, C._g)
    u = np.minimum(u, -1e-12) if h < 2 else np.maximum(u, 1e-12)
    Pu = ndtr(u)
    y = np.empty((n, T)); y[rows, a] = u; y[rows, b] = u
    wt = np.ones(n); col = 2
    if kind == "23":
        c = np.clip((0.5 - Pu) / (1 - Pu), 0, 1) if h < 2 else None
        if h == 0:
            v = U[:, col] * c; wt = c
        elif h == 1:
            v = c + U[:, col] * (1 - c); wt = 1 - c
        else:
            v = U[:, col]
        o1 = ndtri(np.clip(Pu + v * (1 - Pu), 1e-300, 1 - 1e-16)); col += 1
        y[rows, p1] = o1
    nlow = T - 3 if kind == "23" else T - 2
    o2 = np.full(n, -np.inf)
    if nlow > 0:
        low = ndtri(np.clip(U[:, col:col + nlow] * Pu[:, None], 1e-300, 1 - 1e-16))
        o2 = low.max(1)
        mask = np.ones((n, T), bool); mask[rows, a] = False; mask[rows, b] = False
        if kind == "23":
            mask[rows, p1] = False
        idx = np.nonzero(mask)[1].reshape(n, nlow)
        y[rows[:, None], idx] = low
    if kind == "23":
        tau = np.minimum(C.SQ2 * (o1 - u) / 3, C.SQ2 * (u - o2))
    else:
        tau = C.SQ2 * (u - o2) / 3
    return y, a, b, tau, wt


def stratum_mass(T, kind, h):
    _, Z23, Z12 = C.tables(T)
    Z = Z23 if kind == "23" else Z12
    q0 = _q0(T, kind)
    return Z * (q0 if h < 2 else 1 - q0)


def run_points(model, T, kind, h, U, radial=True, chunk=4096):
    vals, costs = [], []
    for i in range(0, len(U), chunk):
        y, a, b, tau, wt = face_points_strat(T, kind, h, U[i:i + chunk])
        if radial:
            r = np.linalg.norm(y, axis=1)
            F, c, _ = W.walk(model, y / r[:, None], a, b, np.minimum(tau / r, 1e8), kind, cdf=hom.tcdf(T))
        else:
            F, c, _ = W.walk(model, y, a, b, np.minimum(tau, 8.5), kind)
        vals.append(wt * F); costs.append(c)
    return np.concatenate(vals), np.concatenate(costs)


def est(model, budget, rng=None, rule_seed=RULE_SEED, radial=True, stratify=True, pilot=PILOT, details=False):
    model = tuple(np.asarray(w, dtype=np.float64) for w in model)
    T = model[2].shape[0]
    if stratify:
        S = strata(T)
    else:          # ablation: one stratum per kind, original face law
        S = [("23", -1), ("12", -1)] if T >= 3 else [("12", -1)]
    nmax = 1 << int(np.ceil(np.log2(max(budget, 64) / 2 + 64)))
    rules = {}
    def pts(j, lo, hi):
        if j not in rules:
            rules[j] = C.sobol(T, nmax, rule_seed + 101 * j)
        return rules[j][lo:hi]
    M = {}
    for j, (kind, h) in enumerate(S):
        M[j] = stratum_mass(T, kind, h) if h >= 0 else (C.tables(T)[1] if kind == "23" else C.tables(T)[2])
    data = {j: [np.zeros(0), np.zeros(0)] for j in range(len(S))}
    used = 0.0
    def add(j, n_new):
        nonlocal used
        kind, h = S[j]; n_have = len(data[j][0])
        n_new = min(n_new, nmax - n_have)
        if n_new <= 0:
            return
        U = pts(j, n_have, n_have + n_new)
        if h >= 0:
            v, c = run_points(model, T, kind, h, U, radial)
        else:
            v, c = _run_plain(model, T, kind, U, radial)
        data[j][0] = np.r_[data[j][0], v]; data[j][1] = np.r_[data[j][1], c]; used += c.sum()
    n0 = 16
    while True:
        for j in range(len(S)):
            add(j, n0 - len(data[j][0]))
        if used >= pilot * budget or 2 * n0 > nmax:
            break
        n0 *= 2
    rem = budget - used
    if rem > 0:
        sc = {j: M[j] * data[j][0].std() / np.sqrt(data[j][1].mean()) for j in data}
        tot = sum(sc.values())
        for j in data:
            share = rem * (sc[j] / tot if tot > 0 else 1 / len(data))
            add(j, int(share / data[j][1].mean()))
    est_err = sum(M[j] * data[j][0].mean() for j in data)
    if details:
        return float(1 - est_err), float(used), {S[j]: (M[j] * data[j][0].mean(), len(data[j][0])) for j in data}
    return float(1 - est_err), float(used)


def _run_plain(model, T, kind, U, radial, chunk=4096):
    vals, costs = [], []
    for i in range(0, len(U), chunk):
        y, a, b, tau = C.face_points(T, kind, U[i:i + chunk], cap=np.inf if radial else 8.5)
        if radial:
            r = np.linalg.norm(y, axis=1)
            F, c, _ = W.walk(model, y / r[:, None], a, b, np.minimum(tau / r, 1e8), kind, cdf=hom.tcdf(T))
        else:
            F, c, _ = W.walk(model, y, a, b, tau, kind)
        vals.append(F); costs.append(c)
    return np.concatenate(vals), np.concatenate(costs)
