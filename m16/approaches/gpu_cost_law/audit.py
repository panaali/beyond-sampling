"""T193 / T196 op-completeness audit of estT (recipe108 and exact) in the T196 unit: FLOPs, multiply-add = 2,
any other scalar op (compare, abs, sqrt, atan2, exp, logical) = 1; memory moves and Python bookkeeping free.

Runs the UNCHANGED estT_base code. Every function that does arithmetic is wrapped: the wrapper charges the complete
FLOPs of that function's own numpy work from the actual shapes (formulas next to each wrapper), and np.linalg calls
are counted where they happen (LAPACK-standard costs below). estT's own Ops.add charges are attributed to the same
function, so the table shows charged vs complete per site. Diagnostic only (recomputations use uncounted originals).

usage: python audit.py H T name mode  -> results/audit/H{H}T{T}_{name}_{mode}.json
"""
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["T192_FAST"] = "0"
sys.path[:0] = [HERE, os.path.join(HERE, "..", "understand_m43"), os.path.join(HERE, "..", "arc_m43_repro"),
                os.path.join(HERE, "..", "..")]
import estT_base as B  # noqa: E402
import m43  # noqa: E402

STACK = ["top"]
FULL = defaultdict(float)      # complete FLOPs per site
CHG = defaultdict(float)       # estT charge per site
CALLS = defaultdict(int)

# ------------------------------------------------------------------ linalg with standard costs (per matrix)
_det, _inv, _norm, _rank, _svd = np.linalg.det, np.linalg.inv, np.linalg.norm, np.linalg.matrix_rank, np.linalg.svd


def _svd_vals(m, n):
    if m < n:
        m, n = n, m
    return 4 * m * n * n - 4 * n ** 3 / 3 + 12 * n * n          # Golub-Van Loan, singular values only


def det(a):
    a = np.asarray(a); n = a.shape[-1]; b = a.size // (n * n) if n else 0
    FULL[STACK[-1] + ":linalg"] += b * (2 * n ** 3 / 3 + n)        # LU + product of the diagonal
    return _det(a)


def inv(a):
    a = np.asarray(a); n = a.shape[-1]; b = a.size // (n * n)
    FULL[STACK[-1] + ":linalg"] += b * 2 * n ** 3
    return _inv(a)


def norm(x, *args, **kw):
    x = np.asarray(x)
    FULL[STACK[-1] + ":linalg"] += 2 * x.size + (x.size // x.shape[-1] if x.ndim > 1 and kw.get("axis") is not None else 1)
    return _norm(x, *args, **kw)


def matrix_rank(M, tol=None, *a, **kw):
    M = np.asarray(M)
    if M.ndim == 2 and M.size:
        m, n = M.shape; FULL[STACK[-1] + ":linalg"] += _svd_vals(m, n) + 2 * min(m, n) + 3
    return _rank(M, tol, *a, **kw)


np.linalg.det, np.linalg.inv, np.linalg.norm, np.linalg.matrix_rank = det, inv, norm, matrix_rank

_add = m43.Ops.add


def _ops_add(self, k):
    CHG[STACK[-1]] += k; _add(self, k)


m43.Ops.add = _ops_add


def wrap(name, extra):
    """extra(args, result) -> complete FLOPs of the function's own non-linalg numpy work."""
    f = getattr(B, name)

    def g(*a, **kw):
        STACK.append(name); CALLS[name] += 1
        try:
            r = f(*a, **kw)
        finally:
            STACK.pop()
        FULL[name] += extra(a, r)
        return r
    setattr(B, name, g)


# ------------------------------------------------------------------ per-function complete counts
def _clip(a, r):
    c, n = a[0], a[1]; m, d = c.R.shape
    s = c.R @ n
    x = 2 * m * d + 1 + 2 * m + 2 * m                           # s = R n, sc, min/max, |s| <= sc
    sc = B.TOL * _norm(n)
    if s.min() >= -sc or s.max() <= sc:
        return 2 * m * d + 1 + 2 * m
    x += 2 * m                                                  # keep masks
    for a_, b_ in PAIRS.get(id(c), []):
        x += 4                                                  # crossing test
        if (s[a_] > sc and s[b_] < -sc) or (s[a_] < -sc and s[b_] > sc):
            x += 2 + 3 * d + d                                  # |s|, |s_a| R_b + |s_b| R_a, / norm
    return x


PAIRS = {}          # id(cone) -> its adjacent pairs (as returned by the last _adjacent_pairs call on it)


def _adj_full(c, ops):
    """estT._adjacent_pairs, same logic, complete counting (Tm & Tm, sums, subset tests with short-circuit)."""
    d = c.R.shape[1]; Tm = c.T; m = len(c.R); out = []
    STACK.append("_adjacent_pairs"); CALLS["_adjacent_pairs"] += 1
    x = 0
    for a in range(m):
        for b in range(a + 1, m):
            Z = Tm[a] & Tm[b]; ops.add(len(c.N)); x += 2 * len(c.N)
            if Z.sum() < d - 2:
                continue
            others = [q for q in range(m) if q != a and q != b]
            ops.add(len(others) * int(Z.sum()))
            zs = int(Z.sum()); hit = False
            for q in others:
                x += zs + 1                                     # gather-and-all over Z (counted as zs logical ops)
                if np.all(Tm[q][Z]):
                    hit = True; break
            if hit:
                continue
            if matrix_rank(c.N[Z]) >= d - 2:
                ops.add(d ** 3); out.append((a, b))
    FULL["_adjacent_pairs"] += x; STACK.pop()
    PAIRS[id(c)] = out
    return out


def _mass_full(c, ops):
    """estT.mass, same logic; complete counting of the triangulation bookkeeping (incidence tests) + orthant calls."""
    d = c.R.shape[1]
    if len(c.R) == d:
        return B.orthant(c.R, ops)
    Tm = c.T
    STACK.append("mass"); CALLS["mass"] += 1

    def faces_of(S, Zset, k):
        out = set()
        for i in range(len(c.N)):
            if i in Zset:
                continue
            S2 = tuple(q for q in S if Tm[q, i]); FULL["mass"] += len(S)
            ops.add(len(S))
            if len(S2) >= k - 1 and S2 not in out:
                Z2 = np.all(Tm[list(S2)], axis=0); FULL["mass"] += len(S2) * len(c.N)
                if d - matrix_rank(c.N[Z2]) == k - 1:
                    out.add(S2)
                ops.add(d ** 3)
        return out

    def tri(S, k):
        if len(S) == k:
            return [S]
        a = S[0]
        Z = set(i for i in range(len(c.N)) if all(Tm[q, i] for q in S)); FULL["mass"] += len(S) * len(c.N)
        res = []
        for F in faces_of(S, Z, k):
            if a in F:
                continue
            for simp in tri(F, k - 1):
                res.append((a,) + simp)
        return res

    simps = tri(tuple(range(len(c.R))), d)
    STACK.pop()
    tot = 0.0
    for simp in simps:
        tot += B.orthant(c.R[list(simp)], ops)
    return tot


def _orthant(a, r):
    R = a[0]; d = len(R)
    x = 2 * d ** 3 + d + 2 * d * d                             # S = B B^T, sqrt diag, outer + divide (inv in linalg)
    if d == 3:
        return x + 3 * 2 + 4
    if d == 4:
        dt = abs(_det(R))
        if dt < B.THIN_DET:
            n = len(B._DU)
            return x + n * (2 * 16 + 4 + 3 + 1 + 2) + 4          # DU R, square, row-sum, q^2, DW/q^2 + sum, scale
        K = len(B._LOG_T)
        x += 4 * (4 + 2 * 4 + 1 + 4 + 4)                          # cofactor sign, n.R_i, sign, /norm (dets in linalg)
        x += 2 * 64 + 6 * (2 + 3 + 3) + 8                          # C = N N^T, atan2 angles, oma, p
        x += 16 + K * 32                                           # Rt = C + s (I - C)
        x += K                                                     # max(det, 0)
        x += 6 * K * (1 + 4 + 2 + 1 + 1 + 1 + 3 + 1)              # sign, om, sqrt(dR om), atan2, max, sqrt, W r apc /, sum
        return x + 3
    raise ValueError(d)


def _ref(a, r):
    c, n_true = a[0], a[1]; m, d = c.R.shape
    on = np.abs(c.R @ n_true) < 1e-9
    k = int(on.sum()) if on.sum() >= d - 1 else m
    return 2 * m * d + 2 * m + k * d + d


def _margin_split(a, r):
    c, A = a[0], a[1]; model, t = a[2], a[3]; exact = a[8]; done = a[9] if len(a) > 9 else ()
    m, d = c.R.shape; H = A.shape[0]
    S = c.R @ A.T
    x = 2 * m * H * d + H * 2                                   # S, sc per gate (norm in linalg)
    cut = 0
    for k in range(H):
        sc = B.TOL * max(_norm(A[k]), 1e-300)
        x += 2 * m
        if not (S[:, k].min() >= -sc or S[:, k].max() <= sc):
            x += 2 * d + 1; cut += 1
        elif S[:, k].min() >= -sc and not S[:, k].max() > sc:
            x += 2 * d + 1
    if exact or not cut:
        return x
    # recipe: per cutting gate, minority rays + edge crossing points, score/flip tests (frozen margin: own wrapper)
    g = np.zeros(H)
    ref = c.R.mean(0)
    pairs = PAIRS.get(id(c), [])
    for k in range(H):
        sc = B.TOL * max(_norm(A[k]), 1e-300)
        if S[:, k].min() >= -sc or S[:, k].max() <= sc or k in done:
            continue
        sk = S[:, k]
        nP = m                                                  # upper bound on minority rays (mask + |s| w each)
        x += 3 * m
        for a_, b_ in pairs:
            x += 3
            if (sk[a_] > 0) != (sk[b_] > 0) and sk[a_] != 0 and sk[b_] != 0:
                x += 2 + 3 * d + d; nP += 1
        x += 4 * nP + 4                                         # sh / (|mm| + eps), max, min/max/compare (flips)
    return x


def _frozen_margin(a, r):
    model, t, A, g, truth, P = a[0], a[1], a[2], a[3], a[4], a[5]
    Wi, Wh, Wo = model; T = Wo.shape[0]; H = len(Wi); n = len(P)
    x = 2 * n * H * T + n * H                                   # P A^T, * g
    x += (T - 1 - t) * n * (2 * H * H + H + H + H)              # h Wh^T, x Wi, +, relu
    x += 2 * n * H * T + n * T + n                              # readout, max over others, subtract
    return x


def _weights_to_margin(a, r):
    model, t = a[0], a[1]; Wi, Wh, Wo = model; T = Wo.shape[0]; H = len(Wi)
    return (T - 1 - t) * (H * H + 2 * H ** 3) + 2 * T * H + 2 * T * H * H + T * H


def _finish(a, r):
    c, g, A, t, model = a[0], a[1], a[2], a[3], a[4]; H = len(g); T = model[2].shape[0]
    return H * A.shape[1] + 2 * H * H * A.shape[1] + 2 * H * T  # g*A, Wh (g A), outer + add


def _cell_error(a, r):
    c, g, A, Wo = a[0], a[1], a[2], a[3]; H = len(g); T = Wo.shape[0]
    if g.sum() == 0:
        return 1
    return H * T + 2 * T * H * T + (T - 1) * T                  # g*A, Wo (g A), L_truth - L_o


B._adjacent_pairs = _adj_full
B.mass = _mass_full
wrap("clip", _clip)
wrap("orthant", _orthant)
wrap("_ref", _ref)
wrap("_margin_split", _margin_split)
wrap("_frozen_margin", _frozen_margin)
wrap("_weights_to_margin", _weights_to_margin)
wrap("_finish", _finish)
wrap("_cell_error", _cell_error)


def main(H, T, name, mode):
    from zoo44 import zoo
    if ":fresh" in name:
        from leaderboard import _noisy
        s_, k_ = name.split(":fresh"); model = _noisy(zoo(H, T, int(s_)), 1, 0.005, int(k_))[0]
    else:
        model = zoo(H, T, int(name))
    o = m43.Ops(); t0 = time.time(); acc = B.estT(model, o, mode=mode)
    sites = sorted(set(k.split(":")[0] for k in FULL) | set(CHG))
    tab = {s: dict(charged=CHG.get(s, 0.0), full=FULL.get(s, 0.0) + FULL.get(s + ":linalg", 0.0),
                   linalg=FULL.get(s + ":linalg", 0.0), calls=CALLS.get(s, 0)) for s in sites}
    full = sum(v["full"] for v in tab.values())
    r = dict(H=H, T=T, model=name, mode=mode, acc=acc, ops_charged=o.n, flops_complete=full,
             ratio_complete_over_charged=full / o.n, sec=time.time() - t0, sites=tab)
    os.makedirs(os.path.join(HERE, "results", "audit"), exist_ok=True)
    json.dump(r, open(os.path.join(HERE, "results", "audit", f"H{H}T{T}_{name.replace(':', '_')}_{mode}.json"), "w"),
              indent=1)
    return r


if __name__ == "__main__":
    r = main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
    print(r["model"], r["mode"], "acc %.12f charged %.4g complete %.4g ratio %.2f" % (
        r["acc"], r["ops_charged"], r["flops_complete"], r["ratio_complete_over_charged"]))
    for s, v in sorted(r["sites"].items(), key=lambda kv: -kv[1]["full"]):
        print("  %-20s charged %.3g complete %.3g (linalg %.3g) calls %d" % (s, v["charged"], v["full"], v["linalg"], v["calls"]))
