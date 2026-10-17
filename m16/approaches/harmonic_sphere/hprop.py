"""Harmonic deduction-projection for the bias-free 2nd-argmax RNN (T38).

The model has no biases, so correctness depends on x/|x| only; accuracy = sigma(correct set) on S^{T-1}.
State = list of branches (exact polyhedral cone K_b, harmonic density g_b of degree <= D, linear map A_b with
h_t = A_b x on K_b, order cell (i, p) = (argmax, 2nd argmax of x)).

  root    : for each of the T(T-1) order cells, g = proj_D(... proj_D(1 * 1[H_1]) ... 1[H_{T-1}])   (model-free)
  step t  : preactivation forms W = Wh A + Wi e_t^T; for each neuron j, the exact cone decides whether
            {W_j x >= 0} contains / misses / splits K_b; only a split cuts g:  g+ = proj_D(g 1[W_j x>=0]),
            g- = g - g+  (mass is conserved exactly); A <- diag(gates) W
  readout : logits = Wo A x; correct on K_b iff logit p >= logit q for all q (halfspaces); the split ones are
            cut in sequence and the last one integrated by Funk-Hecke.
Estimate = sum over leaves of the mass of g_b on the correct side.  No random numbers anywhere.

Variants: cells="first" (order cells at the root, default) or "last" (order constraints applied at readout);
prune=tau drops the cut when |mass| of the branch < tau (the branch then keeps its majority-side gate, i.e.
the gate is decided at the cone's interior point instead of being split).
"""
import numpy as np
from cones import Cone
import sphere as S

_SPACES = {}
_ROOTS = {}


def space(n, D):
    if (n, D) not in _SPACES:
        _SPACES[n, D] = S.Space(n, D)
    return _SPACES[n, D]


def order_cells(T):
    cells = []
    for i in range(T):
        for p in range(T):
            if p == i:
                continue
            rows = [np.eye(T)[i] - np.eye(T)[p]] + [np.eye(T)[p] - np.eye(T)[k] for k in range(T) if k not in (i, p)]
            cells.append(((i, p), rows))
    return cells


def roots(T, D):
    """Model-free root branches (cone, coeffs) per order cell. Cached; cost is PRECOMPUTE."""
    if (T, D) in _ROOTS:
        return _ROOTS[T, D]
    sp = space(T, D); f0 = sp.flops
    out = []
    for cell, rows in order_cells(T):
        K = Cone(T); c = sp.one()
        for a in rows:
            c = sp.cut(c[:, None], (a / np.linalg.norm(a))[None])[:, 0]
            K.add(a)
        out.append((cell, K, c))
    sp.flops = f0                                  # model-free: not charged
    _ROOTS[T, D] = out
    return out


def estimate(model, D, cells="first", prune=0.0, return_info=False):
    Wi, Wh, Wo = model
    H, T = Wh.shape[0], Wo.shape[0]
    sp = space(T, D); sp.flops = 0.0
    lp_flops = 0.0
    if cells == "first":
        br = [(cell, K.copy(), c.copy(), np.zeros((H, T))) for cell, K, c in roots(T, D)]
    else:                                          # "last" or "global": no root split
        br = [(None, Cone(T), sp.one(), np.zeros((H, T)))]
    ncut = 0
    for t in range(T):
        Wt = [Wh @ A + np.outer(Wi, np.eye(T)[t]) for (_, _, _, A) in br]
        live = [(b, K, c, np.zeros(H, bool)) for b, (cell, K, c, A) in enumerate(br)]
        for j in range(H):
            todo, nxt = [], []
            for (b, K, c, g) in live:
                a = Wt[b][j]
                if not np.any(a):
                    nxt.append((b, K, c, g)); continue
                cl = K.classify(a)
                if cl == 0 and prune > 0 and abs(c[0]) < prune:
                    cl = 1 if a @ K.interior_point() >= 0 else -1
                if cl != 0:
                    g = g.copy(); g[j] = cl > 0; nxt.append((b, K, c, g)); continue
                todo.append((b, K, c, g, a))
            if todo:
                C = np.stack([x[2] for x in todo], 1)
                V = np.stack([x[4] / np.linalg.norm(x[4]) for x in todo])
                Cp = sp.cut(C, V); ncut += len(todo)
                for k, (b, K, c, g, a) in enumerate(todo):
                    Kp, Km = K.copy(), K.copy(); Kp.add(a); Km.add(-a)
                    lp_flops += Kp.flops + Km.flops + K.flops; K.flops = 0
                    gp, gm = g.copy(), g.copy(); gp[j] = True
                    nxt.append((b, Kp, Cp[:, k], gp)); nxt.append((b, Km, c - Cp[:, k], gm))
            live = nxt
        for (b, K, c, g) in live:
            lp_flops += K.flops; K.flops = 0
        br = [(br[b][0], K, c, g[:, None] * Wt[b]) for (b, K, c, g) in live]
    if cells == "global":
        acc = readout_global(sp, br, Wo, T); ncut += _NC[0]
        flops = sp.flops + lp_flops + len(br) * 2 * T * T * H
        info = dict(flops=flops, harm_flops=sp.flops, lp_flops=lp_flops, ncut=ncut, nleaf=len(br), m=sp.m, N=sp.N,
                    hilton_flops=ncut * sp.m * D * D + lp_flops)
        return (acc, info) if return_info else (acc, flops)
    # readout
    acc = 0.0; nleaf = len(br); pend = []
    for (cell, K, c, A) in br:
        L = Wo @ A
        if cells == "first":
            items = [(cell, K, c)]
        else:
            items = []
            for cl2, rows in order_cells(T):
                K2, c2, ok = K.copy(), c, True
                for a in rows:
                    s = K2.classify(a)
                    if s == -1:
                        ok = False; break
                    if s == 0:
                        c2 = sp.cut(c2[:, None], (a / np.linalg.norm(a))[None])[:, 0]; ncut += 1; K2.add(a)
                lp_flops += K2.flops
                if ok:
                    items.append((cl2, K2, c2))
        for (cl2, K2, c2) in items:
            p = cl2[1]
            rows, ok = [], True
            for q in range(T):
                if q == p:
                    continue
                a = L[p] - L[q]
                s = K2.classify(a)
                if s == -1:
                    ok = False; break
                if s == 0:
                    rows.append(a)
            lp_flops += K2.flops; K2.flops = 0
            if not ok:
                continue
            dead = False
            while rows:
                a = rows.pop(0)
                s = K2.classify(a)
                if s == -1:
                    dead = True; break
                if s == 1:
                    continue
                if not rows:
                    pend.append((c2, a / np.linalg.norm(a))); dead = True; break
                c2 = sp.cut(c2[:, None], (a / np.linalg.norm(a))[None])[:, 0]; ncut += 1
                K2.add(a)
            lp_flops += K2.flops
            if not dead:
                acc += c2[0]
    if pend:
        C = np.stack([x[0] for x in pend], 1); V = np.stack([x[1] for x in pend])
        acc += float(sp.mass_in(C, V).sum())
    flops = sp.flops + lp_flops + nleaf * 2 * T * T * H
    if return_info:
        return acc, dict(flops=flops, harm_flops=sp.flops, lp_flops=lp_flops, ncut=ncut, nleaf=nleaf,
                         m=sp.m, N=sp.N, hilton_flops=ncut * sp.m * D * D + lp_flops)
    return acc, flops


_NC = [0]


def readout_global(sp, br, Wo, T):
    """sum_b <g_b, f_b>, f_b(x) = 1[argmax Wo A_b x = 2nd argmax x] on the WHOLE sphere (no cone pruning), so the
    Gibbs leakage of g_b outside its cone is scored with the leaf's own (continuous) logits and the true target.
    Tree of cuts, batched over leaves: per p, T-1 logit cuts; per (p, i), T-1 order cuts (last by Funk-Hecke)."""
    E = np.eye(T); _NC[0] = 0
    C0 = np.stack([c for (_, _, c, _) in br], 1)
    Ls = [Wo @ A for (_, _, _, A) in br]
    B = C0.shape[1]; acc = 0.0
    nrm = lambda V: V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-300)
    for p in range(T):
        C = C0
        for q in range(T):
            if q == p:
                continue
            V = nrm(np.stack([L[p] - L[q] for L in Ls]))
            C = sp.cut(C, V); _NC[0] += B
        for i in range(T):
            if i == p:
                continue
            rows = [E[i] - E[p]] + [E[p] - E[k] for k in range(T) if k not in (i, p)]
            Ci = C
            for a in rows[:-1]:
                Ci = sp.cut(Ci, np.tile(a / np.linalg.norm(a), (B, 1))); _NC[0] += B
            a = rows[-1]
            acc += float(sp.mass_in(Ci, np.tile(a / np.linalg.norm(a), (B, 1))).sum())
    return acc
