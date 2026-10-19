"""T46 hybrid: exact enumeration of the gate cells above a fixed mass threshold + the frozen T10 walk quadrature
on the rest.

    P(wrong) = P(wrong, x in E) + P(wrong, x not in E),   E = union of the full (T-step) gate cells of mass >= tau.

Part 1 (deductive, no inputs): a depth-first enumeration of gate cells (cones.lift/clip, as in iter_cells) that
measures every visited prefix cell exactly (pulling triangulation + cone_measure) and prunes a prefix cell of mass
< tau (its descendants are smaller). Each full cell of mass >= tau is split into outcome pieces and its wrong pieces
are measured exactly (as in ladder.exact_acc). FLOPs counted in cones.Ops, as implemented.
Part 2 (deterministic QMC-class quadrature): the frozen T10 estimator (frozen_t10/, sha256 asserted), unchanged, run
with the remaining budget, its walk replaced by mwalk.walk, which gives no mass to pieces whose gate pattern is in E.
rng is ignored everywhere.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cones as Cn  # noqa: E402
import frozen_check  # noqa: E402
import mwalk  # noqa: E402

FE = frozen_check.E
FROZEN_WALK = frozen_check.W.walk


def _fpe_flops(H, T):
    f = lambda H, T: T * (2 * H * H + 2 * H) + 2 * H * T
    return 6e3 * f(H, T) / f(16, 10)


def cell_mass(R, Z, ops, rtol=1e-6):
    """Gaussian measure (in R^d) of the pointed cone with rays R, tight sets Z, to relative accuracy rtol.
    rtol=None: one Grundmann-Moller degree-3 rule per simplex, no refinement (a cheap deterministic proxy; it only
    decides which cells go into E, so it does not affect the exactness of the decomposition)."""
    d = R.shape[1]
    if d == 1:
        return 0.5, 0.0
    tri = Cn.triangulate(R, Z, ops)
    V = np.array([R[list(t)] for t in tri])
    if rtol is None:
        p, e = Cn.cone_measure(V, ops, rtol=np.inf, maxdepth=0, plackett=False, s=1)
    else:
        p, e = Cn.cone_measure(V, ops, rtol=rtol)
    return float(p.sum()), float(e.sum())


def _children_g(R, Z, A, allm, t, model, ops):
    """cones._children, also returning the step-(t+1) gate vector of each child (same arithmetic, same ops)."""
    Wi, Wh, Wo = model
    H = len(Wi)
    order = np.argsort(-np.abs(Wi))
    B = Wh @ A[:, :t]; ops.add(2 * H * H * t, "lift")
    cid = lambda k: 1 + 16 * (t - 1) + k
    k0 = order[0]
    pieces = []
    for sg in (1.0, -1.0):
        R1, Z1 = Cn.lift(R, Z, sg * B[k0], sg * Wi[k0], cid(k0), allm, ops)
        g = np.zeros(H, bool); g[k0] = sg > 0
        pieces.append((R1, Z1, g))
    for k in order[1:]:
        a = np.concatenate([B[k], [Wi[k]]])
        new = []
        for R1, Z1, g in pieces:
            ins, outs = Cn.clip(R1, Z1, a, cid(k), ops)
            if ins is not None:
                g2 = g.copy(); g2[k] = True; new.append((ins[0], ins[1], g2))
            if outs is not None:
                new.append((outs[0], outs[1], g))
        pieces = new
    allm2 = allm.copy()
    for k in range(H):
        allm2 |= Cn.bit(cid(k))
    out = []
    for R1, Z1, g in pieces:
        A2 = A.copy(); A2[:, :t] = B; A2[:, t] = Wi; A2[~g] = 0.0
        out.append(((R1, Z1, A2, allm2), g))
    return out


def exact_part(model, tau, ops=None, rtol_prune=None, rtol=1e-3):
    """Exact wrong mass inside E(tau). Returns dict(wrong, mass, err, keys, counts, flops, fpe)."""
    ops = Cn.Ops() if ops is None else ops
    ops_p = Cn.Ops()                # pruning masses, kept separate for the cost breakdown; charged
    Wi, Wh, Wo = model
    H = len(Wi); T = Wo.shape[0]
    Cn.ZERO_MARGINS[0] = 0
    base = 1 + 16 * (T - 1)
    stack = []
    for s in (-1.0, 1.0):
        g = (s * Wi) > 0
        A = np.zeros((H, T)); A[:, 0] = np.where(g, Wi, 0.0)
        stack.append(((np.array([[s]]), np.zeros((1, Cn.NW), np.uint64), A, Cn.bit(0)), 1, [g]))
    keys = set(); wrong = 0.0; massE = 0.0; err = 0.0
    visited = [0] * (T + 1); kept = [0] * (T + 1)
    while stack:
        c, t, gs = stack.pop()
        visited[t] += 1
        m, _ = cell_mass(c[0], c[1], ops_p, rtol_prune)
        if m < tau:
            continue
        kept[t] += 1
        if t < T:
            for ch, g in _children_g(*c, t, model, ops)[::-1]:
                stack.append((ch, t + 1, gs + [g]))
            continue
        keys.add(np.packbits(np.array(gs).reshape(-1)).tobytes())
        massE += m
        _, bad = Cn.outcome_pieces(c, model, ops, base)
        V = [R[list(tr)] for R, Z in bad for tr in Cn.triangulate(R, Z, ops)]
        if V:
            p, pe = Cn.cone_measure(np.array(V), ops, rtol=rtol)
            wrong += float(p.sum()); err += float(pe.sum())
    parts = dict(ops.parts); parts.update({"prune_" + k: v for k, v in ops_p.parts.items()})
    n = ops.n + ops_p.n
    return dict(wrong=wrong, mass=massE, err=err, keys=frozenset(keys), visited=visited[1:], kept=kept[1:],
                flops=n, parts=parts, fpe=n / _fpe_flops(H, T))


def walk_rest(model, budget, keys, rule_seed=FE.RULE_SEED):
    """Frozen T10 estimator, unchanged, with its walk restricted to the complement of the cells in keys."""
    FE.W.walk = lambda *a, **k: mwalk.walk(*a, excl=keys, **k)
    try:
        acc, used = FE.est(model, budget, None, rule_seed=rule_seed)
    finally:
        FE.W.walk = FROZEN_WALK
    return 1.0 - acc, used


def est(model, budget, rng=None, tau=1e-3, rule_seed=FE.RULE_SEED, details=False):
    model = tuple(np.asarray(w, dtype=np.float64) for w in model)
    ex = exact_part(model, tau)
    rest, used = walk_rest(model, budget - ex["fpe"], ex["keys"], rule_seed)
    a = 1.0 - ex["wrong"] - rest
    if details:
        return a, ex["fpe"] + used, dict(ex, keys=len(ex["keys"]), rest=rest, walk_fpe=used)
    return a, ex["fpe"] + used
