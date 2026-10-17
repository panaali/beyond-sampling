"""exact_acc(model) -> (accuracy, info): deterministic exact accuracy of a bias-free 2nd-argmax RNN, FLOPs counted.

accuracy = 1 - sum of the measures of the convex WRONG pieces (fewer than the correct ones; both partition space)."""
import time

import numpy as np

import cones as E


def enumerate_pieces(model, ops):
    T = model[2].shape[0]
    st = {}
    cells = E.enumerate_cells(model, ops, st)
    base = 1 + 16 * (T - 1)
    good = []; bad = []
    for c in cells:
        g, b = E.outcome_pieces(c, model, ops, base)
        good += g; bad += b
    return cells, st, good, bad


def simplices(pieces, ops, T):
    simp = []
    for R, Z in pieces:
        for tri in E.triangulate(R, Z, ops):
            simp.append(R[list(tri)])
    return np.array(simp).reshape(-1, T, T)


def exact_acc(model, check=False, tol=1e-12):
    """check=True also measures the correct pieces (correct + wrong must sum to 1); that work is not charged."""
    t0 = time.process_time()
    T = model[2].shape[0]
    ops = E.Ops()
    cells, st, good, bad = enumerate_pieces(model, ops)
    V = simplices(bad, ops, T)
    p, err = measure(V, ops, tol)
    info = dict(T=T, cells=len(cells), cells_per_step=st["cells_per_step"], wrong_pieces=len(bad),
                correct_pieces=len(good), simplices=len(V), flops=ops.n, parts=ops.parts,
                err_est=float(np.nansum(err)), err_nan=int(np.isnan(err).sum()), cpu_s=time.process_time() - t0)
    if check:
        dops = E.Ops()
        pg, eg = measure(simplices(good, dops, T), dops, tol)
        info["correct_mass"] = float(pg.sum()); info["total_mass"] = float(pg.sum() + p.sum())
        info["check_err_est"] = float(np.nansum(eg))
    return 1.0 - float(p.sum()), info


def measure(V, ops, tol=1e-12, chunk=20000):
    ps = []; es = []
    for i in range(0, len(V), chunk):
        R = E._corr(V[i:i + chunk], ops)
        p, e = E.orthant(R, ops, tol=tol)
        ps.append(p); es.append(e)
    if not ps:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(ps), np.concatenate(es)
