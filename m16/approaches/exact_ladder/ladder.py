"""exact_acc(model) -> (accuracy, info): deterministic exact accuracy of a bias-free 2nd-argmax RNN, FLOPs counted.

accuracy = 1 - sum of the measures of the convex WRONG pieces (fewer than the correct ones; both partition space).
Streaming: cells are enumerated depth-first and wrong pieces are triangulated and measured in chunks (memory bound)."""
import time

import numpy as np

import cones as E


def exact_acc(model, check=False, rtol=1e-6, chunk=8192):
    """check=True also measures the correct pieces (correct + wrong must sum to 1); that work is not charged."""
    t0 = time.process_time()
    T = model[2].shape[0]
    ops = E.Ops(); dops = E.Ops()
    counts = [0] * T
    base = 1 + 16 * (T - 1)
    acc = dict(wrong=0.0, werr=0.0, good=0.0, gerr=0.0, nw=0, ng=0, sw=0, sg=0)
    buf = {"w": [], "g": []}

    def flush(key, force=False):
        b = buf[key]
        if b and (force or len(b) >= chunk):
            V = np.array(b); buf[key] = []
            p, e = E.cone_measure(V, ops if key == "w" else dops, rtol=rtol)
            if key == "w":
                acc["wrong"] += p.sum(); acc["werr"] += e.sum(); acc["sw"] += len(V)
            else:
                acc["good"] += p.sum(); acc["gerr"] += e.sum(); acc["sg"] += len(V)

    for c in E.iter_cells(model, ops, counts):
        good, bad = E.outcome_pieces(c, model, ops, base)
        acc["nw"] += len(bad); acc["ng"] += len(good)
        for R, Z in bad:
            buf["w"] += [R[list(tr)] for tr in E.triangulate(R, Z, ops)]
        flush("w")
        if check:
            for R, Z in good:
                buf["g"] += [R[list(tr)] for tr in E.triangulate(R, Z, dops)]
            flush("g")
    flush("w", True)
    if check:
        flush("g", True)
    info = dict(T=T, cells=counts[-1], cells_per_step=counts, wrong_pieces=acc["nw"], correct_pieces=acc["ng"],
                simplices=acc["sw"], flops=ops.n, parts=ops.parts, err_est=acc["werr"],
                cpu_s=time.process_time() - t0)
    if check:
        info.update(correct_mass=acc["good"], total_mass=acc["good"] + acc["wrong"], check_err_est=acc["gerr"],
                    correct_simplices=acc["sg"])
    return 1.0 - acc["wrong"], info
