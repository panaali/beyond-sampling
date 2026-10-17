"""Deterministic co-area estimator. est_faces(model, n23, n12, seed) -> (acc, cost, details)."""
import numpy as np
import core as C


def est_faces(model, n23, n12, rule_seed=0, chunk=4096):
    T = model[2].shape[0]
    _, Z23, Z12 = C.tables(T)
    out = {}
    cost = 0.0
    err = 0.0
    for kind, Z, n in (("23", Z23, n23), ("12", Z12, n12)):
        if Z == 0 or n == 0:
            continue
        U = C.sobol(T, n, rule_seed + (0 if kind == "23" else 7))
        ms = []
        for i in range(0, n, chunk):
            y, a, b, tau = C.face_points(T, kind, U[i:i + chunk])
            m, c, _ = C.walk_mass(model, y, a, b, tau, kind)
            ms.append(m); cost += c.sum()
        m = np.concatenate(ms)
        out[kind] = m
        err += Z * m.mean()
    return 1 - err, cost, out
