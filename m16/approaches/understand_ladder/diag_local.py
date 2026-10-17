"""Diagnostic (MC, not an estimator): are a model's errors thin wedges at the true boundaries x_i = x_j?
For misclassified inputs: angle (deg) to the nearest adjacent-rank true boundary, and the number of hidden neurons
active at the final step at the projection of the point onto that boundary.
usage: python diag_local.py H T [n]"""
import sys

import numpy as np

from zoo import forward, noisy_abs, zoo_model


def final_hidden(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h


def local_stats(model, n, rng):
    T = model[2].shape[0]
    x = rng.standard_normal((n, T))
    pred = forward(model, x).argmax(1)
    srt = np.sort(x, 1); truth = np.argsort(x, 1)[:, -2]
    bad = pred != truth
    xb = x[bad]; sb = np.sort(xb, 1)
    # adjacent-rank gaps involving the 2nd largest: (top - 2nd) and (2nd - 3rd)
    g1 = sb[:, -1] - sb[:, -2]; g2 = sb[:, -2] - sb[:, -3]
    gap = np.minimum(g1, g2) / np.sqrt(2) / np.linalg.norm(xb, axis=1)
    ang = np.degrees(np.arcsin(np.clip(gap, 0, 1)))
    # project onto the nearest boundary (set the two closest of the pair equal)
    xp = xb.copy(); o = np.argsort(xb, 1)
    for r in range(len(xb)):
        a, b = (o[r, -1], o[r, -2]) if g1[r] < g2[r] else (o[r, -2], o[r, -3])
        m = (xb[r, a] + xb[r, b]) / 2; xp[r, a] = m; xp[r, b] = m
    nact = (final_hidden(model, xp) > 0).sum(1)
    return bad.mean(), ang, nact


if __name__ == "__main__":
    H, T = int(sys.argv[1]), int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
    rng = np.random.default_rng(0)
    rows = [(f"seed{s}", zoo_model(H, T, s)) for s in range(5)]
    rows += [(f"abs0.05_{k}", m) for k, m in enumerate(noisy_abs(zoo_model(H, T, 0), 3, 0.05, 0))]
    for name, m in rows:
        e, ang, nact = local_stats(m, n, rng)
        q = np.percentile(ang, [50, 90, 99]) if len(ang) else [np.nan] * 3
        print(f"{name:12s} err {e:.5f}  angle to boundary deg p50 {q[0]:5.2f} p90 {q[1]:5.2f} p99 {q[2]:5.2f}  "
              f"active-at-boundary counts {np.bincount(nact, minlength=H + 1)[:H + 1].tolist()}")
