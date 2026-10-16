"""Exact conditional accuracy given x0..x8: integrate the last input x9 analytically.

The last RNN step is h10 = relu(a + Wi*x9) with a = Wh h9, so the logits are piecewise LINEAR in x9 with at
most 16 breakpoints (one per neuron). The target position changes only where x9 crosses the top-2 of x0..x8.
On each piece, 'argmax == target' is an intersection of linear inequalities in x9, i.e. an interval, whose
Gaussian mass is exact. Q(x0..x8) = P(correct | x0..x8) exactly; E[Q] = accuracy.
"""
import numpy as np
from scipy.special import ndtr, stdtr

T = 10


def h9(model, x9):
    Wi, Wh, Wo = model
    h = np.zeros((len(x9), Wh.shape[0]))
    for t in range(T - 1):
        h = np.maximum(h @ Wh.T + x9[:, t:t + 1] * Wi[None], 0)
    return h


def Q(model, x, radial=False):
    """x: (N, >=9); only x[:, :9] used. Returns exact P(correct | x0..x8).
    radial=True: also integrate the prefix's scale. The net has no biases, so correctness is invariant under
    x -> c x (c>0); with x0..8 = R u (R^2 ~ chi2_9 independent of u, x9), correctness depends on (u, x9/R), and
    3 x9/R ~ Student t_9. Returns exact P(correct | direction of x0..x8)."""
    Wi, Wh, Wo = model
    x = x[:, :9]; N = len(x)
    if radial:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        cdf = lambda z: stdtr(9, 3 * z)
    else:
        cdf = ndtr
    a = h9(model, x) @ Wh.T                       # (N,16) pre-activation offset at last step
    w = Wi                                        # slope
    with np.errstate(divide="ignore"):
        bp = np.where(np.abs(w)[None] > 1e-12, -a / w[None], np.nan)
    o = np.argsort(-x, 1); q1, q2 = o[:, 0], o[:, 1]
    m1 = x[np.arange(N), q1]; m2 = x[np.arange(N), q2]
    pts = np.concatenate([bp, m1[:, None], m2[:, None]], 1)
    pts = np.where(np.isnan(pts), np.inf, pts)
    pts = np.sort(pts, 1)
    edges = np.concatenate([np.full((N, 1), -np.inf), pts, np.full((N, 1), np.inf)], 1)
    total = np.zeros(N); idx = np.arange(N)
    for k in range(edges.shape[1] - 1):
        lo, hi = edges[:, k], edges[:, k + 1]
        ok = hi > lo
        if not ok.any():
            continue
        mid = np.where(np.isinf(lo), hi - 1, np.where(np.isinf(hi), lo + 1, (lo + hi) / 2))
        mid = np.where(np.isinf(mid), 0.0, mid)
        D = (a + w[None] * mid[:, None]) > 0
        A = (D * a) @ Wo.T; B = (D * w[None]) @ Wo.T            # logits = A + B x9
        tgt = np.where(mid > m1, q1, np.where(mid > m2, 9, q2))
        At = A[idx, tgt][:, None]; Bt = B[idx, tgt][:, None]
        dA = At - A; dB = Bt - B                                # need dA + dB*x9 > 0 for all j != tgt
        dA[idx, tgt] = 1.0; dB[idx, tgt] = 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -dA / dB
        lower = np.where(dB > 0, r, -np.inf); upper = np.where(dB < 0, r, np.inf)
        never = (dB == 0) & (dA <= 0)
        L = np.maximum(lo, lower.max(1)); U = np.minimum(hi, upper.min(1))
        mass = np.where((U > L) & ok & ~never.any(1), cdf(U) - cdf(L), 0.0)
        total += mass
    return total


# Compute accounting (multiply-adds): forward pass = 10*(16*16+16) + 10*16 = 2880.
# Q: 9 steps (2448) + initial A,B on one piece (2*160) + <=16 rank-1 toggles of (A,B) (16*20)
#    + <=18 pieces x 10 linear inequalities (~18*30) = ~3630  -> 1.26 forward-equivalents. We charge 1.5.
COST_Q = 1.5
