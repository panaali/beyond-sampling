"""Cheap proxy for the face integrand: linearise the network at the tie point (t = 0) and integrate the misclassified
mass of that LINEAR model over the whole window (exact when the window is one linear piece).
Cost: one value pass (prefix before min(a,b) value-only) + one tangent pass from min(a,b), charged as executed."""
import numpy as np
from scipy.special import ndtr


def proxy_lin(model, y, a, b, tau, kind, cdf=ndtr):
    Wi, Wh, Wo = model
    n, T = y.shape; H = Wh.shape[0]; rows = np.arange(n)
    d = np.zeros_like(y); d[rows, a] = 1 / np.sqrt(2); d[rows, b] = -1 / np.sqrt(2)
    tpos = a if kind == "23" else b
    tneg = b if kind == "23" else a
    amin = np.minimum(a, b)
    h = np.zeros((n, H)); dh = np.zeros((n, H))
    for s in range(T):
        z = h @ Wh.T + y[:, s:s + 1] * Wi[None]
        dz = dh @ Wh.T + d[:, s:s + 1] * Wi[None]          # zero rows before amin (cost not charged there)
        g = z > 0
        h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
    L = h @ Wo.T; dL = dh @ Wo.T
    step_f = 2 * H * H + 2 * H
    cost = (T + (T - amin)) / T + (4 * T * H + 4 * T * H) / (T * step_f)
    mass = np.zeros(n); rr = rows
    for side in (0, 1):
        if side == 0:
            p_lo, p_hi, k = -tau, np.zeros(n), tneg
        else:
            p_lo, p_hi, k = np.zeros(n), tau, tpos
        c = L[rr, k][:, None] - L
        e = dL[rr, k][:, None] - dL
        with np.errstate(divide="ignore", invalid="ignore"):
            root = -c / e
        lb = np.where(e > 0, root, -np.inf); ub = np.where(e < 0, root, np.inf)
        bad0 = (e == 0) & (c < 0)
        lb[rr, k] = -np.inf; ub[rr, k] = np.inf
        clo = np.maximum(lb.max(1), p_lo); chi = np.minimum(ub.min(1), p_hi)
        corr = np.where((chi > clo) & ~bad0.any(1), cdf(chi) - cdf(clo), 0.0)
        mass += cdf(p_hi) - cdf(p_lo) - corr
    return mass, cost
