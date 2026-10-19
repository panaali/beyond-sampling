"""T385 estimator (SAMPLING-class, own code): rank-r coordinate-line conditional MC for P(E1) and dP(E1)/dW.

Decomposition: for a fixed rank r, P(E1) = sum_k int p(x_-k) int_{I_r(x_-k)} phi(s) 1{E1(x_-k, x_k = s)} ds dx_-k,
where I_r(x_-k) is the set of s at which x_k = s is the r-th largest value AND g23 > t0 (an interval whose ends are
order statistics of x_-k and t0: model-free). Drawing x from the exact conditional law q = p 1{g23 > t0} / p_gap and
setting k = position of the rank-r value gives (k, x_-k) with density p(x_-k) A / p_gap, A = Phi-mass of I_r, so
   P = p_gap E_q[Z / A],  Z = int_{I_r} phi(s) 1{pick(s) = p3} ds   (unbiased; 0 <= Z/A <= 1).
Z is computed exactly by walking the ReLU cells of the line (affine logits per cell). Its weight-derivative is the
sum over event-boundary points e inside cells (roots of the binding margin m = L_p3 - L_j) of phi(e) dm/dW / |dm/ds|
(I_r's ends do not depend on W). dm/dW at x(e) by torch autograd (independent of the walk's numpy code).
r is drawn per line from a fixed distribution (independent of x), so any mixture of ranks stays unbiased."""
import numpy as np
import torch
from scipy.special import ndtr

import core

TOL = 1e-12


def interval(x, pos, r, t0):
    """Line for rank r (1-based) through the rest of x: returns k, lo, hi, p3 (position of the rank-3 input on I_r)."""
    n = len(x); ar = np.arange(n); v = x[ar[:, None], pos]          # v[:, j] = (j+1)-th largest value
    k = pos[:, r - 1]
    u = np.delete(v, r - 1, axis=1)                                  # order stats of the other nine: u[:, 0] largest
    if r == 1:
        lo, hi, p3 = u[:, 0], u[:, 0] + 9.0, pos[:, 2]               # v1 = s; v2 = u1, v3 = u2 (pos of old rank 3)
    elif r == 2:
        lo, hi, p3 = u[:, 1] + t0, u[:, 0], pos[:, 2]                # v2 = s > u2 + t0, v3 = u2 (old rank 3)
    elif r == 3:
        lo, hi, p3 = u[:, 2], u[:, 1] - t0, k                        # v3 = s
    else:
        lo = u[:, r - 1] if r < 10 else u[:, 8] - 9.0
        hi, p3 = u[:, r - 2], pos[:, 2]                              # v2, v3 unchanged
    return k, lo, hi, p3


def phimass(a, b):
    return np.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))


def walk(model, X, k, lo, hi, p3, max_iter=10000):
    """Exact Z per line and the event-boundary points: list of (line, s_e, competitor j, |dm/ds|)."""
    Wi, Wh, Wo = model; WhT, WoT = Wh.T, Wo.T; n, T = X.shape; H = len(Wi)
    Z = np.zeros(n); s = lo.copy(); act = np.flatnonzero(s < hi); bl, bs, bj, bb = [], [], [], []; cells = 0
    for _ in range(max_iter):
        if len(act) == 0:
            break
        sa = s[act]; Xa = X[act]; ka = k[act]; m = len(act); cells += m
        h = np.zeros((m, H)); dh = np.zeros((m, H)); ends = hi[act].copy()
        for t in range(T):
            on = (ka == t)
            xt = np.where(on, sa, Xa[:, t])
            z = h @ WhT + xt[:, None] * Wi; dz = dh @ WhT + on[:, None] * Wi
            g = (z > TOL) | ((np.abs(z) <= TOL) & (dz > 0))
            cross = (g & (dz < 0)) | (~g & (dz > 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                root = np.where(cross, sa[:, None] - z / dz, np.inf)
            root = np.where(root > sa[:, None], root, np.inf)
            ends = np.minimum(ends, root.min(1))
            h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        L = h @ WoT; dL = dh @ WoT; ar = np.arange(m)
        mg = L[ar, p3[act]][:, None] - L; bt = dL[ar, p3[act]][:, None] - dL
        mg[ar, p3[act]] = 1.0; bt[ar, p3[act]] = 0.0                   # self-margin: always satisfied
        with np.errstate(divide="ignore", invalid="ignore"):
            rt = sa[:, None] - mg / bt
        lower = np.where(bt > 0, rt, -np.inf); upper = np.where(bt < 0, rt, np.inf)
        jl = lower.argmax(1); ju = upper.argmin(1)
        c = np.maximum(sa, lower[ar, jl]); d = np.minimum(ends, upper[ar, ju])
        dead = ((bt == 0) & (mg <= 0)).any(1)
        ok = (d > c) & ~dead
        if ok.any():
            i = np.flatnonzero(ok); Z[act[i]] += phimass(c[i], d[i])
            for side, j, e in ((lower[ar, jl] > sa, jl, c), (upper[ar, ju] < ends, ju, d)):
                sel = i[side[i]]
                if len(sel):
                    bl.append(act[sel]); bs.append(e[sel]); bj.append(j[sel]); bb.append(np.abs(bt[sel, j[sel]]))
        s[act] = ends; act = act[ends < hi[act]]
    else:
        raise RuntimeError("walk did not finish")
    cat = (lambda a, dt: np.concatenate(a) if a else np.zeros(0, dt))
    return Z, cat(bl, int), cat(bs, float), cat(bj, int), cat(bb, float), cells


def margin_grads(model, X, k, s, p3, j):
    """dm/dW (432) at x = X with x_k = s, m = L_p3 - L_j, by torch autograd (float64), one row per point."""
    if len(s) == 0:
        return np.zeros((0, 432)), np.zeros(0)
    w = torch.tensor(core.flat(model), dtype=torch.float64, requires_grad=True)
    Xe = X.copy(); Xe[np.arange(len(s)), k] = s; Xe = torch.tensor(Xe)
    Wi, Wh, Wo = w[:16], w[16:272].reshape(16, 16), w[272:].reshape(10, 16)
    out = []
    for i in range(len(s)):
        h = torch.zeros(16, dtype=torch.float64)
        for t in range(10):
            h = torch.relu(Wh @ h + Xe[i, t] * Wi)
        L = Wo @ h; m = L[p3[i]] - L[j[i]]
        g, = torch.autograd.grad(m, w); out.append((g.numpy().copy(), float(m.detach())))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def run_chunk(model, law, rng, n, rank_probs):
    """One chunk of n lines. Returns per-hit-line rows: Z/A * p_gap and gradient * p_gap / A (432), plus counters."""
    x, pos = law.draw(rng, n)
    ranks = rng.choice(len(rank_probs), size=n, p=rank_probs) + 1
    k = np.empty(n, int); lo = np.empty(n); hi = np.empty(n); p3 = np.empty(n, int)
    for r in np.unique(ranks):
        i = np.flatnonzero(ranks == r); k[i], lo[i], hi[i], p3[i] = interval(x[i], pos[i], r, law.t0)
    A = phimass(lo, hi)
    Z, bl, bs, bj, bb, cells = walk(model, x, k, lo, hi, p3)
    G, mres = margin_grads(model, x[bl], k[bl], bs, p3[bl], bj)
    coef = np.exp(-bs ** 2 / 2) / np.sqrt(2 * np.pi) / bb * law.p_gap / A[bl]
    hit = np.flatnonzero(Z > 0); idx = {l: q for q, l in enumerate(hit)}
    grad = np.zeros((len(hit), 432))
    for e in range(len(bl)):
        grad[idx[bl[e]]] += coef[e] * G[e]
    return dict(zval=Z[hit] / A[hit] * law.p_gap, grad=grad, rank=ranks[hit], n=n, n_end=len(bl), cells=cells,
                max_mres=float(np.abs(mres).max()) if len(mres) else 0.0, rank_counts=np.bincount(ranks, minlength=11)[1:])
