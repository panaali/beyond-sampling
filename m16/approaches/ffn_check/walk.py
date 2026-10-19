"""T394: exact great-circle walk for a bias-free one-hidden-layer ReLU net, logits = W2 relu(W1 x). Own code.

circle(W1, W2, u, v) returns (exact fraction of t in [0, 2pi) where x(t) = u cos t + v sin t is classified
correctly, FLOPs charged). Events: gate flips at phi_k -+ pi/2, pair crossings x_i = x_j, and quadrant breaks
pi/2, pi, 3pi/2 (so each arc is <= pi/2 and every logit-difference sinusoid has <= 1 zero on it).
FLOP charge follows the README's pre-registered list; trig/atan2 = TRIG FLOPs. Numpy work beyond the charge:
the stable argsort of events (charged n log2 n) and index bookkeeping (no arithmetic).
"""
import numpy as np

T, TRIG, TWO_PI = 10, 20, 2 * np.pi
PI_, PJ_ = np.triu_indices(T, 1)


def load(path):
    import json
    d = json.load(open(path))
    return np.array(d["W1"]), np.array(d["W2"])


def fwd_flops(W1, W2):
    return 2 * (W1.size + W2.size) + W1.shape[0]


def forward(W1, W2, x):
    return np.maximum(x @ W1.T, 0) @ W2.T


def accuracy(W1, W2, n, seed, chunk=1_000_000):
    rng, ok = np.random.default_rng(seed), 0
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, T))
        ok += int((forward(W1, W2, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
    return ok / n


def circle(W1, W2, u, v):
    h = W1.shape[0]
    a, c = W1 @ u, W1 @ v
    phi = np.arctan2(c, a)
    du, dv = u[PI_] - u[PJ_], v[PI_] - v[PJ_]
    psi = np.arctan2(dv, du)                       # x_i - x_j > 0 on (psi - pi/2, psi + pi/2)
    npair = len(psi)
    flops = 4 * h * T + h * TRIG + 2 * npair + npair * TRIG
    # event table: angle, kind (0 gate, 1 pair, 2 quadrant), payload index, sign (+1 on / overtake, -1 off / drop)
    ang = np.concatenate([phi - np.pi / 2, phi + np.pi / 2, psi - np.pi / 2, psi + np.pi / 2,
                          [np.pi / 2, np.pi, 1.5 * np.pi]]) % TWO_PI
    kind = np.concatenate([np.zeros(2 * h, int), np.ones(2 * npair, int), np.full(3, 2)])
    idx = np.concatenate([np.arange(h), np.arange(h), np.arange(npair), np.arange(npair), np.zeros(3, int)])
    sgn = np.concatenate([np.ones(h), -np.ones(h), np.ones(npair), -np.ones(npair), np.zeros(3)])
    o = np.argsort(ang, kind="stable")
    ang, kind, idx, sgn = ang[o], kind[o], idx[o], sgn[o]
    ne = len(ang)
    flops += int(ne * np.log2(ne))
    # gate track: A, B after each gate event (rank-1 update: 2T mults + 2T adds for A and B)
    g = kind == 0
    gk, gs = idx[g], sgn[g]
    A0, B0 = W2 @ np.where(a > 0, a, 0.0), W2 @ np.where(a > 0, c, 0.0)   # at t = 0+: on iff a > 0
    flops += 2 * h + 4 * h * T
    A = np.vstack([A0, A0 + np.cumsum((gs * a[gk])[:, None] * W2[:, gk].T, 0)])
    B = np.vstack([B0, B0 + np.cumsum((gs * c[gk])[:, None] * W2[:, gk].T, 0)])
    flops += g.sum() * 4 * T
    # rank track: rank 9 = largest; target = rank 8. Pair event (i over j: +1/-1)
    p = kind == 1
    pk, ps = idx[p], sgn[p]
    r0 = np.argsort(np.argsort(u)).astype(float)
    dR = np.zeros((p.sum(), T))
    dR[np.arange(p.sum()), PI_[pk]] = ps
    dR[np.arange(p.sum()), PJ_[pk]] = -ps
    R = np.vstack([r0, r0 + np.cumsum(dR, 0)])
    tgt_state = np.argmax(R == 8, 1)
    flops += 2 * T + p.sum() * T + (p.sum() + 1) * T
    # arcs: arc e runs from ang[e-1] (0 for e = 0) to ang[e] (2pi for the last); state = events before it
    lo = np.concatenate([[0.0], ang]); hi = np.concatenate([ang, [TWO_PI]])
    ng = np.concatenate([[0], np.cumsum(g)]); npv = np.concatenate([[0], np.cumsum(p)])
    keep = hi > lo
    lo, hi, ng, npv = lo[keep], hi[keep], ng[keep], npv[keep]
    na = len(lo)
    tg = tgt_state[npv]
    Aa, Ba = A[ng], B[ng]
    r = np.arange(na)
    DA = Aa[r, tg][:, None] - Aa; DB = Ba[r, tg][:, None] - Ba        # logit_t - logit_k = DA cos + DB sin
    flops += 2 * ne * TRIG
    # the kept arcs' ends: lo_e = previous angle, hi_e = this angle
    ends = np.concatenate([[0.0], ang, [TWO_PI]])
    jlo = np.flatnonzero(keep); jhi = jlo + 1
    ce, se = np.cos(ends), np.sin(ends)               # once per event angle (and 0, 2pi)
    clo, slo, chi, shi = ce[jlo], se[jlo], ce[jhi], se[jhi]
    flo = DA * clo[:, None] + DB * slo[:, None]
    fhi = DA * chi[:, None] + DB * shi[:, None]
    flops += na * 8 * T
    plo, phi_ = flo > 0, fhi > 0
    self_ = np.zeros_like(plo); self_[r, tg] = True
    L = np.broadcast_to(lo[:, None], plo.shape).copy(); U = np.broadcast_to(hi[:, None], plo.shape).copy()
    ch = (plo != phi_) & ~self_
    ri, ki = np.nonzero(ch)
    z = (np.arctan2(DB[ri, ki], DA[ri, ki]) + np.pi / 2 - lo[ri]) % np.pi + lo[ri]
    z = np.clip(z, lo[ri], hi[ri])
    flops += len(ri) * (TRIG + 4)
    up = phi_[ri, ki]                       # negative -> positive: feasible [z, hi]; else [lo, z]
    L[ri[up], ki[up]] = z[up]
    U[ri[~up], ki[~up]] = z[~up]
    dead = ~plo & ~phi_ & ~self_
    U[dead] = L[dead]                       # empty
    Lm, Um = L.max(1), U.min(1)
    flops += na * 6 * T
    return float(np.clip(Um - Lm, 0, None).sum() / TWO_PI), int(flops)


def brute(W1, W2, u, v, n=100_000):
    t = (np.arange(n) + 0.5) * TWO_PI / n
    x = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    return float((forward(W1, W2, x).argmax(1) == np.argsort(x, 1)[:, -2]).mean())
