"""Exact great-circle walk for a depth-1 bias-free ReLU MLP (logits = W2 relu(W1 x)), incremental and vectorised.

x(theta) = u cos(theta) + v sin(theta), theta uniform on [0, 2pi), u, v ~ N(0, I): x(theta) ~ N(0, I) for every
theta, so the exact fraction of the circle that the net classifies correctly is an unbiased estimate of accuracy.

On the circle every hidden pre-activation is a_k cos + c_k sin, so each hidden unit flips exactly twice, at angles
known in closed form. Between flips the logits are A cos + B sin with A = W2 (g * a), B = W2 (g * c), updated by a
rank-1 term per flip. The target changes where two inputs cross (45 pairs, 2 angles each). On each sub-arc between
consecutive events (all shorter than pi) the gates and target t are fixed, and "logit t beats logit k" is a
sinusoid with at most one zero inside the sub-arc, so the correct set is one interval found from endpoint signs and
zeros. Its length / 2pi is summed.

FLOPs are counted for the event-driven algorithm described in the comments of `circle` (numpy does some redundant
vectorised work that does not change the result and is not charged; each such place is marked); atan2/cos/sin are
charged TRIG = 20 FLOPs each.
"""
import numpy as np

T, TRIG = 10, 20


def circle(W1, W2, u, v):
    """Return (exact correct fraction of the great circle through u, v; FLOPs)."""
    h = W1.shape[0]
    a, c = W1 @ u, W1 @ v                                   # pre-activation = a cos + c sin = r cos(th - phi)
    phi = np.arctan2(c, a)
    flops = 2 * 2 * W1.size + h * TRIG
    # unit k is on for th in (phi - pi/2, phi + pi/2): on-event at phi - pi/2, off-event at phi + pi/2
    on_ev, off_ev = np.mod(phi - np.pi / 2, 2 * np.pi), np.mod(phi + np.pi / 2, 2 * np.pi)
    # target events: x_i - x_j = du cos + dv sin crosses 0
    i, j = np.triu_indices(T, 1)
    tz = np.mod(np.arctan2(v[i] - v[j], u[i] - u[j]) + np.pi / 2, 2 * np.pi)
    tev = np.concatenate([tz, np.mod(tz + np.pi, 2 * np.pi)])
    flops += 2 * len(i) + len(i) * TRIG + len(i)
    ev = np.concatenate([on_ev, off_ev, tev])
    order = np.argsort(ev)
    ev_s = ev[order]
    n_ev = len(ev)
    flops += int(n_ev * np.log2(n_ev))
    # gates at theta = 0 (start of [0, 2pi)): on iff a > 0
    g0 = (a > 0).astype(float)
    # rank-1 increments to (A, B) at each gate event; zero for target events
    sgn = np.zeros(n_ev)
    sgn[:h], sgn[h:2 * h] = 1.0, -1.0
    unit = np.concatenate([np.arange(h), np.arange(h), np.full(len(tev), -1)])
    s_o, u_o = sgn[order], unit[order]
    gate = u_o >= 0
    dA = np.zeros((n_ev, T)); dB = np.zeros((n_ev, T))
    dA[gate] = s_o[gate, None] * W2[:, u_o[gate]].T * a[u_o[gate], None]
    dB[gate] = s_o[gate, None] * W2[:, u_o[gate]].T * c[u_o[gate], None]
    A0, B0 = W2 @ (g0 * a), W2 @ (g0 * c)
    flops += 2 * 2 * W2.size + 2 * h * (2 * T + 2 * T)       # A0, B0; each of 2h rank-1 updates (A and B)
    A = A0 + np.vstack([np.zeros(T), np.cumsum(dA, 0)])    # (n_ev + 1, T): logits coefficients on each sub-arc
    B = B0 + np.vstack([np.zeros(T), np.cumsum(dB, 0)])
    lo = np.concatenate([[0.0], ev_s]); hi = np.concatenate([ev_s, [2 * np.pi]])
    keep = hi > lo
    A, B, lo, hi = A[keep], B[keep], lo[keep], hi[keep]
    n_arc = len(lo)
    # target on each sub-arc, from its midpoint (cost: 2T mults + adds per arc, plus a 2nd-largest scan of 10)
    mid = 0.5 * (lo + hi)
    cm, sm = np.cos(mid), np.sin(mid)
    X = cm[:, None] * u[None] + sm[:, None] * v[None]
    tg = np.argsort(X, 1)[:, -2]
    # charged as an event-driven tracker: the target changes only at target events (a rank swap, ~5 FLOPs each);
    # the midpoint argsort above computes the same labels and is not charged
    flops += 2 * len(i) * 5 + T * T
    # f_k(th) = (A_t - A_k) cos th + (B_t - B_k) sin th > 0 for all k != t
    r = np.arange(n_arc)
    dAk = A[r, tg][:, None] - A; dBk = B[r, tg][:, None] - B
    cl, sl, ch, sh = np.cos(lo), np.sin(lo), np.cos(hi), np.sin(hi)
    f_lo = dAk * cl[:, None] + dBk * sl[:, None]
    f_hi = dAk * ch[:, None] + dBk * sh[:, None]
    z = np.mod(np.arctan2(dBk, dAk) + np.pi / 2, np.pi)     # zeros of f_k are z and z + pi
    z = np.where(z < lo[:, None], z + np.pi, z)
    z = np.where(z < lo[:, None], z + np.pi, z)
    # cos/sin once per event angle (shared by neighbouring arcs); f at both arc ends (A cos + B sin and the
    # difference to the target: 4 FLOPs per logit per end); zeros (atan2) only where f changes sign in the arc;
    # numpy evaluates all zeros, but only the sign-changing ones are used and charged
    n_zero = int(((f_lo > 0) != (f_hi > 0)).sum())
    flops += 2 * n_ev * TRIG + n_arc * 2 * 4 * T + n_zero * (TRIG + 4)
    # feasible sub-interval of [lo, hi] for each k: the positive part of a sinusoid with <= 1 zero in the arc
    L = np.broadcast_to(lo[:, None], f_lo.shape).copy(); H = np.broadcast_to(hi[:, None], f_lo.shape).copy()
    pos_lo, pos_hi = f_lo > 0, f_hi > 0
    L = np.where(~pos_lo & pos_hi, np.minimum(z, hi[:, None]), L)
    H = np.where(pos_lo & ~pos_hi, np.minimum(z, hi[:, None]), H)
    none = ~pos_lo & ~pos_hi
    H = np.where(none, L, H)
    self_k = np.zeros_like(none); self_k[r, tg] = True
    L = np.where(self_k, lo[:, None], L); H = np.where(self_k, hi[:, None], H)
    Lm, Hm = L.max(1), H.min(1)
    frac = float(np.clip(Hm - Lm, 0, None).sum() / (2 * np.pi))
    flops += n_arc * (6 * T)
    return frac, flops


def walk(model, rng):
    """Interface like walks.circle_walk: (value, flops, pieces, aux, flops)."""
    u, v = rng.standard_normal(T), rng.standard_normal(T)
    f, fl = circle(model.W[0], model.W[1], u, v)
    return f, fl, 2 * model.W[0].shape[0], 0, fl
