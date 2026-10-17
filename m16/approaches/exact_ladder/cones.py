"""Exact accuracy of a bias-free ReLU RNN M_H,T (2nd-argmax task) by cone enumeration + exact cone measure.

No random inputs anywhere (deterministic, mechanistic). Pipeline:
 1. Gate-cell enumeration over time (incremental, exact). The gate pattern of step t depends only on (x_1..x_t),
    so a step-t cell is a pointed polyhedral cone in R^t. Lifting to R^(t+1) and cutting by the 16 step-(t+1)
    pre-activation hyperplanes is done in the double-description (DD) representation: extreme rays + the set of
    constraints tight at each ray (bitsets). New rays come from adjacent (+,-) ray pairs (combinatorial
    adjacency test, Fukuda-Prodon), so tight sets are exact (no tolerances accumulate). Empty cells never appear.
 2. Outcome split per final cell: logits are linear (L = Wo A_T x); clip by "argmax = j" (T-1 halfspaces), then
    split by the order hyperplanes x_j - x_k keeping pieces where exactly one coordinate exceeds x_j
    (x_j is 2nd largest). Those convex pieces are the correct cones.
 3. Exact measure: pulling triangulation of each correct cone into simplicial cones (from the DD face lattice),
    then the Gaussian (= solid-angle) measure of each simplicial cone as an orthant probability P(Y >= 0),
    Y ~ N(0, R): closed forms for T <= 3, Plackett's identity integrated along R(t) = I + t(R - I) with
    tanh-sinh quadrature (deterministic, a-posteriori error estimate) for T = 4, 5; T = 6 nests it once.

Every FLOP that depends on the model is counted in Ops (multiply-add = 2 FLOPs; a transcendental = 20 FLOPs;
a 64-bit bitset AND/compare = 1 FLOP, popcount = 1 FLOP), as implemented.
"""
import math

import numpy as np

TRANS = 20          # FLOPs charged per transcendental (asin, sqrt, exp, tanh, ...)
TOL = 1e-11         # |a.r| below this (unit a, unit r) counts as tight (only degenerate data hits it)


class Ops:
    def __init__(self):
        self.n = 0.0
        self.parts = {}

    def add(self, k, part="misc"):
        self.n += k
        self.parts[part] = self.parts.get(part, 0.0) + k


# ------------------------------------------------------------------ DD cones
NW = 2              # 64-bit words per tight-set bitset (constraint ids < 128)


def bit(c):
    z = np.zeros(NW, np.uint64)
    z[c // 64] = np.uint64(1) << np.uint64(c % 64)
    return z


def clip(R, Z, a, cid, ops, want=(True, True)):
    """Split cone (rays R (k,d) unit rows, tight bitsets Z (k,NW)) by a.x >= 0 (constraint id cid).
    Returns (inside, outside), each (R, Z) or None if that side is not full-dimensional."""
    k, d = R.shape
    a = a / np.sqrt(a @ a); ops.add(3 * d + TRANS, "clip")
    v = R @ a; ops.add(2 * k * d, "clip")
    pos = v > TOL; neg = v < -TOL; zer = ~(pos | neg)
    b = bit(cid)
    if not neg.any() or not pos.any():
        Zn = Z.copy(); Zn[zer] |= b
        if not neg.any():
            return ((R, Zn) if want[0] else None), None
        return None, ((R, Zn) if want[1] else None)
    P = np.nonzero(pos)[0]; M = np.nonzero(neg)[0]
    com = Z[P][:, None, :] & Z[M][None, :, :]                       # (p, m, NW)
    cnt = np.bitwise_count(com).sum(-1)
    ops.add(len(P) * len(M) * NW * 2, "adj")
    pi, mi = np.nonzero(cnt >= d - 2)
    newR = []; newZ = []
    if len(pi):
        C = com[pi, mi]                                              # (c, NW)
        # contained[c, w] = C[c] subset of Z[w]
        cont = ((C[:, None, :] & ~Z[None, :, :]) == 0).all(-1)       # (c, k)
        ops.add(len(pi) * k * NW * 2, "adj")
        ncont = cont.sum(1)                                          # includes p and m themselves
        ok = ncont == 2
        pi, mi, C = pi[ok], mi[ok], C[ok]
        if len(pi):
            p = P[pi]; m = M[mi]
            nr = v[p][:, None] * R[m] - v[m][:, None] * R[p]
            nr /= np.sqrt((nr * nr).sum(1))[:, None]
            ops.add(len(p) * (3 * d + 3 * d + TRANS + d), "clip")
            newR = [nr]; newZ = [C | b]
    Zz = Z[zer] | b
    ins = outs = None
    if want[0]:
        ins = (np.concatenate([R[pos], R[zer]] + newR), np.concatenate([Z[pos], Zz] + newZ))
    if want[1]:
        outs = (np.concatenate([R[neg], R[zer]] + newR), np.concatenate([Z[neg], Zz] + newZ))
    return ins, outs


def lift(R, Z, a, alpha, cid, allmask, ops):
    """C x R intersected with a.x' + alpha*y >= 0 (alpha != 0), in R^(d+1). Returns (R, Z)."""
    k, d = R.shape
    y = -(R @ a) / alpha; ops.add(2 * k * d + k, "lift")
    nr = np.concatenate([R, y[:, None]], 1)
    nr /= np.sqrt((nr * nr).sum(1))[:, None]; ops.add(k * (3 * (d + 1) + TRANS), "lift")
    e = np.zeros((1, d + 1)); e[0, d] = np.sign(alpha)
    return np.concatenate([nr, e]), np.concatenate([Z | bit(cid), allmask[None]])


# ------------------------------------------------------------------ enumeration
def enumerate_cells(model, ops, stats=None):
    """Yield final gate cells (R, Z, A_T, allmask) in R^T. A_T: (H,T) with h_T = A_T x on the cell."""
    Wi, Wh, Wo = model
    H = len(Wi); T = Wo.shape[0]
    order = np.argsort(-np.abs(Wi))       # lift on the unit with the largest |Wi| (well-conditioned)
    cells = []
    for s in (1.0, -1.0):                 # step 1: x_1 > 0 or < 0; gates fixed
        g = (s * Wi) > 0
        A = np.zeros((H, T)); A[:, 0] = np.where(g, Wi, 0.0)
        cells.append((np.array([[s]]), np.zeros((1, NW), np.uint64), A, bit(0)))
    counts = [len(cells)]
    for t in range(1, T):                 # add x_{t+1} (0-based column t)
        nxt = []
        for R, Z, A, allm in cells:
            B = Wh @ A[:, :t]; ops.add(2 * H * H * t, "lift")     # z_{t+1} = B x' + Wi y
            cid = lambda k: 1 + 16 * (t - 1) + k
            k0 = order[0]
            pieces = []
            for sg in (1.0, -1.0):
                R1, Z1 = lift(R, Z, sg * B[k0], sg * Wi[k0], cid(k0), allm, ops)
                g = np.zeros(H, bool); g[k0] = sg > 0
                pieces.append((R1, Z1, g))
            for k in order[1:]:
                a = np.concatenate([B[k], [Wi[k]]])
                new = []
                for R1, Z1, g in pieces:
                    ins, outs = clip(R1, Z1, a, cid(k), ops)
                    if ins is not None:
                        g2 = g.copy(); g2[k] = True; new.append((ins[0], ins[1], g2))
                    if outs is not None:
                        new.append((outs[0], outs[1], g))
                pieces = new
            allm2 = allm.copy()
            for k in range(H):
                allm2 |= bit(cid(k))
            for R1, Z1, g in pieces:
                A2 = A.copy(); A2[:, :t] = B; A2[:, t] = Wi; A2[~g] = 0.0
                nxt.append((R1, Z1, A2, allm2))
        cells = nxt
        counts.append(len(cells))
    if stats is not None:
        stats["cells_per_step"] = counts
    return cells


def outcome_pieces(cell, model, ops, base_id):
    """Split one final cell into convex outcome pieces. Returns (correct, wrong), lists of (R, Z).
    For each argmax region j: count c = #{k: x_k > x_j} by splitting on x_j - x_k; c == 1 is correct;
    a branch is closed as wrong as soon as c == 2 (it stays wrong), and c == 0 at the end is wrong."""
    R, Z, A, _ = cell
    Wo = model[2]; T = Wo.shape[0]
    L = Wo @ A; ops.add(2 * T * A.shape[0] * T, "final")
    dead = not A.any()
    good = []; bad = []
    for j in range(T):
        if dead:
            if j != 0:
                continue
            Q = (R, Z)          # all h_T = 0: logits tie, argmax returns index 0
        else:
            Q = (R, Z)
            for k in range(T):
                if k == j:
                    continue
                Q, _ = clip(Q[0], Q[1], L[j] - L[k], base_id + k, ops, want=(True, False)); ops.add(T, "final")
                if Q is None:
                    break
            if Q is None:
                continue
        S = [(Q, 0)]
        for k in range(T):
            if k == j:
                continue
            a = np.zeros(T); a[j] = 1.0; a[k] = -1.0
            nS = []
            for (P, c) in S:
                ins, outs = clip(P[0], P[1], a, base_id + T + k, ops)
                if ins is not None:
                    nS.append((ins, c))
                if outs is not None:
                    if c + 1 >= 2:
                        bad.append(outs)
                    else:
                        nS.append((outs, c + 1))
            S = nS
            if not S:
                break
        for P, c in S:
            (good if c == 1 else bad).append(P)
    return good, bad


# ------------------------------------------------------------------ triangulation
def _rank(M, ops):
    ops.add(2 * M.shape[0] * M.shape[1] ** 2 + 2 * M.shape[1] ** 3, "tri")
    return np.linalg.matrix_rank(M, tol=1e-9)


def triangulate(R, Z, ops):
    """Pulling triangulation of the pointed cone with rays R and tight bitsets Z. Returns list of index tuples."""
    k, d = R.shape
    # unpack bitsets to a boolean incidence matrix (k, ncons)
    inc = np.unpackbits(Z.view(np.uint8), axis=1, bitorder="little").astype(bool)
    used = np.nonzero(inc.any(0))[0]
    inc = inc[:, used]
    out = []

    def rec(F, f):
        if len(F) == f:
            out.append(F); return
        v = F[0]
        sub = inc[list(F)]
        seen = set()
        for c in range(sub.shape[1]):
            col = sub[:, c]
            if col[0] or col.all() or not col.any():
                continue
            G = tuple(F[i] for i in np.nonzero(col)[0])
            if len(G) < f - 1 or G in seen:
                continue
            seen.add(G)
            if _rank(R[list(G)], ops) != f - 1:
                continue
            n0 = len(out)
            rec(G, f - 1)
            for i in range(n0, len(out)):
                out[i] = (v,) + out[i]

    rec(tuple(range(k)), d)
    return out


# ------------------------------------------------------------------ orthant probabilities
def _corr(V, ops):
    """Simplicial cones with rays as rows of V (S,T,T) -> correlation R of Y = N x (cone = {N x >= 0})."""
    S, T, _ = V.shape
    N = np.linalg.inv(np.transpose(V, (0, 2, 1)))   # cone{v_i} = {x : (V^T)^{-1} x >= 0}
    C = N @ np.transpose(N, (0, 2, 1))
    s = 1 / np.sqrt(np.einsum("sii->si", C))
    ops.add(S * (2 * T ** 3 + 2 * T ** 3 + T * TRANS + 2 * T * T), "measure")
    return C * s[:, :, None] * s[:, None, :]


def _p2(r):
    return 0.25 + np.arcsin(np.clip(r, -1, 1)) / (2 * np.pi)


def _p3(R):
    return 0.125 + (np.arcsin(np.clip(R[..., 0, 1], -1, 1)) + np.arcsin(np.clip(R[..., 0, 2], -1, 1))
                    + np.arcsin(np.clip(R[..., 1, 2], -1, 1))) / (4 * np.pi)


def _ts_nodes(level, umax=3.2):
    h = 2.0 ** -level
    u = np.arange(-int(umax / h), int(umax / h) + 1) * h
    q = np.exp(-np.pi * np.sinh(u))                 # t = 1/(1+q)
    t = 1 / (1 + q)
    w = np.pi * np.cosh(u) * q / (1 + q) ** 2      # dt/du
    return u, t, w


def _cond_pairs(T):
    return [(i, j) for i in range(T) for j in range(i + 1, T)]


def _plackett_integrand(R, t, ops, inner_level=None):
    """dP_n/dt along R(t) = I + t(R - I) for a batch R (S,n,n) at scalar t."""
    S, n, _ = R.shape
    Rt = R * t; idx = np.arange(n); Rt[:, idx, idx] = 1.0
    tot = np.zeros(S)
    for i, j in _cond_pairs(n):
        rho = Rt[:, i, j]
        K = [k for k in range(n) if k not in (i, j)]
        Ri = Rt[:, K][:, :, [i, j]]                                  # (S, n-2, 2)
        det = np.maximum(1 - rho ** 2, 1e-300)
        inv = np.stack([np.stack([np.ones(S), -rho], -1), np.stack([-rho, np.ones(S)], -1)], -2) / det[:, None, None]
        Sc = Rt[:, K][:, :, K] - Ri @ inv @ np.transpose(Ri, (0, 2, 1))
        d = 1 / np.sqrt(np.clip(np.einsum("sii->si", Sc), 1e-300, None))
        Rc = Sc * d[:, :, None] * d[:, None, :]
        m = n - 2
        if m == 2:
            pc = _p2(Rc[:, 0, 1])
        elif m == 3:
            pc = _p3(Rc)
        else:
            pc = orthant(Rc, ops, level=inner_level)[0]
        phi = 1 / (2 * np.pi * np.sqrt(det))
        tot += R[:, i, j] * phi * pc
        ops.add(S * (8 + 2 * m * 2 * 2 + 2 * m * m * 2 + m * TRANS + m * m * 2 + 2 * TRANS + 4
                     + (3 * TRANS if m == 3 else TRANS if m == 2 else 0)), "measure")
    return tot


def orthant(R, ops, level=None, tol=1e-13, maxlevel=7):
    """P(Y >= 0), Y ~ N(0, R), batch (S,n,n). Returns (p, error estimate per item)."""
    S, n, _ = R.shape
    if n == 1:
        return np.full(S, 0.5), np.zeros(S)
    if n == 2:
        ops.add(S * (TRANS + 3), "measure"); return _p2(R[:, 0, 1]), np.zeros(S)
    if n == 3:
        ops.add(S * (3 * TRANS + 6), "measure"); return _p3(R), np.zeros(S)
    base = 2.0 ** -n
    # tanh-sinh with level doubling; each level reuses the previous nodes (odd nodes only are new)
    res = np.zeros(S); prev = None; err = np.full(S, np.inf)
    act = np.arange(S)
    lv0 = 3 if level is None else level
    sums = np.zeros(S)
    for lv in range(lv0, (maxlevel if level is None else level) + 1):
        u, t, w = _ts_nodes(lv)
        if lv == lv0:
            sel = np.ones(len(u), bool)
            sums[act] = 0
            hfac = 1.0
        else:
            sel = (np.round(u / 2.0 ** -lv).astype(int) % 2) != 0
            hfac = 0.5
        acc = np.zeros(len(act))
        Ra = R[act]
        for ti, wi in zip(t[sel], w[sel]):
            acc += wi * _plackett_integrand(Ra, ti, ops, inner_level=3)
        h = 2.0 ** -lv
        newsum = hfac * sums[act] + h * acc if lv > lv0 else h * acc
        if lv > lv0:
            err[act] = np.abs(newsum - sums[act])
        sums[act] = newsum
        if level is not None:
            break
        if lv > lv0:
            done = err[act] < tol
            act = act[~done]
            if len(act) == 0:
                break
    err[~np.isfinite(err)] = np.nan
    return base + sums, err
