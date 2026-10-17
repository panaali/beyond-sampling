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


_BITS = np.zeros((64 * NW, NW), np.uint64)
for _c in range(64 * NW):
    _BITS[_c, _c // 64] = np.uint64(1) << np.uint64(_c % 64)


def bit(c):
    return _BITS[c]


def clip(R, Z, a, cid, ops, want=(True, True)):
    """Split cone (rays R (k,d) unit rows, tight bitsets Z (k,NW)) by a.x >= 0 (constraint id cid).
    Returns (inside, outside), each (R, Z) or None if that side is not full-dimensional."""
    k, d = R.shape
    a = a / math.sqrt(a @ a)
    v = R @ a; ops.add(3 * d + TRANS + 2 * k * d, "clip")
    pos = v > TOL; neg = v < -TOL
    npos = pos.sum(); nneg = neg.sum()
    if npos + nneg == k and (npos == 0 or nneg == 0):      # fast path: no split, no tight rays
        if nneg == 0:
            return ((R, Z) if want[0] else None), None
        return None, ((R, Z) if want[1] else None)
    zer = ~(pos | neg)
    b = _BITS[cid]
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
def _children(R, Z, A, allm, t, model, ops):
    """Cut the lifted cell C x R by the 16 step-(t+1) gate hyperplanes. Returns the child cells."""
    Wi, Wh, Wo = model
    H = len(Wi)
    order = np.argsort(-np.abs(Wi))       # lift on the unit with the largest |Wi| (well-conditioned)
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
    out = []
    for R1, Z1, g in pieces:
        A2 = A.copy(); A2[:, :t] = B; A2[:, t] = Wi; A2[~g] = 0.0
        out.append((R1, Z1, A2, allm2))
    return out


def iter_cells(model, ops, counts):
    """Depth-first generator of final gate cells (R, Z, A_T, allmask) in R^T; counts[t] = cells at step t+1."""
    Wi, Wh, Wo = model
    H = len(Wi); T = Wo.shape[0]
    roots = []
    for s in (1.0, -1.0):                 # step 1: x_1 > 0 or < 0; gates fixed
        g = (s * Wi) > 0
        A = np.zeros((H, T)); A[:, 0] = np.where(g, Wi, 0.0)
        roots.append((np.array([[s]]), np.zeros((1, NW), np.uint64), A, bit(0)))
    counts[0] += 2
    stack = [(c, 1) for c in roots[::-1]]
    while stack:
        c, t = stack.pop()
        if t == T:
            yield c; continue
        ch = _children(*c, t, model, ops)
        counts[t] += len(ch)
        stack += [(x, t + 1) for x in ch[::-1]]


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


# ------------------------------------------------------------------ simplicial cone measure by the solid-angle integral
# For unit rays v_1..v_T (rows of V), P(Z in cone) = |det V| Gamma(T/2)/(2 pi^(T/2)) * int_Delta |V^T mu|^(-T) d mu
# over the standard (T-1)-simplex of barycentric weights mu (Lebesgue measure in mu_1..mu_{T-1}). The integrand is
# smooth and bounded on narrow cones (thin cones are not ill-conditioned: the smallness sits in |det V|).
# Grundmann-Moller rules of degree 2s+1, error = |Q_s - Q_{s-1}|, adaptive bisection of the longest edge.
_GM = {}


def gm_rule(n, s):
    """Grundmann-Moller rule on the standard n-simplex (volume 1/n!): barycentric nodes (N, n+1), weights (N,)."""
    if (n, s) in _GM:
        return _GM[n, s]
    from itertools import product
    from math import factorial
    d = 2 * s + 1
    nodes = []; w = []
    for i in range(s + 1):
        m = s - i
        c = (-1) ** i * 2.0 ** (-2 * s) * (d + n - 2 * i) ** d / (factorial(i) * factorial(d + n - i))
        for beta in product(range(m + 1), repeat=n + 1):
            if sum(beta) != m:
                continue
            nodes.append([(2 * b + 1) / (d + n - 2 * i) for b in beta]); w.append(c)
    _GM[n, s] = (np.array(nodes), np.array(w))
    return _GM[n, s]


def _gm_eval(V, s, ops):
    S, T, _ = V.shape
    mu, w = gm_rule(T - 1, s)
    X = np.einsum("nt,std->snd", mu, V)
    q = (X * X).sum(-1)
    ops.add(S * len(w) * (2 * T * T + 2 * T + TRANS + 2), "measure")
    return (q ** (-T / 2)) @ w


def cone_measure(V, ops, rtol=1e-6, atol=1e-16, s=3, maxdepth=40, chunk=4096):
    """Gaussian measure of simplicial cones (rows of V[i] = unit rays). Returns (p, err) per input simplex.
    Depth-first over a bounded work stack (memory stays O(chunk x maxdepth))."""
    S, T, _ = V.shape
    if S == 0:
        return np.zeros(0), np.zeros(0)
    if T == 2:      # exact: angle / 2 pi
        c = (V[:, 0] * V[:, 1]).sum(1); d = V[:, 0, 0] * V[:, 1, 1] - V[:, 0, 1] * V[:, 1, 0]
        ops.add(S * (6 + TRANS), "measure")
        return np.abs(np.arctan2(np.abs(d), c)) / (2 * np.pi), np.zeros(S)
    if T == 3:      # exact: Van Oosterom-Strackee solid angle / 4 pi
        a, b, c = V[:, 0], V[:, 1], V[:, 2]
        num = np.abs(np.einsum("si,si->s", a, np.cross(b, c)))
        den = 1 + (a * b).sum(1) + (b * c).sum(1) + (a * c).sum(1)
        ops.add(S * (9 + 5 + 18 + 3 + TRANS + 2), "measure")
        om = 2 * np.arctan2(num, den)
        return om / (4 * np.pi), np.zeros(S)
    from math import gamma, pi
    cst = gamma(T / 2) / (2 * pi ** (T / 2))
    p = np.zeros(S); e = np.zeros(S)
    stack = [(V[i:i + chunk], np.arange(i, min(i + chunk, S)), 0) for i in range(0, S, chunk)][::-1]
    while stack:
        W, idx, depth = stack.pop()
        if len(W) > chunk:
            stack += [(W[i:i + chunk], idx[i:i + chunk], depth) for i in range(0, len(W), chunk)]
            continue
        det = np.abs(np.linalg.det(W)); ops.add(len(W) * (2 * T ** 3 // 3 + T), "measure")
        hi = _gm_eval(W, s, ops); lo = _gm_eval(W, s - 1, ops)
        val = cst * det * hi; err = cst * det * np.abs(hi - lo)
        ok = (err <= rtol * val + atol) | (depth >= maxdepth)
        np.add.at(p, idx[ok], val[ok]); np.add.at(e, idx[ok], err[ok])
        if ok.all():
            continue
        Wb = W[~ok]; ib = idx[~ok]
        # bisect the longest edge (smallest ray dot product)
        G = Wb @ np.transpose(Wb, (0, 2, 1)); ops.add(len(Wb) * (2 * T ** 3 + 3 * T + TRANS), "measure")
        G[:, np.arange(T), np.arange(T)] = 2.0
        flat = G.reshape(len(Wb), -1).argmin(1); a, b = flat // T, flat % T
        r = np.arange(len(Wb))
        mid = Wb[r, a] + Wb[r, b]; mid /= np.linalg.norm(mid, axis=1)[:, None]
        W1 = Wb.copy(); W1[r, a] = mid
        W2 = Wb.copy(); W2[r, b] = mid
        stack.append((np.concatenate([W1, W2]), np.concatenate([ib, ib]), depth + 1))
    return p, e
