"""T426: T398b machinery (event_cone bnb.py + cone_count.split_fast + relax2.prune_stage, branch claude/t373-event-cone
59c5ddf1..292da037) generalised to any hidden width H and sequence length T. Logic is copied line for line; the only
changes are (a) H and D = T are read from the loaded model instead of the constants 16 / 10, (b) the model is set by
set_model() instead of m16/common.py. Event E1(t0) = {x_p1 >= x_p2 >= x_p3 + t0 >= every low + t0, argmax L = p3}.
Model: h_0 = 0, h_t = relu(Wh h_{t-1} + Wi x_{t-1}), L = Wo h_T, no biases."""
import os

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(HERE, "..", "..", "..", "data", "t426", "weights")
TOL = 1e-9
EPS = 1e-7
DEPTH = 6
DELTA = 1 / 3
Wi = Wh = Wo = None
H = D = None


def set_model(tag):
    """tag: registry tag (m_H_T_streamS) or 'orig' (ARC's M_16,10, alg_zoo.example_2nd_argmax)."""
    global Wi, Wh, Wo, H, D
    if tag == "orig":
        import sys
        sys.path.insert(0, os.path.join(HERE, "..", ".."))
        import common
        Wi, Wh, Wo = common.Wi.copy(), common.Wh.copy(), common.Wo.copy()
    else:
        d = np.load(os.path.join(WDIR, f"{tag}.npz"))
        Wi, Wh, Wo = (d[k].astype(np.float64) for k in ("Wi", "Wh", "Wo"))
    H, D = len(Wi), Wo.shape[0]
    return Wi, Wh, Wo


def run(x):
    hs = np.zeros((len(x), x.shape[1] + 1, H))
    for t in range(x.shape[1]):
        hs[:, t + 1] = np.maximum(hs[:, t] @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return hs


# ---- exact prefix cells (cone_count.py) ----
def cone(tri):
    p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]; e = np.eye(D)
    return np.array([e[p1] - e[p2], e[p2] - e[p3]] + [e[p3] - e[l] for l in lows])


def interior(tri):
    x = np.zeros(D); p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]
    x[p1], x[p2], x[p3] = .9, .6, .3
    x[lows] = -np.linspace(.05, .9, len(lows))
    return x


def lp(c, G, b, box, sign):
    r = linprog(-sign * c, A_ub=-G, b_ub=-b, bounds=[(-box, box)] * D, method="highs")
    return -sign * r.fun if r.status == 0 else np.nan, (r.x if r.status == 0 else None)


def split_plain(G, b, c, box):
    nc = np.linalg.norm(c)
    if nc == 0:
        return "off", None
    hi, _ = lp(c / nc, G, b, box, 1); lo, _ = lp(c / nc, G, b, box, -1)
    if hi <= TOL:
        return "off", None
    if lo >= -TOL:
        return "on", None
    return "split", None


def split_fast(G, xi, c):
    nc = np.linalg.norm(c)
    if nc == 0:
        return "off", None, None
    u = c / nc; s = u @ xi; z = np.zeros(len(G))
    if s > TOL:
        lo, xs = lp(u, G, z, 1.0, -1)
        if lo >= -TOL:
            return "on", None, None
        lam = (-lo / 2) / (s - lo)
        return "split", xi, lam * xi + (1 - lam) * xs
    if s < -TOL:
        hi, xs = lp(u, G, z, 1.0, 1)
        if hi <= TOL:
            return "off", None, None
        lam = (hi / 2) / (hi - s)
        return "split", lam * xi + (1 - lam) * xs, xi
    d, _ = split_plain(G, z, c, 1.0)
    if d != "split":
        return d, None, None
    _, xp = lp(u, G, z, 1.0, 1); _, xm = lp(u, G, z, 1.0, -1)
    return "split", .5 * (xi + xp), .5 * (xi + xm)


def children(cell, t):
    """One exact splitting step of one cell (bnb.step). cell = (G, xi, A, hist)."""
    G, xi, A, hist = cell
    Z = Wh @ A; Z[:, t] += Wi
    pieces = [(G, xi, [])]
    for j in range(H):
        nxt = []
        for Gp, xp, g in pieces:
            d, xpos, xneg = split_fast(Gp, xp, Z[j])
            if d == "off":
                nxt.append((Gp, xp, g + [0]))
            elif d == "on":
                nxt.append((Gp, xp, g + [1]))
            else:
                nxt.append((np.vstack([Gp, Z[j]]), xpos, g + [1])); nxt.append((np.vstack([Gp, -Z[j]]), xneg, g + [0]))
        pieces = nxt
    out = []
    for Gp, xp, g in pieces:
        g = np.array(g, float)
        out.append((Gp, xp, g[:, None] * Z, hist + np.packbits(g > 0).tobytes()))
    return out


def gates_of(hist, k, fix):
    g = -np.ones((D, H), np.int8)
    if k:
        g[:k] = np.unpackbits(np.frombuffer(hist, np.uint8)).reshape(k, -1)[:, :H]
    for (s, j), v in fix.items():
        g[k + s, j] = v
    return g


def prefix_keys(hs, k):
    """Packed k-step gate histories of forward passes hs (N, T+1, H), as in children()."""
    return [r.tobytes() for r in np.packbits(hs[:, 1:k + 1] > 0, axis=2).reshape(len(hs), -1)]


# ---- pruning (relax2.py) ----
def solve(c, P, bounds):
    ub, bu, eq, be = P
    return linprog(c, A_ub=np.array(ub) if ub else None, b_ub=np.array(bu) if bu else None,
                   A_eq=np.array(eq) if eq else None, b_eq=np.array(be) if be else None, bounds=bounds, method="highs")


class Relax:
    def __init__(self, G, A, k, tri):
        self.k, self.tri = k, tri; self.nb = D - k; self.nv = D + H * self.nb
        p1, p2, p3 = tri
        self.base_ub, self.base_bu = [], []
        for g in G:
            r = np.zeros(self.nv); r[:D] = -g; self.base_ub.append(r); self.base_bu.append(0.)
        r = np.zeros(self.nv); r[p2] = -1; r[p3] = 1; self.base_ub.append(r); self.base_bu.append(-DELTA)
        self.Z = []
        prev = np.zeros((H, self.nv)); prev[:, :D] = A
        for s in range(self.nb):
            Zs = Wh @ prev; Zs[:, k + s] += Wi; self.Z.append(Zs)
            prev = np.zeros((H, self.nv)); prev[:, D + H * s:D + H * (s + 1)] = np.eye(H)
        L = Wo @ prev
        self.ev_ub = [L[j] - L[p3] for j in range(D) if j != p3]; self.ev_bu = [EPS] * (D - 1)
        self.lo = np.full((self.nb, H), -np.inf); self.hi = np.full((self.nb, H), np.inf)

    def var(self, s, j):
        return D + H * s + j

    def program(self, upto, event, fix=None):
        ub = list(self.base_ub); bu = list(self.base_bu); eq, be = [], []
        bounds = [(-1, 1)] * D + [(0, 0)] * (H * self.nb)
        fix = fix or {}
        for s in range(upto):
            for j in range(H):
                v = self.var(s, j); z = self.Z[s][j]; l, u = self.lo[s, j], self.hi[s, j]; f = fix.get((s, j))
                if u <= 0 or f == 0:
                    bounds[v] = (0, 0)
                    if f == 0:
                        ub.append(z.copy()); bu.append(0.)
                elif l >= 0 or f == 1:
                    r = -z.copy(); r[v] += 1; eq.append(r); be.append(0.); bounds[v] = (0, None)
                    if f == 1:
                        ub.append(-z.copy()); bu.append(0.)
                else:
                    bounds[v] = (0, u)
                    r = z.copy(); r[v] -= 1; ub.append(r); bu.append(0.)
                    r = -u / (u - l) * z; r[v] += 1; ub.append(r); bu.append(-u * l / (u - l))
        if event:
            ub += self.ev_ub; bu += self.ev_bu
        return (ub, bu, eq, be), bounds

    def forward_bounds(self):
        for s in range(self.nb):
            P, bounds = self.program(s, False)
            for j in range(H):
                r1 = solve(self.Z[s][j], P, bounds); r2 = solve(-self.Z[s][j], P, bounds)
                if r1.status == 2 or r2.status == 2:
                    return False
                self.lo[s, j] = r1.fun; self.hi[s, j] = -r2.fun
        return True

    def feasible(self, fix=None):
        P, bounds = self.program(self.nb, True, fix)
        return solve(np.zeros(self.nv), P, bounds).status != 2

    def event_tighten(self):
        P, bounds = self.program(self.nb, True)
        lo, hi = self.lo.copy(), self.hi.copy()
        for s in range(self.nb):
            for j in range(H):
                if self.lo[s, j] >= 0 or self.hi[s, j] <= 0:
                    continue
                r1 = solve(self.Z[s][j], P, bounds); r2 = solve(-self.Z[s][j], P, bounds)
                if r1.status == 2 or r2.status == 2:
                    return False
                lo[s, j] = max(lo[s, j], r1.fun); hi[s, j] = min(hi[s, j], -r2.fun)
        self.lo, self.hi = lo, hi
        return True

    def split_order(self):
        c = [(s, -(u * -l) / (u - l), j) for s in range(self.nb) for j in range(H)
             for l, u in [(self.lo[s, j], self.hi[s, j])] if l < 0 < u]
        return [(s, j) for s, _, j in sorted(c)]

    def split_prune(self, order, depth):
        self.leaves = []
        def rec(fix, i):
            if not self.feasible(fix):
                self.leaves.append(dict(fix)); return True
            if i >= min(depth, len(order)):
                return False
            sj = order[i]
            return rec({**fix, sj: 1}, i + 1) and rec({**fix, sj: 0}, i + 1)
        return rec({}, 0)


def prune_stage(G, A, k, tri, depth=DEPTH, cert=None):
    """Stage that prunes the cell (0 norm, 1 base, 2 event-tightened, 3 split) or -1 if it survives."""
    R = Relax(G, A, k, tri)
    if not R.forward_bounds():
        st = 0
    elif not R.feasible():
        st = 1
    elif not R.event_tighten() or not R.feasible():
        st = 2
    elif depth > 0 and R.split_prune(R.split_order(), depth):
        st = 3
    else:
        return -1
    if cert is not None:
        cert.extend(R.leaves if st == 3 else [{}])
    return st
