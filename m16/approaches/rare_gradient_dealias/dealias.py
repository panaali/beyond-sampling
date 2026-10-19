"""T386: K3 line rule with a generalised, fixed (model-free, deterministic) set of lines, and its exact weight-gradient
by T372's boundary terms. A rule is: per v2 region a list of (v2 node, v2 weight) and, per v2 node, a g12 rule
(nodes, weights summing to 1). The K3 rule of T364/T372 is rule("gl20", ("gauss", 3)).
est_grad(model, t0, rule) -> dict(est, grad (432,), C_walk, C_grad (FPE), n_end). Deductive: no random numbers.
Shifted-grid families (the de-aliasing): v2 "mid<N>x<M>" = the average over M shifted N-panel midpoint grids
(node j + (m + 1/2)/M of panel width), g12 "qmid<K>x<M>" = the average over M shifted K-cell quantile grids of the
exact g12 | v2 law (g12 >= 0), each node at the (k + (m + 1/2)/M)/K quantile, weight 1/(K M)."""
import numpy as np
from scipy.special import ndtr

import t372_k3src as K
import t372_lgrad as lgrad
import t372_walk as walk

TMAX = 6.0
G, TC, NT, TRI = K.G, K.TC, K.NT, K.TRI


def v2_rule(spec, lo, hi):
    if spec == "gl20":
        return G._gl(lo, hi, 20)
    n, m = map(int, spec[3:].split("x"))                       # "mid<N>x<M>"
    h = (hi - lo) / n; off = (np.arange(m) + 0.5) / m
    nodes = (lo + h * (np.arange(n)[:, None] + off[None])).ravel()
    return nodes, np.full(len(nodes), h / m)


def _g12_cdf(v2):
    g = np.linspace(0.0, 12.0, 120001); w = G._pdf(v2 + g)
    c = np.concatenate([[0.0], np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(g))])
    return g, c / c[-1]


def g12_rule(spec, v2):
    kind, k = spec
    if kind == "gauss":
        return TC.gauss_g12(v2, 0.0, k)
    kk, m = k                                                   # ("qmid", (K, M))
    q = ((np.arange(kk)[:, None] + (np.arange(m)[None] + 0.5) / m) / kk).ravel()
    g, c = _g12_cdf(v2)
    return np.interp(q, c, g), np.full(len(q), 1.0 / len(q))


def lines(v2, g12spec):
    """Rows (g12 node k, triple i) for one v2 node, as T340c.node_lines (a = 0)."""
    v3s = v2 - K.rare._cond_g23(v2); v4s = float(G.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
    m4 = float(G.mu(np.array([v4s]))[0]); dm4 = float(G.dmu(v4s)[0])
    gn, gw = g12_rule(g12spec, v2); xs, dxs, om = [], [], []
    for g12, w in zip(gn, gw):
        x, dx = G.inputs(v2 + g12, v2, v2, v2 - delta, m4 + dm4 * (v2 - delta - v4s), 0.0, -1.0, -1.0, -dm4)
        xs.append(x); dxs.append(dx); om.append(np.full(NT, w))
    return np.concatenate(xs), np.concatenate(dxs), np.concatenate(om)


def est_grad(model, t0, v2spec="gl20", g12spec=("gauss", 3), chunk_rows=45000, want_grad=True):
    model = tuple(np.asarray(w, float) for w in model)
    est = 0.0; grad = np.zeros(sum(w.size for w in model)); Cw = 0.0; Cg = 0.0; n_end = 0
    for name, lo, hi in G.REGIONS:
        vn, vw = v2_rule(v2spec, lo, hi)
        wgt = 90 * G._pdf(vn) * (1.0 - ndtr(vn)) * vw            # = tab["dens"] * tab["blind12"] * wts
        per = len(g12_rule(g12spec, vn[0])[0]) * NT; step = max(1, chunk_rows // per)
        for s in range(0, len(vn), step):
            js = np.arange(s, min(s + step, len(vn))); L = [lines(vn[j], g12spec) for j in js]
            x = np.concatenate([l[0] for l in L]); dx = np.concatenate([l[1] for l in L]); om = np.concatenate([l[2] for l in L])
            nrow = len(x); node_of_row = np.repeat(np.arange(len(js)), per); nodes = vn[js]
            k = np.tile(TRI[:, 1], nrow // NT); k3 = np.tile(TRI[:, 2], nrow // NT)
            pieces, _, f = K.T354.walk_seg(model, x, dx, k, k3, t0, TMAX); Cw += f
            W, f = K.rare.mass(pieces, nrow, node_of_row, nodes, t0); Cw += f
            est += float((W * om[:, None]).sum(0) @ wgt[js]) / NT; Cw += nrow * len(js) * 3
            if not want_grad:
                continue
            rowfac = om * wgt[js][node_of_row] / NT
            R = np.concatenate([p[0] for p in pieces]); A = np.concatenate([p[1] for p in pieces]); B = np.concatenate([p[2] for p in pieces])
            A = np.maximum(A, t0); m = B > A; R, A, B = walk.merge(R[m], A[m], B[m])
            v2 = nodes[node_of_row]
            for e, keep in ((A, A > t0 + 1e-9), (B, B < TMAX - 1e-9)):
                r, ee = R[keep], e[keep]
                if not len(r):
                    continue
                Gm, J, S, f = lgrad.endpoint_grads(model, x, dx, k3, r, ee); Cg += f
                fd = 8 * ndtr(v2[r] - ee) ** 7 * np.exp(-(v2[r] - ee) ** 2 / 2) / np.sqrt(2 * np.pi)
                grad += ((rowfac[r] * fd)[:, None] * Gm).sum(0); n_end += len(r)
    return dict(est=est, grad=grad, C_walk=Cw / K.FPE, C_grad=Cg / K.FPE, n_end=n_end)
