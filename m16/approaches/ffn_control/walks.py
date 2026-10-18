"""Exact 1-D walks for bias-free ReLU networks (MLPs and the M_16,10 RNN), with FLOP accounting.

Along a line x(s) = b + s*d every bias-free ReLU network is exactly piecewise linear in s. We walk the pieces
left to right (the same algorithm as m16/lineint.py, generalised): at each piece we evaluate all pre-activations
and the logits with their slopes (one "forward-with-slope", charged at 2 forward passes of FLOPs), step to the
next gate breakpoint, and inside the piece follow the argmax envelope of the affine logits. The target
(2nd-argmax of x(s)) is piecewise constant with breakpoints where two coordinates of x(s) cross. The correct set
is a finite union of intervals, integrated exactly against a weight:

- 'coord'  : d = e_j, b_j = 0, s ~ N(0,1)          -> exact P(correct | x_{-j})   (conditional MC)
- 'circle' : x(theta) = cos(theta) u + sin(theta) v, theta uniform. By homogeneity the signs at x(theta) equal
             those at u + tan(theta) v, so a great circle is two lines (u + t v and -u - t v) with weight
             dt / (pi (1 + t^2)), each half weighted 1/2.                         -> exact circle fraction

Both are unbiased for the accuracy (x(theta) ~ N(0, I) for every fixed theta; x with x_j ~ N(0,1)).
Each walk returns (value, flops).
"""
import numpy as np
from scipy.special import ndtr

T = 10
ERF_FLOPS = 20          # charge per ndtr/arctan evaluation
ENV_FLOPS = 6 * T       # charge per argmax-envelope step (10 values, 10 slopes, ratios, min)


class MLP:
    def __init__(self, weights):
        self.W = [np.asarray(w, dtype=np.float64) for w in weights]
        macs = sum(w.size for w in self.W)
        self.fwd_flops = 2 * macs + sum(w.shape[0] for w in self.W[:-1])

    def pre_and_logits(self, x, d):
        """x, d: (T,). Returns pre-activations, their slopes, logits, logit slopes (all at x, direction d)."""
        h, dh, zs, dzs = x, d, [], []
        for w in self.W[:-1]:
            z, dz = w @ h, w @ dh
            zs.append(z); dzs.append(dz)
            g = z > 0
            h, dh = np.where(g, z, 0.0), np.where(g, dz, 0.0)
        return np.concatenate(zs), np.concatenate(dzs), self.W[-1] @ h, self.W[-1] @ dh

    def forward(self, x):
        h = x
        for w in self.W[:-1]:
            h = np.maximum(h @ w.T, 0)
        return h @ self.W[-1].T


class RNN:
    def __init__(self, Wi, Wh, Wo):
        self.Wi, self.Wh, self.Wo = (np.asarray(a, dtype=np.float64) for a in (Wi, Wh, Wo))
        H = len(self.Wi)
        self.fwd_flops = 2 * (T * (H + H * H) + T * H) + T * H

    def pre_and_logits(self, x, d):
        H = len(self.Wi)
        h, dh, zs, dzs = np.zeros(H), np.zeros(H), [], []
        for t in range(T):
            z, dz = self.Wh @ h + self.Wi * x[t], self.Wh @ dh + self.Wi * d[t]
            zs.append(z); dzs.append(dz)
            g = z > 0
            h, dh = np.where(g, z, 0.0), np.where(g, dz, 0.0)
        return np.concatenate(zs), np.concatenate(dzs), self.Wo @ h, self.Wo @ dh

    def forward(self, x):
        h = np.zeros((len(x), len(self.Wi)))
        for t in range(T):
            h = np.maximum(h @ self.Wh.T + x[:, t:t + 1] * self.Wi[None], 0)
        return h @ self.Wo.T


def _target_breaks(b, d, lo, hi):
    """Sorted s where two coordinates of b + s d cross, inside (lo, hi), and the target on each segment."""
    i, j = np.triu_indices(T, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = -(b[i] - b[j]) / (d[i] - d[j])
    r = np.sort(r[np.isfinite(r) & (r > lo) & (r < hi)])
    edges = np.concatenate([[lo], r, [hi]])
    mids = 0.5 * (edges[:-1] + edges[1:])
    mids[0] = (r[0] - 1.0) if r.size else 0.0          # avoid +-1e8 midpoints at the ends
    mids[-1] = (r[-1] + 1.0) if r.size else 0.0
    tg = np.argsort(b[None] + mids[:, None] * d[None], 1)[:, -2]
    flops = 3 * len(i) + 4 * T * len(mids)
    return edges, tg, flops


def walk(model, b, d, lo, hi, cdf, max_pieces=100000):
    """Exact weight of {s in [lo, hi]: argmax logits(b + s d) == 2nd-argmax(b + s d)}.

    Returns (mass, total flops, pieces, aux flops), aux = target breakpoints + envelope + erf work."""
    edges, tg, aux = _target_breaks(b, d, lo, hi)      # aux: FLOPs outside the per-piece forward-with-slope
    s, mass, pieces, eps, flops = lo, 0.0, 0, 1e-9, 0
    ti = 0
    while s < hi and pieces < max_pieces:
        sp = s + eps * max(1.0, abs(s))
        z, dz, lg, dlg = model.pre_and_logits(b + sp * d, d)
        pieces += 1
        flops += 2 * model.fwd_flops + 3 * len(z)
        z, lg = z - dz * (sp - s), lg - dlg * (sp - s)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = s - z / dz
        r = r[(r > s + 1e-12 * max(1.0, abs(s))) & np.isfinite(r)]
        nxt = min(hi, r.min()) if r.size else hi
        a = s
        while a < nxt:
            while edges[ti + 1] <= a:
                ti += 1
            val = lg + dlg * (a - s)
            k = int(np.argmax(val + dlg * 1e-7 * max(1.0, abs(a))))
            with np.errstate(divide="ignore", invalid="ignore"):
                c = a + (val[k] - val) / (dlg - dlg[k])
            c = c[(c > a + 1e-13 * max(1.0, abs(a))) & (dlg > dlg[k])]
            bnd = min(nxt, c.min()) if c.size else nxt
            aux += ENV_FLOPS
            # target segments inside [a, bnd]
            u, tj = a, ti
            while u < bnd:
                v = min(bnd, edges[tj + 1])
                if tg[tj] == k and v > u:
                    mass += cdf(v) - cdf(u)
                    aux += 2 * ERF_FLOPS
                u = v
                if v == edges[tj + 1]:
                    tj += 1
            a = bnd
        s = nxt
    return mass, flops + aux, pieces, aux


GAUSS = (-9.0, 9.0, ndtr)
CAUCHY = (-1e8, 1e8, lambda t: np.arctan(t) / np.pi)


def coord_walk(model, rng):
    b = rng.standard_normal(T)
    j = int(rng.integers(T))
    b[j] = 0.0
    d = np.zeros(T); d[j] = 1.0
    return walk(model, b, d, *GAUSS)


def circle_walk(model, rng):
    u, v = rng.standard_normal(T), rng.standard_normal(T)
    m1, f1, p1, a1 = walk(model, u, v, *CAUCHY)
    m2, f2, p2, a2 = walk(model, -u, -v, *CAUCHY)
    return 0.5 * (m1 + m2), f1 + f2, p1 + p2, a1 + a2
