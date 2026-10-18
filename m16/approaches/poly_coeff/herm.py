"""Multivariate probabilists' Hermite algebra on total degree <= d (model-free tables).

A polynomial is a coefficient vector over the multi-indices alpha (|alpha| <= d, n variables):
    p(x) = sum_alpha c_alpha He_alpha(x),  He_alpha(x) = prod_i He_{alpha_i}(x_i).
Under x ~ N(0, I): E[He_alpha He_beta] = alpha! delta, so mean = c_0 and E[p^2] = sum c_alpha^2 alpha!.
Products use the linearisation He_a He_b = sum_r C(a,r) C(b,r) r! He_{a+b-2r}, truncated to degree <= d
(the "truncated algebra"). Tables are model-free and built once; their construction is not charged.
"""
import itertools
from math import comb, factorial

import numpy as np


def multi_indices(n, d):
    out = [()]
    out = [a for a in itertools.product(range(d + 1), repeat=n) if sum(a) <= d]
    out.sort(key=lambda a: (sum(a), tuple(-v for v in a)))
    return out


class Algebra:
    def __init__(self, n, d):
        self.n, self.d = n, d
        self.idx = multi_indices(n, d)
        self.pos = {a: i for i, a in enumerate(self.idx)}
        self.M = len(self.idx)
        self.norm2 = np.array([np.prod([factorial(v) for v in a]) for a in self.idx], float)  # alpha!
        self.deg = np.array([sum(a) for a in self.idx])
        # variable-of-each-index mask: which variables an index uses (for the shared-innovation sparsity)
        self.maxvar = np.array([max([i for i, v in enumerate(a) if v] or [-1]) for a in self.idx])
        self._prod = None

    def var(self, i):
        c = np.zeros(self.M); e = [0] * self.n; e[i] = 1; c[self.pos[tuple(e)]] = 1.0; return c

    def const(self, v=1.0):
        c = np.zeros(self.M); c[0] = v; return c

    def product_table(self):
        """Sparse structure constants (I, J, K, w): He_I He_J -> sum w He_K, truncated to |K| <= d."""
        if self._prod is not None:
            return self._prod
        I, J, K, W = [], [], [], []
        for i, a in enumerate(self.idx):
            for j, b in enumerate(self.idx):
                # per-coordinate expansions
                terms = [((), 1.0)]
                ok = True
                for ai, bi in zip(a, b):
                    new = []
                    for g, w in terms:
                        for r in range(min(ai, bi) + 1):
                            new.append((g + (ai + bi - 2 * r,), w * comb(ai, r) * comb(bi, r) * factorial(r)))
                    terms = new
                for g, w in terms:
                    if sum(g) <= self.d:
                        I.append(i); J.append(j); K.append(self.pos[g]); W.append(w)
        self._prod = (np.array(I), np.array(J), np.array(K), np.array(W))
        return self._prod

    def mul(self, p, q, active=None):
        """Truncated product. `active` (int) restricts to indices using only variables < active
        (shared innovation: at step t only x_1..x_t exist). Returns (coeffs, flops)."""
        I, J, K, W = self.product_table() if active is None else self.active_table(active)
        out = np.zeros(self.M)
        np.add.at(out, K, W * p[I] * q[J])
        return out, 3 * len(W)          # 2 mults + 1 add per structure constant

    def active_table(self, t):
        if not hasattr(self, "_act"):
            self._act = {}
        if t not in self._act:
            I, J, K, W = self.product_table()
            m = (self.maxvar[I] < t) & (self.maxvar[J] < t)
            self._act[t] = (I[m], J[m], K[m], W[m])
        return self._act[t]

    def mean(self, p):
        return p[0]

    def var_(self, p):
        return float((p[1:] ** 2 * self.norm2[1:]).sum())

    def eval(self, p, x):
        """Evaluate at points x (N, n) (diagnostics only)."""
        N = len(x)
        He = np.ones((self.n, self.d + 1, N))
        for i in range(self.n):
            He[i, 1] = x[:, i]
            for k in range(2, self.d + 1):
                He[i, k] = x[:, i] * He[i, k - 1] - (k - 1) * He[i, k - 2]
        out = np.zeros(N)
        for c, a in zip(p, self.idx):
            if c == 0:
                continue
            v = np.full(N, c)
            for i, ai in enumerate(a):
                if ai:
                    v = v * He[i, ai]
            out += v
        return out


if __name__ == "__main__":
    for d in (1, 2, 3, 4):
        A = Algebra(10, d)
        I, J, K, W = A.product_table()
        per_t = [len(A.active_table(t)[0]) for t in range(1, 11)]
        print(f"d={d} M={A.M} nnz(full)={len(W)} nnz by active t={per_t}")
