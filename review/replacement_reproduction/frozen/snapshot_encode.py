"""Interpretable encodings of pre-activations (Phase B) and their reuse in the replacement test (Phase C).

For step t (prefix x_0..x_{t-1}, current input xc = x_{t-1}) the variables are
  m_k   k-th largest prefix value (running top-k), a_k its age (t-1 - position),
  K     number of positive prefix values,
  clock cos/sin(theta * a_k), theta = 2 pi / 8.42 (dominant clock, FINDINGS §2).
Every family is positively homogeneous of degree 1 in x (no intercepts), like the network (no biases).

A family is (cell, base): within each discrete cell (a tuple of ages and/or K), pre_t[j] is a linear function
of the base vector. Cells give the piecewise structure (which inputs are records), bases the magnitudes.
"""
import numpy as np

THETA = 2 * np.pi / 8.42


def prefix_stats(x, t):
    xs = x[:, :t]
    idx = np.argsort(-xs, 1)
    m = np.take_along_axis(xs, idx, 1)
    a = (t - 1) - idx
    K = (xs > 0).sum(1)
    return m, a, K


def _get(m, k):
    return m[:, k] if k < m.shape[1] else np.zeros(len(m))


def _age(a, k, t):
    return a[:, k] if k < a.shape[1] else np.full(len(a), t)   # t = "absent"


# ---- base vectors -------------------------------------------------------------------------------
def base_vector(name, x, t, m, a, K):
    xc = x[:, t - 1]
    if name == "m3":                # x_t and running top-3
        cols = [xc, _get(m, 0), _get(m, 1), _get(m, 2)]
    elif name == "m3clock":         # + clock phasors of the top-3 ages
        cols = [xc, _get(m, 0), _get(m, 1), _get(m, 2)]
        for k in range(3):
            ak = _age(a, k, t)
            cols += [_get(m, k) * np.cos(THETA * ak), _get(m, k) * np.sin(THETA * ak)]
    elif name == "m5":
        cols = [xc] + [_get(m, k) for k in range(5)]
    elif name == "m5rest":          # + positive and negative parts of the remaining inputs (ranks >= 6)
        rest = m[:, 5:] if m.shape[1] > 5 else np.zeros((len(m), 1))
        cols = [xc] + [_get(m, k) for k in range(5)] + [np.maximum(rest, 0).sum(1), np.minimum(rest, 0).sum(1)]
    elif name == "all":             # every prefix value by rank (the exact linear form given rank-only gates)
        cols = [xc] + [_get(m, k) for k in range(10)]
    else:
        raise ValueError(name)
    return np.stack(cols, 1)


def cell_index(name, t, a, K, m):
    """Integer cell id per row."""
    T1 = t + 1
    if name == "none":
        return np.zeros(len(a), int)
    a1, a2, a3, a4 = (_age(a, k, t) for k in range(4))
    if name == "a1":
        return a1
    if name == "a12":
        return a1 * T1 + a2
    if name == "a123":
        return (a1 * T1 + a2) * T1 + a3
    if name == "a123K":
        return ((a1 * T1 + a2) * T1 + a3) * 4 + np.minimum(K, 3)
    if name == "a1234K":
        return (((a1 * T1 + a2) * T1 + a3) * T1 + a4) * 4 + np.minimum(K, 3)
    if name == "a1234Ks":           # + sign of m5 (does the 5th value pass the ReLU)
        s = (_get(m, 4) > 0).astype(int)
        return ((((a1 * T1 + a2) * T1 + a3) * T1 + a4) * 4 + np.minimum(K, 3)) * 2 + s
    raise ValueError(name)


FAMILIES = [  # (label, cell, base) in increasing complexity
    ("L: x_t,m1..m3", "none", "m3"),
    ("C: +clock(age1..3)", "none", "m3clock"),
    ("A1: age1 x [x_t,m1..m3]", "a1", "m3"),
    ("A12: (age1,age2) x [x_t,m1..m3]", "a12", "m3"),
    ("A123: (age1..3) x [x_t,m1..m5]", "a123", "m5"),
    ("A123K: (age1..3,K) x [x_t,m1..m5,rest+-]", "a123K", "m5rest"),
    ("A1234K: (age1..4,K) x [x_t,m1..m5,rest+-]", "a1234K", "m5rest"),
    ("A1234K-all: (age1..4,K) x [x_t,m1..m10]", "a1234K", "all"),
]


class CellLinear:
    """Per-cell ridge regression of Y (N, q) on base vectors V (N, d)."""

    def __init__(self, lam=1e-6):
        self.lam = lam

    def fit(self, cell, V, Y):
        ids, inv = np.unique(cell, return_inverse=True)
        C, d, q = len(ids), V.shape[1], Y.shape[1]
        XtX = np.zeros((C, d, d))
        XtY = np.zeros((C, d, q))
        for p in range(d):
            for r in range(p, d):
                XtX[:, p, r] = XtX[:, r, p] = np.bincount(inv, V[:, p] * V[:, r], C)
            for j in range(q):
                XtY[:, p, j] = np.bincount(inv, V[:, p] * Y[:, j], C)
        G, B = XtX.sum(0), XtY.sum(0)          # pooled fallback for unseen cells (normal equations)
        self.pooled = np.linalg.solve(G + 1e-9 * np.trace(G) / d * np.eye(d), B)
        scale = np.trace(XtX, axis1=1, axis2=2)[:, None, None] / d + 1e-12
        XtX += self.lam * scale * np.eye(d)[None]
        self.coef = np.linalg.solve(XtX, XtY)           # (C, d, q)
        self.ids = ids
        self.count = np.bincount(inv, minlength=C)
        return self

    def predict(self, cell, V):
        pos = np.searchsorted(self.ids, cell)
        pos = np.clip(pos, 0, len(self.ids) - 1)
        seen = self.ids[pos] == cell
        out = V @ self.pooled
        rows = np.where(seen)[0]
        for i in range(0, len(rows), 20_000):          # chunked to bound memory
            r = rows[i:i + 20_000]
            out[r] = np.einsum("nd,ndq->nq", V[r], self.coef[pos[r]])
        self.unseen_frac = 1 - seen.mean()
        return out


def design(fam, x, t):
    label, cname, bname = fam
    m, a, K = prefix_stats(x, t)
    return cell_index(cname, t, a, K, m), base_vector(bname, x, t, m, a, K)


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clip(1e-300)


# ---- comparator hinges (added after Phase B: n5/n9 and the fast neurons read the PREVIOUS steps' comparators) ----
CMP = [1, 6, 7, 8]
HFAMILIES = [  # (label, cell, base, lags) ; lags = number of past steps whose comparator hinges are added
    ("H1: A123K + comparator hinges lag 1", "a123K", "m5rest", 1),
    ("H3: A123K + comparator hinges lags 1-3", "a123K", "m5rest", 3),
    ("Hall: A123K + comparator hinges all lags", "a123K", "m5rest", 9),
    ("A12K-Hall: (age1,age2,K) + all comparator hinges", "a12K", "m5rest", 9),
]


class HingeBank:
    """Hinge variables c_{k,s} = ReLU(F_{k,s}(x)) for the comparators k in CMP and steps s, where F_{k,s} is the
    A123K formula for comparator k at step s (fit once on training data; held-out R^2 >= 0.999)."""

    def fit(self, x, pre_cmp):            # pre_cmp: (N, T+1, 4) original comparator pre-activations
        fam = FAMILIES[5]
        self.models = {}
        for s in range(1, x.shape[1] + 1):
            c, V = design(fam, x, s)
            self.models[s] = CellLinear().fit(c, V, pre_cmp[:, s])
        return self

    def transform(self, x):
        out = np.zeros((len(x), x.shape[1] + 1, len(CMP)))
        for s, mdl in self.models.items():
            c, V = design(FAMILIES[5], x, s)
            out[:, s] = np.maximum(mdl.predict(c, V), 0)
        return out


def design_h(hfam, x, t, hinges):
    label, cname, bname, lags = hfam
    m, a, K = prefix_stats(x, t)
    if cname == "a12K":
        a1, a2 = _age(a, 0, t), _age(a, 1, t)
        cell = (a1 * (t + 1) + a2) * 4 + np.minimum(K, 3)
    else:
        cell = cell_index(cname, t, a, K, m)
    V = base_vector(bname, x, t, m, a, K)
    L = [hinges[:, t - l] for l in range(1, min(lags, t - 1) + 1)]
    return cell, np.concatenate([V] + L, 1)
