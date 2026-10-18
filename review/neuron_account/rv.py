"""Independent review of T241 (neuron account of M_16,10): shared helpers, written from T241's README definitions,
not from its code. Weights are read from ARC's public checkpoint (data/m16.pth, sha256 checked).

Notation as in T241: step t = 1..10 reads x_{t-1}; h_t = state after it; pre_t = Wh h_{t-1} + Wi x_{t-1}.
"""
import hashlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PTH = os.path.join(HERE, "data", "m16.pth")
SHA = "498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb"
T, H = 10, 16
CMP = [1, 6, 7, 8]


def weights():
    """From the .pth (sha256 checked) via torch once; cached as data/w.npz (torch inflates virtual memory)."""
    cache = os.path.join(HERE, "data", "w.npz")
    raw = open(PTH, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == SHA
    if not os.path.exists(cache):
        import torch
        sd = torch.load(PTH, map_location="cpu", weights_only=False)
        f = lambda k: sd[k].double().numpy()
        np.savez(cache, Wi=f("rnn.weight_ih_l0")[:, 0], Wh=f("rnn.weight_hh_l0"), Wo=f("linear.weight"))
    w = np.load(cache)
    return w["Wi"], w["Wh"], w["Wo"]


Wi, Wh, Wo = weights()


def gauss(n, seed):
    """Review streams use a different generator (Philox) and seeds from T241 (PCG64, 2410xx)."""
    return np.random.Generator(np.random.Philox(seed)).standard_normal((n, T))


def states(x, dtype=np.float64):
    """pre (N, T+1, H) and h (N, T+1, H); index t = after step t (index 0 = zeros). Computed in float64,
    stored in dtype."""
    n = len(x); pre = np.zeros((n, T + 1, H), dtype); h = np.zeros((n, T + 1, H), dtype)
    for s in range(0, n, 100_000):
        hc = np.zeros((min(n, s + 100_000) - s, H))
        for t in range(1, T + 1):
            p = hc @ Wh.T + np.outer(x[s:s + 100_000, t - 1], Wi)
            hc = np.maximum(p, 0); pre[s:s + len(hc), t] = p; h[s:s + len(hc), t] = hc
    return pre, h


def answer(x):
    """True 2nd-argmax position."""
    return np.argsort(-x, axis=1)[:, 1]


# ---------------------------------------------------------------- interpretable variables (T241 README, Phase B)
def prefix(x, t):
    """Running order statistics of x_0..x_{t-1}: values m (desc), ages a = (t-1) - position, K = #positive."""
    P = x[:, :t]
    o = np.argsort(-P, axis=1, kind="stable")
    return np.take_along_axis(P, o, 1), (t - 1) - o, (P > 0).sum(1)


def col(M, k, fill):
    return M[:, k] if k < M.shape[1] else np.full(len(M), fill, dtype=M.dtype)


def cell_of(kind, t, a, K):
    b = t + 1                                              # age t = "rank absent"
    ag = [col(a, k, t) for k in range(3)]
    Kc = np.minimum(K, 3)
    if kind == "1":
        return ag[0]
    if kind == "12":
        return ag[0] * b + ag[1]
    if kind == "123":
        return (ag[0] * b + ag[1]) * b + ag[2]
    if kind == "123K":
        return ((ag[0] * b + ag[1]) * b + ag[2]) * 4 + Kc
    if kind == "12K":
        return (ag[0] * b + ag[1]) * 4 + Kc
    raise ValueError(kind)


def base_of(kind, x, t, m):
    cur = x[:, t - 1]
    if kind == "m3":
        return np.column_stack([cur] + [col(m, k, 0.0) for k in range(3)])
    if kind == "m5":
        return np.column_stack([cur] + [col(m, k, 0.0) for k in range(5)])
    if kind == "m5rest":
        rest = m[:, 5:]
        rp = np.maximum(rest, 0).sum(1) if rest.shape[1] else np.zeros(len(m))
        rn = np.minimum(rest, 0).sum(1) if rest.shape[1] else np.zeros(len(m))
        return np.column_stack([cur] + [col(m, k, 0.0) for k in range(5)] + [rp, rn])
    raise ValueError(kind)


# (name, cell kind, base kind, hinge lags)
FAMS = [("A1", "1", "m3", 0), ("A12", "12", "m3", 0), ("A123", "123", "m5", 0), ("A123K", "123K", "m5rest", 0),
        ("H1", "123K", "m5rest", 1), ("H3", "123K", "m5rest", 3), ("Hall", "123K", "m5rest", 9),
        ("A12K-Hall", "12K", "m5rest", 9)]


def design(fam, x, t, hinge=None):
    _, ck, bk, lags = fam
    m, a, K = prefix(x, t)
    V = base_of(bk, x, t, m)
    if lags:
        L = [hinge[:, t - l] for l in range(1, min(lags, t - 1) + 1)]
        if L:
            V = np.column_stack([V] + L)
    return cell_of(ck, t, a, K), V


class CellRidge:
    """Per-cell least squares (relative ridge 1e-6, as T241, no intercept), pooled fit for cells unseen in training.
    Normal equations accumulated per cell by sorting rows (an implementation independent of T241's bincounts)."""

    def fit(self, c, V, Y):
        o = np.argsort(c, kind="stable"); cs = c[o]
        ids, start = np.unique(cs, return_index=True); stop = np.r_[start[1:], len(cs)]
        d = V.shape[1]; self.ids = ids; self.coef = np.zeros((len(ids), d, Y.shape[1]))
        G = V.T @ V; self.pooled = np.linalg.solve(G + 1e-10 * np.trace(G) / d * np.eye(d), V.T @ Y)
        for i, (s, e) in enumerate(zip(start, stop)):
            Xi = V[o[s:e]]; Yi = Y[o[s:e]]
            A = Xi.T @ Xi
            A += (1e-6 * np.trace(A) / d + 1e-300) * np.eye(d)
            self.coef[i] = np.linalg.solve(A, Xi.T @ Yi) if e - s >= 1 else self.pooled
        return self

    def predict(self, c, V):
        pos = np.clip(np.searchsorted(self.ids, c), 0, len(self.ids) - 1)
        seen = self.ids[pos] == c
        out = V @ self.pooled
        for s in range(0, len(c), 50_000):
            sl = slice(s, s + 50_000); r = np.flatnonzero(seen[sl]) + s
            out[r] = np.einsum("nd,ndq->nq", V[r], self.coef[pos[r]])
        self.unseen = 1 - seen.mean()
        return out


def r2(y, f):
    return 1 - ((y - f) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clip(1e-300)
