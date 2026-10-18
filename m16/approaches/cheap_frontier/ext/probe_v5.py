"""Probe v5 (T274a): constructed inputs on an exact-law design, the network's own answer at each.

Node i (i = 0..N-1):
  * position triple (p1, p2, p3) = TRIPLES[i mod 720]  -- exact: positions are uniform and independent of values;
  * top-3 values from a fixed scrambled Sobol point u in (0,1)^3 by the exact Renyi conditional quantiles of the
    order statistics of 10 i.i.d. N(0,1):  M1 = Q(u1^(1/10)),  M2 = Q(Phi(M1) u2^(1/9)),  M3 = Q(Phi(M2) u3^(1/8));
  * ranks 4..10: canonical values Q(Phi(M3) (8-j)/8), j = 1..7 (expected uniform order statistics of 7 draws
    below M3), placed on the remaining positions in a fixed permutation (seeded by the node index).
Estimate = fraction of nodes where the network returns p2. Every node costs 1 FPE. Deterministic (rng unused).
Labels: positions and top-3 law exact (deductive); network answer exact (deductive); lower ranks fixed at
canonical values (approximation); averaging over fixed nodes (inductive, Hilton / web_log).
"""
import itertools
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

T = 10
TRIPLES = np.array([t for t in itertools.permutations(range(T), 3)])
_cache = {}


def design(N, seed=274):
    if (N, seed) in _cache:
        return _cache[N, seed]
    u = qmc.Sobol(3, scramble=True, seed=seed).random(N)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    M1 = ndtri(u[:, 0] ** (1 / 10))
    M2 = ndtri(ndtr(M1) * u[:, 1] ** (1 / 9))
    M3 = ndtri(ndtr(M2) * u[:, 2] ** (1 / 8))
    low = ndtri(ndtr(M3)[:, None] * (np.arange(7, 0, -1) / 8)[None])      # 7 canonical values below M3
    X = np.empty((N, T))
    tri = TRIPLES[np.arange(N) % 720]
    rows = np.arange(N)
    X[rows, tri[:, 0]], X[rows, tri[:, 1]], X[rows, tri[:, 2]] = M1, M2, M3
    rng = np.random.default_rng(seed)
    for i in range(N):
        rest = np.setdiff1d(np.arange(T), tri[i])
        X[i, rng.permutation(rest)] = low[i]
    _cache[N, seed] = (X, tri[:, 1])
    return _cache[N, seed]


def forward_argmax(model, x, chunk=200_000):
    Wi, Wh, Wo = model
    out = []
    for i in range(0, len(x), chunk):
        xx = x[i:i + chunk]
        h = np.zeros((len(xx), Wh.shape[0]))
        for t in range(T):
            h = np.maximum(h @ Wh.T + xx[:, t:t + 1] * Wi[None], 0)
        out.append((h @ Wo.T).argmax(1))
    return np.concatenate(out)


def est(model, budget, rng=None):
    N = int(budget)
    X, ans = design(N)
    return float((forward_argmax(model, X) == ans).mean()), N


# ---------------------------------------------------------------- v5b: weight-derived lower-rank spread
def _lower_cov(M3):
    """Model-free covariance of the 7 values below M3 (sorted, descending), by the delta method on uniform order
    statistics: U_(j) of 7 uniforms has Cov(U_(i), U_(j)) = a_i (1 - a_j) / 9, i <= j in ascending order,
    a_k = k/8; v = Q(Phi(M3) U)."""
    a = np.arange(1, 8) / 8.0                      # ascending uniform means
    C = np.minimum.outer(a, a) * (1 - np.maximum.outer(a, a)) / 9.0
    C = C[::-1, ::-1]                              # descending order (matches low[] above)
    v = ndtri(ndtr(M3)[:, None] * a[::-1][None])
    D = ndtr(M3)[:, None] / np.maximum(np.exp(-0.5 * v ** 2) / np.sqrt(2 * np.pi), 1e-300)
    return D[:, :, None] * C[None] * D[:, None, :]          # (N, 7, 7)


def _forward_gates(model, X):
    Wi, Wh, Wo = model
    N = len(X)
    h = np.zeros((N, Wh.shape[0]))
    G = np.zeros((T, N, Wh.shape[0]), bool)
    for t in range(T):
        p = h @ Wh.T + X[:, t:t + 1] * Wi[None]
        G[t] = p > 0
        h = np.maximum(p, 0)
    return h @ Wo.T, G


def _grad_x(model, G, c):
    """Exact gradient of the logit combination c (N, 10) w.r.t. the 10 inputs, inside each node's gate region."""
    Wi, Wh, Wo = model
    lam = c @ Wo                                    # dL/dh_10
    g = np.zeros((len(c), T))
    for t in range(T - 1, -1, -1):
        mu = lam * G[t]                             # dL/dpre_t
        g[:, t] = mu @ Wi
        lam = mu @ Wh
    return g


def est_spread(model, budget, rng=None, rivals=2, n_low=7):
    """v5b: P(wrong) per node = 1 - prod over the top `rivals` competitors of Phi(d_k / s_k), d_k the exact logit
    margin at the node, s_k its s.d. under the model-free lower-rank covariance through the exact local Jacobian.
    Cost: 1 forward + `rivals` reverse passes per node = (1 + rivals) FPE."""
    N = int(budget) // (1 + rivals)
    X, ans = design(N)
    Z, G = _forward_gates(model, X)
    tri = TRIPLES[np.arange(N) % 720]
    Sig = _lower_cov(X[np.arange(N), tri[:, 2]])
    # positions of the 7 lower values in descending value order
    rows = np.arange(N)
    top = np.zeros((N, T), bool)
    for k in range(3):
        top[rows, tri[:, k]] = True
    low_pos = np.argsort(-np.where(top, -np.inf, X), 1)[:, :7]
    rows = np.arange(N)
    zt = Z[rows, ans]
    Zo = Z.copy()
    Zo[rows, ans] = -np.inf
    order = np.argsort(-Zo, 1)[:, :rivals]
    pc = np.ones(N)
    for r in range(rivals):
        k = order[:, r]
        c = np.zeros((N, T))
        c[rows, ans] += 1
        c[rows, k] -= 1
        g = _grad_x(model, G, c)
        gl = np.take_along_axis(g, low_pos, 1)[:, :n_low]
        s = np.sqrt(np.einsum("ni,nij,nj->n", gl, Sig[:, :n_low, :n_low], gl)) + 1e-12
        pc *= ndtr((zt - Z[rows, k]) / s)
    return float(pc.mean()), N * (1 + rivals) * 1.01


# ---------------------------------------------------------------- v5q: control, full exact-law QMC (no canonical part)
def design_full(N, seed=274):
    key = ("full", N, seed)
    if key in _cache:
        return _cache[key]
    u = np.clip(qmc.Sobol(10, scramble=True, seed=seed).random(N), 1e-12, 1 - 1e-12)
    M = np.empty((N, T))
    prev = np.ones(N)
    for k in range(T):                                  # Renyi: k-th largest given the (k-1)-th
        M[:, k] = ndtri(prev * u[:, k] ** (1 / (T - k)))
        prev = ndtr(M[:, k])
    X = np.empty((N, T))
    tri = TRIPLES[np.arange(N) % 720]
    rows = np.arange(N)
    X[rows, tri[:, 0]], X[rows, tri[:, 1]], X[rows, tri[:, 2]] = M[:, 0], M[:, 1], M[:, 2]
    rng = np.random.default_rng(seed)
    for i in range(N):
        rest = np.setdiff1d(np.arange(T), tri[i])
        X[i, rng.permutation(rest)] = M[i, 3:]
    _cache[key] = (X, tri[:, 1])
    return _cache[key]


def est_full(model, budget, rng=None):
    N = int(budget)
    X, ans = design_full(N)
    return float((forward_argmax(model, X) == ans).mean()), N
