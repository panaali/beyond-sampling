"""Model-derived per-input surrogate built from designed probes (FINDINGS sec. 5 probe estimator, made per-input).

thresholds(model): for every (p1,p2,p3) triple, bisect on a canonical input the relative gap at which the
model flips 2nd->3rd (tau23) and 2nd->1st (tau12). Cost counted in forward evaluations.
g(x): per-input prediction of correctness, hard or soft (logistic in the relative gap around tau).
Its mean E[g] is model-free given tau, so it can be computed from cheap input-only samples."""
import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, T

_ref = np.sort(np.random.default_rng(0).standard_normal((200_000, T)), 1)[:, ::-1]
EOS = _ref.mean(0)
TRIPLES = np.array([(a, b, c) for a in range(T) for b in range(T) for c in range(T) if len({a, b, c}) == 3])
TID = np.zeros((T, T, T), int); TID[TRIPLES[:, 0], TRIPLES[:, 1], TRIPLES[:, 2]] = np.arange(len(TRIPLES))


def canon(seed=0):
    X = np.empty((len(TRIPLES), T)); rs = np.random.default_rng(seed)
    for i, (a, b, c) in enumerate(TRIPLES):
        x = np.empty(T); x[a], x[b], x[c] = EOS[0], EOS[1], EOS[2]
        rest = [j for j in range(T) if j not in (a, b, c)]
        r = np.arange(3, T); rs.shuffle(r); x[rest] = EOS[r]; X[i] = x
    return X


def thresholds(model, iters=8, seed=0, hi0=0.6):
    """tau23, tau12 per triple (relative gaps), evals used. Vectorised bisection on [0, hi0]."""
    X = canon(seed); n = len(X); idx = np.arange(n); a, b, c = TRIPLES.T; v2 = X[idx, b]
    out, evals = [], 0
    for kind in ("23", "12"):
        def bad(gap):
            Y = X.copy()
            if kind == "23": Y[idx, c] = v2 * (1 - gap); wrong = c
            else: Y[idx, a] = v2 / (1 - gap); wrong = a
            return forward(model, Y).argmax(1) == wrong
        lo = np.zeros(n); hi = np.full(n, hi0)
        for _ in range(iters):
            mid = (lo + hi) / 2; bd = bad(mid); evals += n
            lo = np.where(bd, mid, lo); hi = np.where(bd, hi, mid)
        out.append((lo + hi) / 2)
    return out[0], out[1], evals


def feats(x):
    o = np.argsort(-x, 1); s = np.take_along_axis(x, o, 1)
    v1, v2, v3 = s[:, 0], s[:, 1], s[:, 2]
    blind = v2 <= 0
    g12 = (v1 - v2) / np.abs(v1); g23 = (v2 - v3) / np.maximum(np.abs(v2), 1e-12)
    return TID[o[:, 0], o[:, 1], o[:, 2]], g12, g23, blind


def g(x, tau23, tau12, blind_acc, w=0.0):
    """Per-input predicted correctness. w=0: hard thresholds; w>0: logistic, width w*tau (+1e-3)."""
    k, g12, g23, blind = feats(x)
    t23, t12 = tau23[k], tau12[k]
    if w == 0:
        p = ((g23 > t23) & (g12 > t12)).astype(float)
    else:
        sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        p = sig((g23 - t23) / (w * t23 + 1e-3)) * sig((g12 - t12) / (w * t12 + 1e-3))
    return np.where(blind, blind_acc, p)
