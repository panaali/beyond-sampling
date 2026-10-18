"""T274a probe machinery, parameterised by the model (vectorised over the 720 (p1,p2,p3) triples).
Every network call on a constructed input costs 1 FPE (6000 FLOPs at (16,10), T196/LB3 convention)."""
import itertools
import numpy as np

T = 10
_ref = np.sort(np.random.default_rng(0).standard_normal((400_000, T)), 1)[:, ::-1]   # model-free, no network
EOS = _ref.mean(0)
REL23 = np.sort((_ref[:, 1] - _ref[:, 2]) / np.abs(_ref[:, 1]))
REL12 = np.sort((_ref[:, 0] - _ref[:, 1]) / np.abs(_ref[:, 0]))
TRIPLES = np.array([t for t in itertools.permutations(range(T), 3)])            # 720 x 3


def cdf(sorted_sample, g):
    return np.searchsorted(sorted_sample, g) / len(sorted_sample)


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1)


def canonical(order_rows, seed, eos=EOS):
    """Inputs with global ranks 1,2,3 at the given positions and ranks 4..10 shuffled (fixed seed) over the rest."""
    X = np.empty((len(order_rows), T))
    rng = np.random.default_rng(seed)
    for i, o in enumerate(order_rows):
        X[i, o] = eos[:3]
        rest = [p for p in range(T) if p not in o]
        r = np.arange(3, T)
        rng.shuffle(r)
        X[i, rest] = eos[r]
    return X


def bisect(model, X, setter, target_pos, lo, hi, iters):
    """Vectorised bisection on g: returns (threshold, evals). threshold = largest g in [lo,hi] at which the net
    picks target_pos (0 if not at lo, hi if still at hi)."""
    n = len(X)
    at = lambda g: forward(model, setter(X, g)) == target_pos
    ev = 2 * n
    alo, ahi = at(np.full(n, lo)), at(np.full(n, hi))
    L, Hh = np.full(n, lo), np.full(n, hi)
    act = alo & ~ahi
    for _ in range(iters):
        mid = (L + Hh) / 2
        r = at(mid)
        ev += int(act.sum())
        L = np.where(act & r, mid, L)
        Hh = np.where(act & ~r, mid, Hh)
    th = np.where(~alo, 0.0, np.where(ahi, hi, L))
    return th, ev
