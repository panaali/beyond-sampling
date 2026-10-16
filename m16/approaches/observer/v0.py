"""v0: one canonical anchor per triple; Gaussian internal noise from the 7 'rest' inputs through the exact local map."""
import sys
import numpy as np
from scipy.special import ndtr
sys.path.insert(0, '/home/user/clean-neurons/m16')
from core import T, TRIPLES, forward_gates, logit_map, trunc_moments
from quad import grid
from common import Wi, Wh, Wo

EOS = np.array([1.5388, 1.0014, 0.6561, 0.3757, 0.1227, -0.1227, -0.3757, -0.6561, -1.0014, -1.5388])


def anchors(seed=0):
    X = np.empty((len(TRIPLES), T)); rng = np.random.default_rng(seed)
    for i, tr in enumerate(TRIPLES):
        X[i, tr] = EOS[:3]
        rest = [p for p in range(T) if p not in tr]
        ranks = np.arange(3, T); rng.shuffle(ranks); X[i, rest] = EOS[ranks]
    return X


def psychometric_err(D, tri, C, U, G, combine="prod"):
    """D (10 competitors, 10 inputs) margin coefficients logit[p2]-logit[j]. Returns P(err) at grid nodes."""
    p1, p2, p3 = tri
    rest = [p for p in range(T) if p not in tri]
    B = C + U; A = B + G
    mc, vc = trunc_moments(C)
    logq = 0; perr = 0
    for j in range(T):
        if j == p2:
            continue
        d = D[j]
        mu = d[p1] * A + d[p2] * B + d[p3] * C + mc * d[rest].sum()
        sd = np.sqrt(vc * (d[rest] ** 2).sum() + 1e-12)
        q = ndtr(mu / sd)
        logq = logq + np.log(np.maximum(q, 1e-300)); perr = perr + (1 - q)
    return 1 - np.exp(logq) if combine == "prod" else np.minimum(perr, 1)


def estimate(model, n=20, seed=0, combine="prod"):
    X = anchors(seed)
    L, Gt = forward_gates(model, X)
    Amap = logit_map(model, Gt)
    C, U, G, W = grid(n, n, n)
    errs = np.zeros(len(TRIPLES))
    for i, tri in enumerate(TRIPLES):
        D = Amap[i, tri[1]][None] - Amap[i]
        errs[i] = (W * psychometric_err(D, tri, C, U, G, combine)).sum() / 720
    return errs


if __name__ == "__main__":
    e = estimate((Wi, Wh, Wo))
    print('pred err share on b>0:', e.sum())
    np.save('v0_errs.npy', e)
