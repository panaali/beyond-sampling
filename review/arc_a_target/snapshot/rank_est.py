"""Deductive rank-ordering estimator ("RX") for M_16,10 accuracy.

accuracy = sum_K P(K) * mean_{orderings} P(correct | ordering, K)      (exact identity; K = #positive inputs,
P(K) = C(10,K)/1024 and all orderings equally likely, independent of K's ranks).
Approximations:
  * orderings: a fixed deterministic design: every (rank1,rank2,rank3) position triple (720), remaining
    ranks arranged by a fixed permutation (hash of the triple)  -> 7,920 classes (triple, K);
  * gates: fixed at the class's canonical input (expected order statistics given K, placed by rank), which
    makes the logits an exact linear map A_c x (tangent propagation through the RNN with frozen gates);
  * P(argmax(A_c x) = target | class) integrated over the model-free law of x | (ordering, K) with a fixed
    scrambled-Sobol quadrature (no model calls; cost counted in FLOPs).
Compute: forward pass ~ 6k FLOPs; tangent (10 columns) ~ 10 passes; quadrature Q points ~ Q*200 FLOPs.
"""
import sys
import numpy as np
from itertools import permutations
from scipy.stats import qmc, norm
T, H = 10, 16
FLOP_EVAL = T * (2 * H * H + 3 * H) + 2 * T * H       # one single-sequence forward pass

_ref = np.sort(np.random.default_rng(0).standard_normal((2_000_000, T)), 1)[:, ::-1]   # model-free
_K = (_ref > 0).sum(1)
EOSK = np.array([_ref[_K == k].mean(0) for k in range(T + 1)])
PK = np.array([np.math.comb(T, k) if hasattr(np, 'math') else 0 for k in range(T + 1)], float) if False else \
     np.array([__import__('math').comb(T, k) for k in range(T + 1)], float) / 2 ** T

def quad_points(Q, seed=1):
    """Deterministic quadrature for z | K: K sorted |N| values (positives), T-K sorted -|N| values."""
    u = qmc.Sobol(T, scramble=True, seed=seed).random(Q)
    a = norm.ppf(0.5 + 0.5 * np.clip(u, 1e-12, 1 - 1e-12))      # half-normal quantiles
    out = []
    for k in range(T + 1):
        pos = -np.sort(-a[:, :k], 1); neg = -np.sort(a[:, k:], 1)
        out.append(np.concatenate([pos, neg], 1))                # z1 >= ... >= z10
    return np.array(out)                                          # (T+1, Q, T)

def design():
    ords = []
    for j, (p1, p2, p3) in enumerate(permutations(range(T), 3)):
        rest = [i for i in range(T) if i not in (p1, p2, p3)]
        rest = list(np.random.default_rng(1000 + j).permutation(rest))
        ords.append([p1, p2, p3] + rest)                          # ords[r] = position of rank r
    return np.array(ords)

ORDS = design()

def est_rank(model, budget=None, rng=None, Q=64):
    Wi, Wh, Wo = model
    Z = quad_points(Q)
    nO = len(ORDS); total = 0.0; flops = 0
    rank_of_pos = np.argsort(ORDS, 1)                             # rank at each position
    for k in range(T + 1):
        xc = EOSK[k][rank_of_pos]                                 # canonical inputs (nO, T)
        h = np.zeros((nO, H)); J = np.zeros((nO, H, T))
        for t in range(T):
            u = h @ Wh.T + xc[:, t:t + 1] * Wi; g = u > 0
            J = np.einsum('ij,njk->nik', Wh, J); J[:, :, t] += Wi; J *= g[:, :, None]
            h = u * g
        A = np.einsum('oh,nht->not', Wo, J)                      # logits = A x
        flops += nO * 11 * FLOP_EVAL
        X = Z[k][:, rank_of_pos]                                   # (Q, nO, T): x | ordering, K
        L = np.einsum('not,qnt->qno', A, X)
        flops += nO * Q * 2 * T * T
        acc_k = (L.argmax(2) == ORDS[None, :, 1]).mean()
        total += PK[k] * acc_k
    return float(total), flops / FLOP_EVAL

if __name__ == "__main__":
    sys.path.insert(0, '/home/user/clean-neurons/m16')
    from common import Wi, Wh, Wo
    for Q in (16, 64, 256):
        e, c = est_rank((Wi, Wh, Wo), Q=Q); print(f"Q={Q}: estimate {e:.5f}  bias {100*(e-0.953179):+.2f} pp  eval-equiv {c:.0f}")
