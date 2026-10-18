"""rank_exact (m16/approaches/rank_exact/rank_est.py) with cheaper settings. Same method; three knobs:
  Q       scrambled-Sobol points of the model-free law x | (ordering, K)        (rank_exact: 16/64/1024)
  stride  position-balanced subset of the 720 top-3 triples (cf.cyclic_subset)   (rank_exact: all 720)
  kgroups how many canonical gate points per triple: 11 (one per K, rank_exact), or K pooled into groups
          (gates frozen at the P(K)-weighted mean order statistics of the group; integration still per K).
Rest-rank arrangement per triple is rank_exact's (same hash seed). Deterministic, rng unused.
Labels: frozen gates at a model-free canonical input + exact linear map = deductive approximation; exact
integration of that linear surrogate over a model-free law (fixed Sobol) = deductive (quadrature of a derived
closed form); triple subset (stride > 1) = averaging over a fixed subset of orderings, i.e. an INDUCTIVE step.
Cost (FLOPs): per (triple, gate group) one forward + 10-column tangent = 11 x 5920, per (triple, K) Q x 200."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf import FPE, T, H, cyclic_subset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rank_exact"))
import rank_est as R  # noqa: E402   (model-free tables: EOSK, PK, ORDS, quad_points)

FLOP_EVAL = R.FLOP_EVAL
KGROUPS = {11: [[k] for k in range(T + 1)],
           5: [[0, 1, 2], [3, 4], [5], [6, 7], [8, 9, 10]],
           3: [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10]],
           1: [list(range(T + 1))]}
_Z = {}


def _canon(group):
    w = R.PK[group] / R.PK[group].sum()
    return (w[:, None] * R.EOSK[group]).sum(0)


def est_rx(model, Q=16, stride=1, kgroups=11, return_parts=False):
    Wi, Wh, Wo = model
    if Q not in _Z:
        _Z[Q] = R.quad_points(Q)
    Z = _Z[Q]
    sub = np.arange(len(R.ORDS)) if stride == 1 else cyclic_subset(stride)
    ORDS = R.ORDS[sub]; nO = len(ORDS)
    rank_of_pos = np.argsort(ORDS, 1)
    total = 0.0; flops = 0.0; parts = {}
    for grp in KGROUPS[kgroups]:
        xc = _canon(grp)[rank_of_pos]
        h = np.zeros((nO, H)); J = np.zeros((nO, H, T))
        for t in range(T):
            u = h @ Wh.T + xc[:, t:t + 1] * Wi; g = u > 0
            J = np.einsum('ij,njk->nik', Wh, J); J[:, :, t] += Wi; J *= g[:, :, None]
            h = u * g
        A = np.einsum('oh,nht->not', Wo, J)
        flops += nO * 11 * FLOP_EVAL
        for k in grp:
            X = Z[k][:, rank_of_pos]
            L = np.einsum('not,qnt->qno', A, X)
            flops += nO * Q * 2 * T * T
            acc_k = float((L.argmax(2) == ORDS[None, :, 1]).mean())
            parts[k] = acc_k
            total += R.PK[k] * acc_k
    out = (float(total), flops / FPE)
    return out + (parts,) if return_parts else out
