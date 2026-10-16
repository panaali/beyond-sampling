"""Hybrid (sampling-based, unbiased for any weights): design-point importance sampling in Rosenblatt space.

Proposal per sample: region (2nd largest >0 / <=0) stratified with fixed allocation; uniformly random
permutation; u ~ w0 N(0,I) + sum_k w_k N(u*_k(triple), I), with u*_k the design points of the
ORIGINAL model's rank-1 and rank-3 modes (a fixed precomputed design, reused for every noisy copy, like
the stratified estimator's pilot allocation). Weight = phi(u)/q(u). Blind region: plain sampling.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '/home/user/clean-neurons/m16')
from arc_eval import forward  # noqa: E402
from rosen import sorted_from_u, G0  # noqa: E402
from form import chambers  # noqa: E402

torch.set_num_threads(1)
_d = np.load('/home/user/clean-neurons/m16/approaches/reliability/orig_modes_1.npz')
_PERMS = chambers(1, seed=0)                 # row c <-> triple (p1,p2,p3) in itertools order
_TRIPLE_INDEX = {tuple(p[:3]): c for c, p in enumerate(_PERMS)}
_U = _d['u'][[0, 1]]                         # modes rank1 (index 0) and rank3 (index 1): (2, 720, 10)
_B = _d['b'][[0, 1]]


def _vals(u, positive):
    with torch.no_grad():
        return sorted_from_u(torch.as_tensor(u), positive).numpy()


def est_hybrid(model, budget, rng, w_dp=(0.25, 0.25), frac_blind=0.03, bmax=4.0):
    n_b = max(2, int(round(frac_blind * budget))); n_p = budget - n_b
    # positive region
    perm = np.argsort(rng.random((n_p, 10)), 1)               # perm[i, r] = position of rank r
    cidx = np.array([_TRIPLE_INDEX[tuple(p[:3])] for p in perm])
    ustar = _U[:, cidx]                                        # (2, n, 10)
    ok_dp = _B[:, cidx] < bmax                                 # use a design point only if it is reasonably close
    w = np.array(w_dp)[:, None] * ok_dp                        # (2, n)
    w0 = 1 - w.sum(0)
    comp = (rng.random(n_p)[None] > np.cumsum(np.vstack([w0, w]), 0)).sum(0)   # 0 = N(0,I), 1/2 = design points
    u = rng.standard_normal((n_p, 10))
    for k in (1, 2):
        m = comp == k
        u[m] += ustar[k - 1, m]
    # q/phi = w0 + sum_k w_k exp(u.u*_k - |u*_k|^2/2)
    ratio = w0 + sum(w[k] * np.exp((u * ustar[k]).sum(1) - 0.5 * (ustar[k] ** 2).sum(1)) for k in range(2))
    x = np.empty((n_p, 10)); np.put_along_axis(x, perm, _vals(u, True), 1)
    ok = forward(model, x).argmax(1) == perm[:, 1]
    acc_p = 1 - np.mean(~ok / ratio)                         # E_q[1/ratio] = 1 exactly: weight only the failures
    # blind region: plain sampling within the region
    perm = np.argsort(rng.random((n_b, 10)), 1)
    x = np.empty((n_b, 10)); np.put_along_axis(x, perm, _vals(rng.standard_normal((n_b, 10)), False), 1)
    acc_b = np.mean(forward(model, x).argmax(1) == perm[:, 1])
    return float((1 - G0) * acc_p + G0 * acc_b), budget
