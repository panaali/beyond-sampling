"""Deductive FORM/SORM + system-reliability accuracy estimator for M_16,10 (leaderboard interface).

est(model, budget, rng) -> (estimate, cost), cost in forward-pass equivalents.
No random model inputs: chambers (rank -> position maps) are a fixed model-independent design, design
points are found by deterministic search on the model's exact piecewise-linear gradients, and the
probabilities are Gaussian integrals (Phi, bivariate normal via Owen's T). See README.md.

Pipeline per region (2nd largest > 0, and the 'blind' region 2nd largest <= 0, prob 11/1024):
  for every chamber c and wrong rank k: design point u* of g_k(u) = logit[pos rank2] - logit[pos rank k]
  in Rosenblatt space (u = 0 <-> median order statistics), beta = +-|u*|;
  SORM (Hohenbichler) correction from the curvature the Rosenblatt map induces (gate kinks ignored);
  P(fail | c) = midpoint of Ditlevsen bounds of the union of linearised modes (SORM-equivalent betas).
  accuracy = 1 - [(1 - 11/1024) mean_c P(fail|c, pos) + 11/1024 mean_c P(fail|c, blind)].
"""
import os
import sys

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from form import design_points, chambers, tmodel, C
from rosen import G0
from sorm import x_grad, u_hessian, curvatures, sorm_pf
from system import ditlevsen

DEDUCTIVE = True
ROSEN_OVERHEAD = 1.3        # each g evaluation also runs the Rosenblatt map (~2k FLOPs) -> x1.3
SORM_COST = 20              # model-free FD Hessian of w.v(u): ~20 Rosenblatt grads ~ 120k FLOPs per problem

CFG = dict(
    modes_pos=(0, 2, 3, 4, 5, 6, 7, 8, 9), modes_neg=(0, 2, 3, 4, 5, 6, 7, 8, 9),
    iters=6, ngrid=8, nb=10, n_arr=1, stride_pos=1, stride_neg=1, sorm=True, combine='dit_mid',
)


def region(model, positive, modes, stride, cfg):
    tm = tmodel(model)
    perms = torch.as_tensor(chambers(cfg['n_arr'], seed=0))[::stride]
    Cn = len(perms); m = len(modes)
    P = perms.repeat(m, 1); ks = torch.as_tensor(np.repeat(modes, Cn))
    b, a, u = design_points(tm, P, ks, positive, iters=cfg['iters'], nb=cfg['nb'], ngrid=cfg['ngrid'])
    extra = 0.0
    if cfg['sorm']:
        U = torch.as_tensor(u)
        w, _ = x_grad(tm, U, P, ks, positive)
        H, gu = u_hessian(U, w, positive)
        kap = curvatures(H, gu)
        pf = sorm_pf(b, kap, 'hr')
        b = -norm.ppf(np.clip(pf, 1e-15, 1 - 1e-15))
        extra = SORM_COST * len(U)
    b = b.reshape(m, Cn); a = a.reshape(m, Cn, -1)
    if cfg['combine'] == 'sum':
        pfail = norm.cdf(-b).sum(0)
    else:
        lo, up = ditlevsen(b, a)
        pfail = {'dit_lo': lo, 'dit_up': up, 'dit_mid': 0.5 * (lo + up)}[cfg['combine']]
    return float(np.mean(np.clip(pfail, 0, 1))), extra


def estimate(model, cfg=None):
    cfg = {**CFG, **(cfg or {})}
    C.fwd = 0
    f_pos, e1 = region(model, True, cfg['modes_pos'], cfg['stride_pos'], cfg)
    f_neg, e2 = region(model, False, cfg['modes_neg'], cfg['stride_neg'], cfg)
    acc = 1 - ((1 - G0) * f_pos + G0 * f_neg)
    cost = C.fwd * ROSEN_OVERHEAD + e1 + e2
    return acc, cost, dict(f_pos=f_pos, f_neg=f_neg)


# budget -> configuration ladder (cheapest first); chosen = most expensive config whose nominal cost fits
LADDER = [
    dict(modes_pos=(0, 2, 3), modes_neg=(0, 2, 3), iters=1, ngrid=4, nb=5, stride_pos=4, stride_neg=8),
    dict(modes_pos=(0, 2, 3), modes_neg=(0, 2, 3, 4), iters=2, ngrid=4, nb=6, stride_pos=2, stride_neg=4),
    dict(modes_pos=(0, 2, 3, 4), modes_neg=(0, 2, 3, 4, 5), iters=2, ngrid=6, nb=8, stride_pos=1, stride_neg=2),
    dict(),   # full default
]


# nominal costs are model-independent (fixed batch sizes and iteration counts); measured with sweep.py
LADDER_COST = [35154, 110880, 336960, 2483136]


def est(model, budget, rng=None):
    """Most accurate ladder configuration whose (deterministic) cost fits the budget; the cheapest one if none fits.
    rng is unused (deductive)."""
    j = 0
    for i, c in enumerate(LADDER_COST):
        if c <= budget:
            j = i
    acc, cost, _ = estimate(model, LADDER[j])
    return acc, cost
