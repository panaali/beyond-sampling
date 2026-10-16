"""FORM / system-reliability estimator for M_16,10 in Rosenblatt (order-statistic) space.

Chamber c = (positions of ranks 1..10). Within a chamber, x = place(v(u)), u ~ N(0, I_10).
Failure modes k (wrong rank k wins): g_k(u) = logit[pos rank2] - logit[pos rank k]. Correct iff all g_k > 0.
Design point of mode k: argmin |u| s.t. g_k(u) = 0 (HL-RF on exact piecewise-linear gradients).
FORM per mode: P_k = Phi(-beta_k). System: P(correct) ~ Phi_m(beta; R), R = alpha alpha^T (Hohenbichler-Rackwitz),
plus Ditlevsen bounds.
"""
import itertools
import math
import sys

import numpy as np
import torch
from scipy.stats import norm, multivariate_normal

from rosen import sorted_from_u, Phi, G0

sys.path.insert(0, "/home/user/clean-neurons/m16")
torch.set_default_dtype(torch.float64)
T = 10


class Counter:
    fwd = 0       # forward passes (one length-10 sequence = 1); a backward pass counted as 2 forwards


C = Counter()


def tmodel(model):
    return tuple(torch.as_tensor(np.asarray(W)) for W in model)


def logits(tm, x):
    Wi, Wh, Wo = tm
    h = torch.zeros(x.shape[0], Wh.shape[0])
    for t in range(T):
        h = torch.relu(h @ Wh.T + x[:, t:t + 1] * Wi[None])
    return h @ Wo.T


def chambers(n_arr=1, seed=0, top=3):
    """All ordered top-`top` position tuples x n_arr fixed (model-independent) arrangements of the rest."""
    rng = np.random.default_rng(seed)
    perms = []
    for tup in itertools.permutations(range(T), top):
        rest = [i for i in range(T) if i not in tup]
        for a in range(n_arr):
            r = list(rng.permutation(rest))
            perms.append(list(tup) + r)
    return np.array(perms)          # perms[c, r] = position of rank r (0 = max)


def place(v, perms):
    x = torch.empty_like(v)
    x.scatter_(1, perms, v)
    return x


def g_fn(tm, u, perms, ks, positive=True):
    """g for each problem: perms (B,10) long, ks (B,) rank index of the competing class."""
    x = place(sorted_from_u(u, positive), perms)
    L = logits(tm, x)
    p2 = perms[:, 1]; pk = perms.gather(1, ks[:, None])[:, 0]
    return L.gather(1, p2[:, None])[:, 0] - L.gather(1, pk[:, None])[:, 0]


def g_and_grad(tm, u, perms, ks, positive=True):
    u = u.detach().clone().requires_grad_(True)
    g = g_fn(tm, u, perms, ks, positive)
    (gr,) = torch.autograd.grad(g.sum(), u)
    C.fwd += 3 * u.shape[0]
    return g.detach(), gr


def hlrf(tm, perms, ks, positive=True, iters=25, bmax=8.0, damp=1.0, u0=None):
    """Batched (damped) HL-RF. Returns beta (signed: negative if u=0 fails), alpha (unit normal pointing to failure)."""
    B = perms.shape[0]
    u = torch.zeros(B, T) if u0 is None else u0.clone()
    g0 = None
    for it in range(iters):
        g, gr = g_and_grad(tm, u, perms, ks, positive)
        if g0 is None:
            g0 = g.clone()
        n2 = (gr ** 2).sum(1).clamp_min(1e-300)
        unew = ((gr * u).sum(1) - g)[:, None] / n2[:, None] * gr
        r = unew.norm(dim=1, keepdim=True)
        unew = torch.where(r > bmax, unew / r * bmax, unew)
        u = u + damp * (unew - u)
    g, gr = g_and_grad(tm, u, perms, ks, positive)
    alpha = -gr / gr.norm(dim=1, keepdim=True).clamp_min(1e-300)     # direction of decreasing g
    # FORM linearisation at u*: g(u) ~ |grad| (beta - alpha.u), beta = alpha.u* + g(u*)/|grad|
    beta = (u * alpha).sum(1) + g / gr.norm(dim=1).clamp_min(1e-300)
    beta = beta.clamp(-bmax, bmax)
    return beta.numpy(), alpha.numpy(), u.numpy(), g.numpy()
