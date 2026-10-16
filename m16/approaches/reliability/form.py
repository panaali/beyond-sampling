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
torch.set_num_threads(1)
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


@torch.no_grad()
def _g_nograd(tm, u, perms, ks, positive):
    C.fwd += u.shape[0]
    return g_fn(tm, u, perms, ks, positive)


def ray_root(tm, d, perms, ks, positive=True, smax=8.0, nb=10, ngrid=8, g0=None):
    """Smallest s in (0, smax] with g(s d) = 0 found by a coarse grid then bisection (no gradients).
    If g(0)<0, finds first s where g turns positive along -... (handled by caller via sign). Returns s (smax if none)."""
    B = d.shape[0]
    if g0 is None:
        g0 = _g_nograd(tm, torch.zeros(B, T), perms, ks, positive)
    sgn = torch.sign(g0)
    grid = torch.linspace(0, smax, ngrid + 1)[1:]
    lo = torch.zeros(B); hi = torch.full((B,), float('nan'))
    for s in grid:
        gs = _g_nograd(tm, s * d, perms, ks, positive)
        hit = torch.isnan(hi) & (torch.sign(gs) != sgn)
        hi = torch.where(hit, s.expand(B), hi)
        lo = torch.where(torch.isnan(hi), s.expand(B), lo)
    found = ~torch.isnan(hi)
    hi = torch.where(found, hi, torch.full((B,), smax))
    for _ in range(nb):
        mid = (lo + hi) / 2
        gm = _g_nograd(tm, mid[:, None] * d, perms, ks, positive)
        same = torch.sign(gm) == sgn
        lo = torch.where(same & found, mid, lo); hi = torch.where(same | ~found, hi, mid)
    return torch.where(found, (lo + hi) / 2, torch.full((B,), smax)), found, sgn


def design_points(tm, perms, ks, positive=True, iters=6, smax=8.0, nb=10, ngrid=8):
    """Direction iteration: HL-RF proposes a direction, an exact ray search puts the point on g=0;
    keep the shortest. Returns signed beta (negative if the median point u=0 already fails),
    unit normal alpha at the best point (pointing into failure), best point."""
    B = perms.shape[0]
    g, gr = g_and_grad(tm, torch.zeros(B, T), perms, ks, positive)
    g00 = g.clone()
    d = -gr * torch.sign(g)[:, None]
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-300)
    best_s = torch.full((B,), smax); best_u = d * smax; best_a = d.clone()
    for it in range(iters):
        s, found, sgn = ray_root(tm, d, perms, ks, positive, smax, nb, ngrid, g0=g00)
        u = s[:, None] * d
        better = found & (s < best_s)
        if it == iters - 1:          # last ray search: only the gradient at the accepted point is needed
            g, gr = g_and_grad(tm, torch.where(better[:, None], u, best_u), perms, ks, positive)
            a = -gr / gr.norm(dim=1, keepdim=True).clamp_min(1e-300)
            best_a = torch.where(better[:, None], a, best_a)
            best_s = torch.where(better, s, best_s); best_u = torch.where(better[:, None], u, best_u)
            break
        g, gr = g_and_grad(tm, u, perms, ks, positive)
        a = -gr / gr.norm(dim=1, keepdim=True).clamp_min(1e-300)
        best_s = torch.where(better, s, best_s)
        best_u = torch.where(better[:, None], u, best_u)
        best_a = torch.where(better[:, None], a, best_a)
        # HL-RF proposal from this point; for sgn<0 (origin fails) we look for nearest safe point instead
        unew = ((gr * u).sum(1) - g)[:, None] / (gr ** 2).sum(1, keepdim=True).clamp_min(1e-300) * gr
        dn = unew / unew.norm(dim=1, keepdim=True).clamp_min(1e-300)
        d = torch.where(torch.isfinite(dn).all(1, keepdim=True) & (unew.norm(dim=1, keepdim=True) > 1e-9), dn, d)
    beta = best_s * sgn            # sgn = sign g(0)
    return beta.numpy(), best_a.numpy(), best_u.numpy()
