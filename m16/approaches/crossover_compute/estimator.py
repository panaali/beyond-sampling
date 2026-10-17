"""Conditional Gaussian delta method; intended mechanistic, DRI class UNREVIEWED.

No random model inputs. All ordered top-two position chambers are enumerated.
Exact conditional input moments are model-free; network gradients are model-dependent.
Frozen-gate linearization and Gaussian closure are uncontrolled approximations.
The summed marginal failure tails use the first-order rare-event union approximation.
"""
from functools import lru_cache
import itertools
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtr, ndtri
from scipy.stats import beta

SUPPORTS = [(16, 10)]
FIXED_COST = False
ADVICE = None
# Conservative operation allowance for the deterministic moment table (up to 4 strata).
# Reported separately; no trained-model information enters this calculation.
PRECOMPUTE_FLOPS = 10_000_000
LEVELS = (1, 2, 4)


def cost_for(H, T, strata):
    """Conservative scalar FLOP ledger, including input Jacobians.

Jacobian propagation carries only the t past input columns at recurrence step t.
Each Gaussian CDF is charged 200 FLOPs, plus 100 for sqrt/division/guards.
Scalar arithmetic, gates, model weight scans, and reductions are included.
    """
    fwd = T * (2 * H * H + 4 * H) + 2 * T * H
    jac = sum(2 * H * H * t + H * (t + 1) for t in range(T)) + 2 * T * H * T
    margins = T * T + (T - 1) * (2 * T * T + 4 * T + 300)
    chamber = fwd + jac + margins + 10 * T * T
    scan = 2 * (H + H * H + T * H)
    fpe = 6000 * (T * (2 * H * H + 2 * H) + 2 * H * T) / 5760
    return (T * (T - 1) * strata * chamber + scan) / fpe


@lru_cache(maxsize=None)
def moments(T, strata):
    """Input means/covariances, canonical positions [max, second, lower...].

Phi(second) ~ Beta(T-1,2). Given second=b, max and T-2 lower
inputs are mutually independent normal variables truncated above/below b.
Integrals use b in [-12,12]; omitted probability is below 1e-29 at T=10.
    """
    out = []
    edges = ndtri(beta.ppf(np.linspace(0, 1, strata + 1), T - 1, 2))
    edges = np.clip(edges, -12, 12)
    for left, right in zip(edges[:-1], edges[1:]):
        def integrand(b):
            phi = np.exp(-b*b/2) / np.sqrt(2*np.pi)
            cdf, sf = ndtr(b), ndtr(-b)
            a, low = phi / sf, -phi / cdf
            density = T * (T - 1) * phi * cdf**(T - 2) * sf
            return density * np.array([1, a, b, low, 1+b*a, b*b,
                                        1+b*low, a*b, a*low, b*low, low*low])
        z, _ = quad_vec(integrand, left, right, epsabs=1e-11, epsrel=1e-11)
        z /= z[0]
        _, a, b, low, aa, bb, ll, ab, al, bl, lx = z
        mean = np.array([a, b] + [low] * (T-2))
        second = np.full((T, T), lx)
        np.fill_diagonal(second, ll)
        second[0, 0], second[1, 1] = aa, bb
        second[0, 1] = second[1, 0] = ab
        second[0, 2:] = second[2:, 0] = al
        second[1, 2:] = second[2:, 1] = bl
        out.append((mean, second - np.outer(mean, mean)))
    return tuple(out)


def evaluate(model, budget):
    Wi, Wh, Wo = (np.asarray(w, dtype=float) for w in model)
    H, T = Wh.shape[0], Wo.shape[0]
    if (H, T) not in SUPPORTS:
        raise ValueError('Only M_16,10 is supported')
    if not all(np.isfinite(w).all() for w in (Wi, Wh, Wo)):
        raise ValueError('Nonfinite model weights')
    eligible = [s for s in LEVELS if cost_for(H, T, s) <= budget]
    if not eligible:
        raise ValueError(f'Budget {budget} below exhaustive minimum {cost_for(H,T,1):.3f} FPE')
    strata = max(eligible)
    means, covs, labels = [], [], []
    for mean, cov in moments(T, strata):
        for p1, p2 in itertools.permutations(range(T), 2):
            order = [p1, p2] + [p for p in range(T) if p not in (p1, p2)]
            inv = np.argsort(order)
            means.append(mean[inv]); covs.append(cov[np.ix_(inv, inv)]); labels.append(p2)
    x, covs, labels = np.array(means), np.array(covs), np.array(labels)
    n = len(x)
    h, J = np.zeros((n, H)), np.zeros((n, H, 0))
    for t in range(T):
        pre = h @ Wh.T + x[:, t, None] * Wi
        J = np.einsum('ij,bjk->bik', Wh, J, optimize=True)
        J = np.concatenate((J, np.broadcast_to(Wi[None, :, None], (n, H, 1))), axis=2)
        J *= (pre > 0)[:, :, None]
        h = np.maximum(pre, 0)
    logits = h @ Wo.T
    G = np.einsum('ij,bjk->bik', Wo, J, optimize=True)
    row = np.arange(n)
    margin = logits[row, labels, None] - logits
    grad = G[row, labels, None, :] - G
    var = np.einsum('bik,bkl,bil->bi', grad, covs, grad, optimize=True)
    sd = np.sqrt(np.maximum(var, 0))
    tails = ndtr(-margin / np.maximum(sd, 1e-150))
    # Deterministic zero margins follow np.argmax's first-index tie rule.
    zero = sd < 1e-14
    deterministic = ((margin < 0) | ((margin == 0) & (np.arange(T)[None] < labels[:, None])))
    tails = np.where(zero, deterministic.astype(float), tails)
    tails[row, labels] = 0
    union_upper = np.minimum(tails.sum(axis=1), 1)
    union_lower = tails.max(axis=1)
    return dict(estimate=float(1-union_upper.mean()), cost=cost_for(H,T,strata),
                strata=strata, union_gap=float((union_upper-union_lower).mean()))


def est(model, budget, rng):
    """Shared ARC estimator interface. rng deliberately unused; unsupported budgets raise."""
    result = evaluate(model, budget)
    return result['estimate'], result['cost']
