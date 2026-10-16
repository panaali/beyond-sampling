"""Deterministic tie-segment quadrature (sci_feynman). See README.md.
estimate(model, budget, design, shift_seed) -> (acc, cost, parts)
Designs (all per position-configuration, configurations enumerated exactly):
  'iid'   : iid uniform points (diagnosis baseline only)
  'sobol' : 2^k Sobol points, a different Cranley-Patterson shift per configuration
  'prod'  : top variables on a Gauss-Legendre product rule in quantile coordinates x bath on a Latin design
shift_seed=None gives the fixed deterministic rule; an int randomises the shifts (diagnosis of the rule's error).
"""
import zlib
import numpy as np
from scipy.stats import qmc
from numpy.polynomial.legendre import leggauss
import core as C

TRIP = [(a, b, p) for a in range(10) for b in range(a + 1, 10) for p in range(10) if p not in (a, b)]
PAIR = [(a, b) for a in range(10) for b in range(a + 1, 10)]
SHARES = dict(r23=0.70, r12=0.26, blind=0.04)
COST23, COST12 = 7.5, 6.7          # planning values only (actual cost is measured and returned)


def _shift(key, dim, shift_seed):
    if shift_seed is None:
        rs = np.random.default_rng(zlib.crc32(key.encode()))      # fixed, model-independent
    else:
        rs = np.random.default_rng([shift_seed, zlib.crc32(key.encode())])
    return rs.random(dim)


def _sobol(n, dim):
    m = max(int(np.ceil(np.log2(max(n, 1)))), 0)
    return qmc.Sobol(dim, scramble=False).random_base2(m)[:n] if n > 1 else np.full((1, dim), 0.5)


def unit_points(design, n, dim, key, shift_seed, ntop=2):
    if design == "iid":
        rs = np.random.default_rng(None if shift_seed is None else [shift_seed, zlib.crc32(key.encode())])
        return rs.random((n, dim))
    if design == "sobol":
        P = _sobol(n, dim) + 0.5 / max(n, 1)                      # centre the (0-anchored) net
        return (P + _shift(key, dim, shift_seed)) % 1.0
    if design == "prod":
        # ntop top coordinates on a Gauss-Legendre grid; remaining coordinates: shifted Sobol (Latin-like)
        g = int(round((n) ** (1.0 / ntop) * 0 + 0))
        raise NotImplementedError
    raise ValueError(design)


def r23(model, n_per, design, shift_seed):
    tot, cost = 0.0, 0.0
    for (a, b, p) in TRIP:
        U = unit_points(design, n_per, 9, f"r23{a}{b}{p}", shift_seed)
        u = C.q23(U[:, 0]); o1 = C.tn_above(U[:, 1], u); bath = C.tn_below(U[:, 2:], u[:, None])
        n = len(U); y = np.empty((n, 10)); y[:, a] = u; y[:, b] = u; y[:, p] = o1
        rest = [k for k in range(10) if k not in (a, b, p)]; y[:, rest] = bath
        tau = np.maximum(C.SQ2 * np.minimum((o1 - u) / 3, u - bath.max(1)), 0)
        tex = np.where(u < 0, -C.SQ2 * u, 0.0)
        m, c = C.walk_mass(model, y, np.full(n, a), np.full(n, b), tau, tex)
        tot += m.mean(); cost += c.sum()
    return C.Z23 * 2 * np.sqrt(np.pi) * tot, cost


def r12(model, n_per, design, shift_seed):
    tot, cost = 0.0, 0.0
    for (a, b) in PAIR:
        U = unit_points(design, n_per, 9, f"r12{a}{b}", shift_seed)
        w = C.q12(U[:, 0]); bath = C.tn_below(U[:, 1:], w[:, None])
        n = len(U); y = np.empty((n, 10)); y[:, a] = w; y[:, b] = w
        rest = [k for k in range(10) if k not in (a, b)]; y[:, rest] = bath
        tau = np.maximum(C.SQ2 * np.minimum((w - bath.max(1)) / 3, w), 0)
        m, c = C.walk_mass(model, y, np.full(n, a), np.full(n, b), tau, np.zeros(n), top=True)
        tot += m.mean(); cost += c.sum()
    return C.Z12 * 2 * np.sqrt(np.pi) * tot, cost


def blind(model, n_per, design, shift_seed):
    """P(err & 2nd largest <= 0): 11 exact cases (no positive input, or one positive at position p)."""
    tot, cost = 0.0, 0.0
    for p in range(-1, 10):
        U = unit_points(design, n_per, 10, f"blind{p}", shift_seed)
        x = C.tn_below(U, 0.0)
        if p >= 0:
            x[:, p] = C.tn_above(U[:, p], 0.0)
        err = C.forward(model, x).argmax(1) != np.argsort(x, 1)[:, -2]
        tot += err.mean() / 1024; cost += len(x)
    return tot, cost


def estimate(model, budget, design="sobol", shift_seed=None, shares=SHARES):
    n23 = max(1, int(budget * shares["r23"] / (360 * COST23)))
    n12 = max(1, int(budget * shares["r12"] / (45 * COST12)))
    nb = max(1, int(budget * shares["blind"] / 11))
    e23, c23 = r23(model, n23, design, shift_seed)
    e12, c12 = r12(model, n12, design, shift_seed)
    eb, cb = blind(model, nb, design, shift_seed)
    return 1 - (e23 + e12 + eb), c23 + c12 + cb, dict(r23=e23, r12=e12, blind=eb, n=(n23, n12, nb))
