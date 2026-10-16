"""Laplace-tube quadrature (LTQ): deterministic estimator of M_16,10-type accuracy.

accuracy = 1 - P_BLIND * e_blind - sum_{360 triples} (Z23/360) E_face[F] - sum_{45 pairs} (Z12/45) E_face[F]
(exact co-area identity of ../tube; Z's are model-free). Differences from ../tube/estimator.py:
  * every position triple / pair gets ITS OWN 9-D (+2 tail) low-discrepancy point set (rotated scrambled
    Sobol with a fixed per-stratum shift), instead of one 10-D Sobol whose 10th coordinate picks the triple;
    per-stratum point sets keep the QMC structure in the face coordinates (diag_qmc9.py: ~2x variance gain);
  * no importance tables fitted on any model (uniform strata, model-free masses);
  * the normal line is walked exactly only for slopes |kappa| < k0 (where ~99% of the band mass lies),
    the rest of the segment gets one deterministic quadrature node (tube.F_hybrid).
seed selects the (fixed) point-set randomisation; the leaderboard entry uses seed=0 (deterministic).
"""
import numpy as np
from scipy.stats import qmc
from lp import *
from lp import _trunc_below, _trunc_above, _inv
from tube import F_hybrid, blind_points, forward_from, P_BLIND, Z23, Z12, CDF12, PAIRS

PAIRS = np.array(PAIRS)


def _sets(nstrata, m, d, seed):
    """(nstrata, m, d) points: one scrambled Sobol base of size m, rotated per stratum by a fixed shift."""
    rs = np.random.default_rng(seed)
    base = qmc.Sobol(d, scramble=True, seed=int(rs.integers(1 << 31))).random(m)
    sh = rs.random((nstrata, 1, d))
    return (base[None] + sh) % 1.0


def build12(pair, w, rest):
    n = len(w); x0 = np.empty((n, T)); d = np.zeros((n, T))
    for k in range(n):
        a, b = pair[k]; oth = [i for i in range(T) if i not in (a, b)]
        x0[k, a] = x0[k, b] = w[k]; x0[k, oth] = rest[k]; d[k, a], d[k, b] = 0.5, -0.5
    hi = np.minimum(2 * (w - rest.max(1)) / 3, 2 * w)
    return dict(x0=x0, d=d, lo=np.zeros(n), hi=hi, tpos=pair[:, 1], tneg=pair[:, 0])


def e23(model, m, seed, k0):
    P = _sets(360, m, 11, seed).reshape(-1, 11)
    combo = np.repeat(COMBO, m, 0)
    u = _inv(CDF23, P[:, 0]); xc = _trunc_above(u, P[:, 1]); rest = _trunc_below(u[:, None], P[:, 2:9])
    f = build23(combo, u, xc, rest)
    F, C = F_hybrid(model, f, k0, P[:, 9:11])
    return Z23 * F.mean(), C.sum(), F


def e12(model, m, seed, k0):
    P = _sets(45, m, 11, seed + 7919).reshape(-1, 11)
    pair = np.repeat(PAIRS, m, 0)
    w = _inv(CDF12, P[:, 0]); rest = _trunc_below(w[:, None], P[:, 1:9])
    f = build12(pair, w, rest)
    F, C = F_hybrid(model, f, k0, P[:, 9:11])
    return Z12 * F.mean(), C.sum(), F


def eblind(model, n, seed):
    rs = np.random.default_rng(seed + 104729)
    xb = blind_points(n, shift=rs.random(11))
    return P_BLIND * float((forward_from(model, xb).argmax(1) != np.argsort(xb, 1)[:, -2]).mean()), n
