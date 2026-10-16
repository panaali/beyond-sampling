"""est_boundary(model, budget, rng) -> (accuracy estimate, compute used): boundary-layer estimator.

error = A + B + P_R * rate_R + (11/1024) * rate_blind
  A: 2nd-vs-3rd tie-surface integral of the exact misclassified mass in a band |x_(2)-x_(3)| < gA (zones2)
  B: 1st-vs-2nd tie-surface integral, band |x_(1)-x_(2)| < gB outside A
  R: plain sampling of the rest (non-blind), P_R by exact 1-D quadrature
  blind: plain sampling of {x_(2) <= 0}, probability exactly 11/1024
Compute is counted in forward-pass equivalents (each recurrent step's matvec = 0.1; a pass carrying a slope
counts value + slope). Two-phase Neyman allocation: a pilot (25% of the budget) at fixed default fractions,
then the rest allocated by the pilot's per-term s.d. and cost. All samples are used in the final means.
"""
import sys
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
import zones as Z
import zones2 as Z2

GA, GB = 0.15, 0.1
P_BLIND = 11 / 1024
P_R = quad(lambda v: 720 * norm.pdf(v) * norm.sf(v + GB) * norm.cdf(v - GA) ** 8 / 8, 0, 12, epsabs=1e-14)[0]
DEFAULT_FRAC = dict(A=0.62, B=0.12, R=0.22, blind=0.04)     # from the pilot on the original model only
COST0 = dict(A=2.9, B=1.9, R=1.0, blind=1.0)


def _zone_draw(n, rng, zone):
    out = []; got = 0
    while got < n:
        x = rng.standard_normal((max(4 * n, 1000) if zone == 3 else max(120 * n, 10000), 10))
        x = x[Z.zone_of(x, GA, GB) == zone]; out.append(x); got += len(x)
    return np.concatenate(out)[:n]


def _run(model, k, n, rng):
    if n <= 0:
        return np.zeros(0), 0.0
    if k == "A":
        v, _, f = Z2.term(model, Z2.geom_A(model, n, rng, GA), rng); return v, f.sum()
    if k == "B":
        v, _, f = Z2.term(model, Z2.geom_B(model, n, rng, GA, GB), rng); return v, f.sum()
    x = _zone_draw(n, rng, 3 if k == "R" else 0)
    return Z.point_wrong(model, x).astype(float), float(n)


WEIGHT = dict(A=1.0, B=1.0, R=P_R, blind=P_BLIND)


def est_boundary(model, budget, rng, pilot=0.25, details=False):
    vals = {k: [] for k in WEIGHT}; cost = {k: 0.0 for k in WEIGHT}
    for k in WEIGHT:
        n = max(8, int(pilot * budget * DEFAULT_FRAC[k] / COST0[k]))
        v, c = _run(model, k, n, rng); vals[k].append(v); cost[k] += c
    # Neyman: budget share ∝ W_k s_k sqrt(c_k)
    s, c = {}, {}
    for k in WEIGHT:
        v = np.concatenate(vals[k]); n = len(v)
        c[k] = cost[k] / n
        sd = v.std(ddof=1)
        if k in ("R", "blind"):                       # Bernoulli: floor with a +1/+2 prior
            p = (v.sum() + 1) / (n + 2); sd = max(sd, np.sqrt(p * (1 - p)))
        s[k] = WEIGHT[k] * max(sd, 1e-6)
    share = {k: s[k] * np.sqrt(c[k]) for k in WEIGHT}; tot = sum(share.values())
    for k in WEIGHT:
        target_n = int(budget * share[k] / tot / c[k])
        have = sum(len(v) for v in vals[k])
        v, cc = _run(model, k, target_n - have, rng); vals[k].append(v); cost[k] += cc
    terms = {k: WEIGHT[k] * np.concatenate(vals[k]).mean() for k in WEIGHT}
    err = sum(terms.values())
    used = sum(cost.values())
    if details:
        return 1 - err, used, terms, {k: sum(len(v) for v in vals[k]) for k in WEIGHT}, cost
    return 1 - err, used
