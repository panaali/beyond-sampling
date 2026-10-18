"""T317d (T322): T317c with T320's three convention gaps closed; no change to the walk, the merge or the Phi profile.
(1) Every model-free quantity is precomputed ONCE at import into TABLES, outside est. That covers the GL nodes and
    weights, Phi(nodes), Phi(nodes)^8, the blind table 1 - Phi(nodes), the density 90 phi(nodes), the transverse
    points, the v4 node, mu(v4), dmu(v4), and the 720 line inputs and directions.
(2) TABLES is listed with sizes by table_sizes() (and in the README).
(3) The transverse points are the exact-law conditional means given v2 in the region (deterministic quadrature),
    not the seeded Monte Carlo of r1w2._regions.
est(model, budget=None) -> (estimate, FLOPs as executed); the charges are cont_c's."""
import os
import sys

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_c  # noqa: E402
from cont_c import r0, r1w, r1w2, TMAX  # noqa: E402  (r1w2 is imported by cont; its MC REG is NOT used here)

DEDUCTIVE = True
G = 20
REGIONS = (("nb", 0.0, 6.5), ("b", -6.5, 0.0))          # v2 ranges, as in r1w2.REG[False]


def exact_points(lo, hi, n=400_001):
    """E[v2], E[v1 - v2], E[v2 - v3] for the top-3 of 10 iid N(0,1) given v2 in (lo, hi], by deterministic quadrature.
    Density of v2: 90 phi (1 - Phi) Phi^8. Given v2: E[v1 - v2] = (phi - v2 (1 - Phi)) / (1 - Phi) (closed form);
    E[v2 - v3] = int_{-inf}^{v2} (Phi(v) / Phi(v2))^8 dv (computed as a cumulative integral on a fine grid)."""
    v = np.concatenate([np.linspace(-14.0, lo, n), np.linspace(lo, hi, n)[1:]])      # lo is a grid node
    P = ndtr(v); p = norm.pdf(v); Q = ndtr(-v)
    P8 = P ** 8; dv = np.diff(v)
    cum8 = np.concatenate([[0.0], np.cumsum((P8[1:] + P8[:-1]) * dv / 2)])        # int_{-14}^{v} Phi^8 (tail < 1e-40)
    f = 90 * p * Q * P8
    w = np.zeros(len(v)); w[n - 1:-1] += dv[n - 1:] / 2; w[n:] += dv[n - 1:] / 2    # trapezoid weights on [lo, hi]
    Z = (w * f).sum()
    v2 = (w * f * v).sum() / Z
    g12 = (w * 90 * p * P8 * (p - v * Q)).sum() / Z
    g23 = (w * 90 * p * Q * cum8).sum() / Z
    return float(v2), float(g12), float(g23)


def _tables(points="exact"):
    out = []
    for name, lo, hi in REGIONS:
        if points == "exact":
            v2s, g12s, g23s = exact_points(lo, hi)
        else:                                                                       # "mc": T317c's points (checks)
            v2s, g12s, g23s = next(r[3] for r in r1w2.REG[False] if r[0] == name)
        nodes, wts = cont_c.cont._gl(lo, hi, G)
        v3s = v2s - g23s; v4s = float(r1w.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
        m4 = float(r0.mu(np.array([v4s]))[0]); dm4 = float(r1w.dmu(v4s)[0])
        t = dict(name=name, nodes=nodes, wts=wts, phi0=dict(p=ndtr(nodes), m8=ndtr(nodes) ** 8),
                 blind12=1.0 - ndtr(nodes), dens=90 * norm.pdf(nodes), point=(v2s, g12s, g23s), v4=v4s, m4=m4, dm4=dm4)
        t["line23"] = r1w.inputs(v2s + g12s, v2s, v2s, v2s - delta, m4 + dm4 * (v2s - delta - v4s), 0.0, -1.0, -1.0, -dm4)
        if name.startswith("nb"):
            t["line12"] = r1w.inputs(v2s, v2s, v3s, v4s, m4, 1.0, 0.0, 0.0, 0.0)
        out.append(t)
    return out


TABLES = _tables("exact")                                 # model-free, computed once at import, uncharged
TABLES_MC = None                                          # built lazily, only for the equality check against T317c


def table_sizes(tables=None):
    rows = []
    for t in tables or TABLES:
        for k, v in t.items():
            if k == "name":
                continue
            if isinstance(v, dict):
                rows += [(t["name"], f"{k}.{kk}", np.size(vv)) for kk, vv in v.items()]
            elif isinstance(v, tuple) and k.startswith("line"):
                rows += [(t["name"], f"{k}.x", v[0].size), (t["name"], f"{k}.dx", v[1].size)]
            else:
                rows.append((t["name"], k, int(np.size(v))))
    return rows


def est(model, budget=None, points="exact"):
    global TABLES_MC
    if points == "mc" and TABLES_MC is None:
        TABLES_MC = _tables("mc")
    tabs = TABLES if points == "exact" else TABLES_MC
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = r1w.TRI[:, 1]; NT = len(k); flops = 0.0; tot = np.zeros(NT)
    for t in tabs:
        nodes = t["nodes"]
        x, dx = t["line23"]
        p23, f = cont_c.walk_counted(model, x, dx, k, TMAX); flops += f
        R, A, B, ops = cont_c.merge(p23, NT, TMAX); flops += ops
        I23, f = cont_c.profile(R, A, B, NT, nodes, "23", TMAX, t["phi0"]); flops += f
        if "line12" in t:
            x, dx = t["line12"]
            p12, f = cont_c.walk_counted(model, x, dx, k, TMAX); flops += f
            R, A, B, ops = cont_c.merge(p12, NT, TMAX); flops += ops
            I12, f = cont_c.profile(R, A, B, NT, nodes, "12", TMAX, t["phi0"]); flops += f
        else:
            I12 = np.repeat(t["blind12"][None], NT, 0)                              # blind: model-free (V1)
        tot += (I12 * I23 * t["dens"][None]) @ t["wts"]
        flops += NT * len(nodes) * 4
    flops += NT
    return float(tot.mean()), float(flops)
