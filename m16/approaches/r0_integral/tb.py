"""T297 integrator TB (deterministic; no random or Sobol nodes): per position triple, R0's flip thresholds along the gap
directions by Newton steps on the exact margin (value + forward-mode tangent), and model-free exact masses of the
region assumed correct, by a fixed Gauss-Legendre rule in v2.
Variant (n23, newton, nblind): number of v2 anchors for tau23(v2), Newton steps per root, blind verdicts (1 or 2).
est(model, n23=1, newton=1, nblind=1) -> (acc_R0 estimate, charged FLOPs)."""
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

import r0

CAP = 3.0
G_NODES, G_LO, G_HI = 48, 0.0, 6.5
_t, _w = np.polynomial.legendre.leggauss(G_NODES)
V2N = G_LO + (G_HI - G_LO) * (_t + 1) / 2; V2W = (G_HI - G_LO) / 2 * _w
FLOP_PHI = 20


def _model_free_constants():
    """Anchors from the exact law (model-free): quantiles of v2 | v2 > 0, median gaps, blind canonical skeletons."""
    v = np.linspace(-6, 8, 200001); dv = v[1] - v[0]
    f2 = 90 * norm.pdf(v) * (1 - ndtr(v)) * ndtr(v) ** 8
    pos = v > 0; c = np.cumsum(f2 * pos) * dv; c /= c[-1]
    q = lambda p: float(np.interp(p, c, v))
    rng = np.random.default_rng(0)                      # model-free table of order statistics (no model calls)
    x = -np.sort(-rng.standard_normal((400_000, 10)), 1)[:, :3]
    nb = x[:, 1] > 0
    g12 = float(np.median((x[:, 0] - x[:, 1])[nb])); g23 = float(np.median((x[:, 1] - x[:, 2])[nb]))
    b1 = (x[:, 1] <= 0) & (x[:, 0] > 0); b0 = x[:, 0] <= 0
    return dict(q=q, g12=g12, g23=g23, blind1=x[b1].mean(0), blind0=x[b0].mean(0),
                P_b1=10 / 1024, P_b0=1 / 1024)


MF = _model_free_constants()


def anchors(n23):
    return [MF["q"]((i + 0.5) / n23) for i in range(n23)]


def _margin(model, tri, v1, v2, v3, d1, d3, dfree):
    """R0 margin L_p2 - max_{j != p2} L_j and its derivative along (dv1, dv3, dmu) = (d1, d3, dfree)."""
    x = r0.r0_inputs(tri, v1, v2, v3)
    dx = np.repeat(dfree[:, None], r0.T, 1); r = np.arange(len(tri))
    dx[r, tri[:, 0]] = d1; dx[r, tri[:, 1]] = 0.0; dx[r, tri[:, 2]] = d3
    L, dL = r0.logits(model, x, dx)
    L2 = L.copy(); L2[r, tri[:, 1]] = -np.inf; j = L2.argmax(1)
    return L[r, tri[:, 1]] - L[r, j], dL[r, tri[:, 1]] - dL[r, j]


def dmu(v3):
    m = r0.mu(v3)
    return -v3 * m + m * m


def root23(model, tri, v2a, g12a, newton):
    n = len(tri); g = np.full(n, 1e-6); v1 = np.full(n, v2a + g12a); v2 = np.full(n, v2a); evals = 0
    tau = np.zeros(n); done = np.zeros(n, bool)
    for it in range(newton):
        v3 = v2 - g
        m, dm = _margin(model, tri, v1, v2, v3, np.zeros(n), -np.ones(n), -dmu(v3)); evals += 2
        if it == 0:
            pos0 = m > 0; tau[pos0] = 0.0; done |= pos0
        step = np.where(dm > 0, -m / np.where(dm > 0, dm, 1), np.inf)
        newg = np.clip(g + step, 0, CAP)
        g = np.where(done, g, newg)
    tau[~done] = g[~done]
    return tau, evals


def root12(model, tri, v2a, g23a, newton):
    n = len(tri); g = np.full(n, 1e-6); v2 = np.full(n, v2a); v3 = np.full(n, v2a - g23a); evals = 0
    tau = np.zeros(n); done = np.zeros(n, bool)
    for it in range(newton):
        m, dm = _margin(model, tri, v2 + g, v2, v3, np.ones(n), np.zeros(n), np.zeros(n)); evals += 2
        if it == 0:
            pos0 = m > 0; tau[pos0] = 0.0; done |= pos0
        step = np.where(dm > 0, -m / np.where(dm > 0, dm, 1), np.inf)
        g = np.where(done, g, np.clip(g + step, 0, CAP))
    tau[~done] = g[~done]
    return tau, evals


def est(model, n23=1, newton=1, nblind=1, use12=True, return_parts=False):
    tri = r0.TRIPLES; n = len(tri); flops = 0.0
    A = anchors(n23); taus = []
    for a in A:
        t, e = root23(model, tri, a, MF["g12"], newton); taus.append(t); flops += e * n * r0.FLOP_FWD
    if use12:
        t12, e = root12(model, tri, MF["q"](0.5), MF["g23"], newton); flops += e * n * r0.FLOP_FWD
    else:
        t12 = np.zeros(n)                                       # cheap variant: no 1st-vs-2nd boundary
    # tau23(v2): piecewise-linear through the anchors (constant outside), at the GL nodes
    T23 = np.array([np.interp(V2N, A, [tt[i] for tt in taus]) if n23 > 1 else np.full(G_NODES, taus[0][i])
                    for i in range(n)])                                   # (720, G)
    integrand = 90 * norm.pdf(V2N)[None] * (1 - ndtr(V2N[None] + t12[:, None])) * ndtr(V2N[None] - T23) ** 8
    M = integrand @ V2W
    flops += n * G_NODES * (2 * FLOP_PHI + 12)
    # blind part: one verdict at the blind canonical skeleton (v1 > 0 >= v2), or two (also v1 <= 0)
    cases = ([] if nblind == 0 else [("blind1", MF["P_b1"] + (MF["P_b0"] if nblind == 1 else 0.0))]) + \
        ([("blind0", MF["P_b0"])] if nblind == 2 else [])                 # nblind = 0: blind assumed wrong
    B = np.zeros(n)
    for key, P in cases:
        v = MF[key]
        c = r0.r0_correct(model, tri, np.full(n, v[0]), np.full(n, v[1]), np.full(n, v[2])); flops += n * r0.FLOP_FWD
        B += P * c
    acc = float((M + B).mean())
    if return_parts:
        return acc, flops, dict(M=M, B=B, t12=t12, t23=np.array(taus))
    return acc, flops


if __name__ == "__main__":
    # sanity: with all thresholds 0 and blind correct, the masses must sum to 1
    s = (90 * norm.pdf(V2N) * (1 - ndtr(V2N)) * ndtr(V2N) ** 8) @ V2W
    print("nonblind mass", s, "expected", 1 - 11 / 1024, "anchors", anchors(1), anchors(2), "g12", MF["g12"], "g23", MF["g23"])
