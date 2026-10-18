"""T360: own implementation of T340c (variant B with g12 integrated by a K-node Gauss rule of g12 | v2), on my T312
walk. Deductive; cost charged as executed (walk cells per T312 + Phi at weight-dependent endpoints)."""
import sys
import numpy as np
from scipy.special import ndtr

sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/rare_check")
import rare as R  # noqa: E402  (my T350 helpers: line23, node_point, NODES, walk, TRI)

FPE, F_PHI, TEND = 5948.0, 20.0, R.TEND


def gauss_g12(v2, t, K):
    """K-node Gauss rule for g | v2 with density proportional to phi(v2 + g) on g >= t (Stieltjes on a fine grid).
    Returns nodes, weights summing to 1 - Phi(v2 + t)."""
    g = np.linspace(t, t + 12, 60001); w = R.phi(v2 + g); w = w / w.sum()
    a = np.zeros(K); b = np.zeros(K); p_prev = np.zeros_like(g); p = np.ones_like(g); nrm_prev = 1.0
    for j in range(K):
        nrm = (w * p * p).sum(); a[j] = (w * g * p * p).sum() / nrm
        if j > 0:
            b[j] = nrm / nrm_prev
        p_new = (g - a[j]) * p - (b[j] if j > 0 else 0.0) * p_prev
        p_prev, p, nrm_prev = p, p_new, nrm
    J = np.diag(a) + np.diag(np.sqrt(b[1:]), 1) + np.diag(np.sqrt(b[1:]), -1)
    x, U = np.linalg.eigh(J)
    return x, U[0] ** 2 * (1 - ndtr(v2 + t))


def _flat(ints):
    r, a, b = [], [], []
    for i, iv in enumerate(ints):
        for lo, hi in iv:
            r.append(i); a.append(lo); b.append(hi)
    return np.array(r, int), np.array(a), np.array(b)


def m23(v2, a, b):
    return (ndtr(v2 - a) ** 8 - ndtr(v2 - b) ** 8) / 8


def lines(v2, g12_nodes):
    _, v3, v4, m4, dm4 = R.node_point(v2)
    X, D = [], []
    for g in g12_nodes:
        x0, d = R.line23(v2, g, v3, v4, m4, dm4); X.append(x0); D.append(d)
    return np.concatenate(X), np.concatenate(D)


def est(model, K, t0s=(1.0, 1.25, 1.5), ts=(0.5, 0.75, 1.0, 1.25)):
    """Returns ({t0: E1}, C_E1 in FPE, {t: E2}, C_E2 in FPE)."""
    e1 = {t0: 0.0 for t0 in t0s}; e2 = {t: 0.0 for t in ts}; c1 = c2 = 0.0; NT = R.NT
    for reg in ("nb", "b"):
        for v2, wv in zip(*R.NODES[reg]):
            dens = wv * R.phi(v2)
            if t0s:                                                     # E1: label p3, g12 law on g >= 0
                gn, gw = gauss_g12(v2, 0.0, K)
                X, D = lines(v2, gn)
                ints, c = R.walk(model, X, D, np.tile(R.TRI[:, 2], K)); c1 += c
                r, a, b = _flat(ints); c1 += 2 * len(r) * F_PHI / FPE
                wrow = np.repeat(gw, NT)[r]
                for t0 in t0s:
                    lo = np.maximum(a, t0); ok = b > lo
                    e1[t0] += dens * (wrow[ok] * m23(v2, lo[ok], b[ok])).sum()
            for t in ts:                                                # E2(t): label p2, g12 law on g >= t
                gn, gw = gauss_g12(v2, t, K)
                X, D = lines(v2, gn)
                ints, c = R.walk(model, X, D, np.tile(R.TRI[:, 1], K)); c2 += c
                r, a, b = _flat(ints)
                lo = np.maximum(a, t); hi = np.minimum(b, TEND); ok = hi > lo
                c2 += 2 * ok.sum() * F_PHI / FPE
                wrow = np.repeat(gw, NT)
                full = wrow.sum() * m23(v2, t, TEND)                    # [t, 6] for every row (model-free)
                corr = (wrow[r[ok]] * m23(v2, lo[ok], hi[ok])).sum()
                e2[t] += dens * (full - corr)
    return e1, c1, e2, c2
