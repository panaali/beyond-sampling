"""Gaussian / Edgeworth orthant probabilities P(m > 0) from cumulants (orders 1-4) of a k-vector m (T80).
Integration by a fixed set of plain Gaussian draws of the explicit endpoint density (no model calls).
Checked on known distributions in test_endpoint.py."""
import numpy as np

NG = 1 << 22


def cumulants(m):
    """mean, cov, kappa3, kappa4 of rows of m (n, k)."""
    mu = m.mean(0); c = m - mu; n = len(m)
    C = c.T @ c / n
    c2 = (c[:, :, None] * c[:, None, :]).reshape(n, -1)
    k3 = (c2.T @ c).reshape((m.shape[1],) * 3) / n
    m4 = (c2.T @ c2).reshape((m.shape[1],) * 4) / n
    k4 = m4 - np.einsum("ij,kl->ijkl", C, C) - np.einsum("ik,jl->ijkl", C, C) - np.einsum("il,jk->ijkl", C, C)
    return mu, C, k3, k4


_G = {}


def gauss_draws(k):
    if k not in _G:
        _G[k] = np.random.default_rng(777 + k).standard_normal((NG, k))
    return _G[k]


def endpoint(mu, C, k3, k4):
    """P(m > 0) under Gaussian / Edgeworth densities with the given cumulants; returns dict."""
    k = len(mu)
    ev, U = np.linalg.eigh(C)
    ev = np.maximum(ev, 1e-14 * ev.max())
    L = U * np.sqrt(ev)               # C = L L^T
    Li = (U / np.sqrt(ev)).T          # whitening
    A = np.einsum("ia,jb,kc,abc->ijk", Li, Li, Li, k3)
    B = np.einsum("ia,jb,kc,ld,abcd->ijkl", Li, Li, Li, Li, k4)
    alpha = np.einsum("iik->k", A)
    AA = np.einsum("ijk,ijl->kl", A, A); F = (A * A).sum()
    Bd = np.einsum("iikl->kl", B); B0 = np.einsum("iijj->", B)
    s = {"gauss": 0.0, "edge3": 0.0, "edge4": 0.0}
    W = gauss_draws(k)
    if not (np.any(k3) or np.any(k4)):            # Gaussian only: no Edgeworth sums needed
        p = sum(int(((W[c0:c0 + (1 << 18)] @ L.T + mu) > 0).all(1).sum()) for c0 in range(0, NG, 1 << 18)) / NG
        return {"gauss": p, "edge3": p, "edge4": p}
    for c0 in range(0, NG, 1 << 18):
        w = W[c0:c0 + (1 << 18)]
        wi = w[((w @ L.T + mu) > 0).all(1)]
        if len(wi) == 0:
            continue
        a3 = np.einsum("ijk,ni,nj,nk->n", A, wi, wi, wi, optimize=True)
        a1 = wi @ alpha
        he3 = a3 - 3 * a1
        u = np.einsum("ijk,nj,nk->ni", A, wi, wi, optimize=True)
        c33 = (a3 ** 2 - 6 * a1 * a3 - 9 * (u * u).sum(1) + 9 * a1 ** 2 + 18 * (u @ alpha)
               + 18 * np.einsum("nk,kl,nl->n", wi, AA, wi) - 9 * alpha @ alpha - 6 * F)
        b4 = np.einsum("ijkl,ni,nj,nk,nl->n", B, wi, wi, wi, wi, optimize=True)
        he4 = b4 - 6 * np.einsum("nk,kl,nl->n", wi, Bd, wi) + 3 * B0
        s["gauss"] += len(wi); s["edge3"] += (1 + he3 / 6).sum(); s["edge4"] += (1 + he3 / 6 + he4 / 24 + c33 / 72).sum()
    return {kk: v / NG for kk, v in s.items()}


