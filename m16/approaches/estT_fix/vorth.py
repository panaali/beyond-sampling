"""Vectorised 4-D Plackett orthant mass for a batch of simplicial cones with arbitrary (t, weight*dt) nodes."""
import numpy as np
IU = [(i, j) for i in range(4) for j in range(i + 1, 4)]
KL = [tuple(q for q in range(4) if q not in (i, j)) for i, j in IU]


def corr(Rays):
    B = np.linalg.inv(np.swapaxes(Rays, 1, 2)); S = B @ np.swapaxes(B, 1, 2)
    dg = np.sqrt(np.einsum('nii->ni', S)); return S / (dg[:, :, None] * dg[:, None, :])


def orth_v(C, t, w):
    """C (n,4,4) correlations; nodes t (K,), weights w (K,) incl. jacobian."""
    n = len(C); r = np.stack([C[:, i, j] for i, j in IU], 1).clip(-1, 1)          # (n,6)
    p = 1 / 16 + np.arcsin(r).sum(1) / (8 * np.pi)
    Rt = np.eye(4) + t[None, :, None, None] * (C - np.eye(4))[:, None]          # (n,K,4,4)
    Om = np.linalg.inv(Rt); acc = np.zeros(n)
    for q, (i, j) in enumerate(IU):
        k, l = KL[q]
        pc = (-Om[..., k, l] / np.sqrt(Om[..., k, k] * Om[..., l, l])).clip(-1, 1)
        acc += (w * r[:, q:q + 1] * np.arcsin(pc) / np.sqrt(np.maximum(1e-300, 1 - (t * r[:, q:q + 1]) ** 2))).sum(1)
    return p + acc / (4 * np.pi ** 2)


def rule_pow(n, p):
    x, w = np.polynomial.legendre.leggauss(n); s = 0.5 * (x + 1)
    return 1 - (1 - s) ** p, 0.5 * w * p * (1 - s) ** (p - 1)


def rule_log(n, V=70.0):
    """1 - t = exp(-v), v in [0, V]: resolves features at every scale of 1 - t."""
    x, w = np.polynomial.legendre.leggauss(n); v = 0.5 * V * (x + 1)
    return 1 - np.exp(-v), 0.5 * V * w * np.exp(-v)
