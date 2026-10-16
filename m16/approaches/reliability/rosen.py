"""Rosenblatt map u (iid N(0,1), R^10) -> sorted sample conditioned on the sign of the 2nd largest.

Breaks the homogeneity degeneracy: u = 0 maps to the median ('canonical') order statistics, and a
failure cone in x-space becomes a region with a finite-distance design point in u-space.
Order: u0 -> v2 (2nd largest, conditioned >0 or <0), u1 -> v1 | v2, u2..u9 -> v3..v10 top-down.
"""
import math
import torch

torch.set_default_dtype(torch.float64)
N = 10
SQ2 = math.sqrt(2.0)


def Phi(z):
    return 0.5 * torch.erfc(-z / SQ2)


def P2(F):            # CDF of 2nd largest of 10 in F-space: 10F^9 - 9F^10
    return 10 * F ** 9 - 9 * F ** 10


def dP2(F):
    return 90 * F ** 8 * (1 - F)


G0 = P2(torch.tensor(0.5)).item()          # P(v2 <= 0) = 11/1024


def v2_from_u(u0, positive=True):
    V = Phi(u0)
    c = G0 + V * (1 - G0) if positive else V * G0
    lo = torch.full_like(c, 0.5 if positive else 0.0); hi = torch.full_like(c, 1.0 if positive else 0.5)
    cd = c.detach()
    for _ in range(60):
        mid = (lo + hi) / 2
        m = P2(mid) < cd
        lo = torch.where(m, mid, lo); hi = torch.where(m, hi, mid)
    F0 = ((lo + hi) / 2).detach()
    F = F0 - (P2(F0) - c) / dP2(F0)        # one Newton step: attaches the exact first derivative
    return F


def sorted_from_u(u, positive=True):
    """u: (B,10) -> v: (B,10), v[:,0] >= v[:,1] >= ... (v[:,1] > 0 iff positive)."""
    F2 = v2_from_u(u[:, 0], positive)
    v2 = torch.special.ndtri(F2)
    # v1 | v2: truncated above v2. 1-F1 = (1-F2) * Phi(-u1)
    v1 = -torch.special.ndtri((1 - F2) * Phi(-u[:, 1]))
    out = [v1, v2]
    logF = torch.log(F2)
    for r in range(2, N):                  # v_{r+1} = max of (N-r) iid below previous
        k = N - r
        logF = logF + torch.special.log_ndtr(u[:, r]) / k
        out.append(torch.special.ndtri(torch.exp(logF)))
    return torch.stack(out, 1)
