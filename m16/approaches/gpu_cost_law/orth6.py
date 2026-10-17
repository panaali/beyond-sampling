"""T192: Gaussian mass of 6-D simplicial cones (needed for exact enumeration / recipe108 at T = 6), batched torch.

Wide simplices (|det R| >= THIN6): Plackett along R(t) = I + t (C - I) (C = correlation of the dual normals, as
estT): P6 = 1/64 + sum_ij asin(c_ij) / (32 pi) + sum_ij int_0^1 c_ij / (2 pi sqrt(1 - t^2 c_ij^2)) (P4(G_ij(t)) - 1/16) dt,
G_ij(t) = correlation of the other 4 variables given (i, j) under R(t). Entry (k, l) of G is the partial correlation
from the cofactors of the 4x4 block {i,j,k,l} (exactly the d = 5 quantities of estT._orth5), with
1 - G_kl^2 = det(R_Q) (1 - t^2 c_ij^2) / (cof_kk cof_ll). P4(G) - 1/16 is estT's d = 4 Plackett form applied to G
(asin and 1 - |G| from the accurate expressions above). Both integrals use estT's log nodes 1 - t = e^-v, v in [0,70].
Thin (|det R| < THIN6): mass = |det R| / pi^3 int_simplex |R^T u|^-6 du (Duffy product Gauss rule), as estT for d=4,5.
Cost (ops charge, m43 convention): see OPS6 below.
"""
import itertools
import math

import numpy as np
import torch

THIN6 = [1e-6]
WIDE6 = [1e-2]    # T192 calibration: ray-space rule with NMID^5 nodes for THIN6 <= |det| < WIDE6 (max rel 5.7e-7 vs
NMID = [8]        # nested Plackett on the 120 widest of 1.15M (8,6) zoo0 simplices); Plackett only above WIDE6
NT = [128]        # outer log nodes
NS = [128]        # inner log nodes
NTHIN = [6]       # Duffy points per dimension (6^5 = 7776)
_C = {}


def _log(n, dev):
    xg, wg = np.polynomial.legendre.leggauss(n); vg = 35.0 * (xg + 1); S = np.exp(-vg)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    return t(S), t(1 - S), t(35.0 * wg * S)


def _duffy5(n, dev):
    x, w = np.polynomial.legendre.leggauss(n); x = (x + 1) / 2; w = w / 2
    a, b, c, e, f = np.meshgrid(x, x, x, x, x, indexing="ij")
    W = np.einsum("i,j,k,l,m->ijklm", w, w, w, w, w)
    u1 = a; u2 = (1 - a) * b; u3 = (1 - a) * (1 - b) * c; u4 = (1 - a) * (1 - b) * (1 - c) * e
    u5 = (1 - a) * (1 - b) * (1 - c) * (1 - e) * f
    J = (1 - a) ** 4 * (1 - b) ** 3 * (1 - c) ** 2 * (1 - e)
    U = np.stack([1 - u1 - u2 - u3 - u4 - u5, u1, u2, u3, u4, u5], -1).reshape(-1, 6)
    t = lambda q: torch.as_tensor(q, dtype=torch.float64, device=dev)
    return t(U), t((W * J).ravel())


def _consts(dev):
    key = (dev, NT[0], NS[0], NTHIN[0], NMID[0])
    if key not in _C:
        _C[key] = (_log(NT[0], dev), _log(NS[0], dev), _duffy5(NTHIN[0], dev), _duffy5(NMID[0], dev))
    return _C[key]


_IU = [(i, j) for i in range(6) for j in range(i, 6)]
_PHI = {}


def _ray(R, dt, DU, DW):
    """|det R| / pi^3 sum_k DW_k |R^T u_k|^-6, with |R^T u|^2 = u^T (R R^T) u evaluated as one GEMM over the 21 Gram
    entries (unit rays in one narrow cone: |R^T u|^2 >= ~0.4, no cancellation)."""
    key = (DU.data_ptr(), DU.shape[0])
    if key not in _PHI:
        _PHI[key] = torch.stack([DU[:, i] * DU[:, j] * (1.0 if i == j else 2.0) for i, j in _IU], 1)    # (K, 21)
    G = R @ R.transpose(1, 2)
    g = torch.stack([G[:, i, j] for i, j in _IU], 1)                                                    # (n, 21)
    q = g @ _PHI[key].T                                                                                  # (n, K)
    return dt / math.pi ** 3 * (DW[None] / q ** 3).sum(1)


def det3(M):
    a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def det4(M):
    out = 0.0
    for q in range(4):
        cols = [c for c in range(4) if c != q]
        out = out + (-1) ** q * M[..., 0, q] * det3(M[..., 1:, :][..., cols])
    return out


def cof4(M, k, l):
    rows = [q for q in range(4) if q != k]; cols = [q for q in range(4) if q != l]
    return (-1) ** (k + l) * det3(M[..., rows, :][..., cols])


def _p4_minus(G, oma, sasn, LS):
    """P4(G) - 1/16 for correlation matrices G (..., 4, 4), with accurate 1 - |G_kl| (oma, (..., 6)) and asin(G_kl)
    (sasn, (..., 6)) supplied; estT d = 4 Plackett form, inner log nodes LS."""
    S, Tt, W = LS
    I = torch.eye(4, dtype=G.dtype, device=G.device)
    Gs = G[..., None, :, :] + S[:, None, None] * (I - G)[..., None, :, :]      # (..., K, 4, 4)
    dR = torch.clamp(det4(Gs), min=0.0)
    acc = 0.0
    for p, (k, l) in enumerate(itertools.combinations(range(4), 2)):
        m, n = [q for q in range(4) if q != k and q != l]
        cmn = cof4(Gs, m, n)
        r = G[..., k, l]; ra = torch.clamp(r.abs(), max=1.0)
        om = (oma[..., p, None] + ra[..., None] * S) * (1 + ra[..., None] * Tt)
        apc = torch.atan2(-cmn, torch.sqrt(dR * om))
        acc = acc + (W * r[..., None] * apc / torch.sqrt(torch.clamp(om, min=1e-300))).sum(-1)
    return sasn.sum(-1) / (8 * math.pi) + acc / (4 * math.pi ** 2)


def _wide(N):
    """N (n, 6, 6) unit dual normals (rows). Returns masses (n,)."""
    dev = N.device; n = N.shape[0]
    (S, Tt, W), LS, _, _ = _consts(dev)
    C = N @ N.transpose(1, 2)
    I = torch.eye(6, dtype=N.dtype, device=dev)
    Rt = C[:, None] + S[None, :, None, None] * (I - C)[:, None]                 # (n, K, 6, 6)
    # 4x4 blocks Q (15 of them): det and cofactors, per node
    dQ, cQ = {}, {}
    for Q in itertools.combinations(range(6), 4):
        RQ = Rt[..., list(Q), :][..., :, list(Q)]
        dQ[Q] = torch.clamp(det4(RQ), min=0.0)
        for a_ in range(4):
            for b_ in range(a_, 4):
                cQ[Q, Q[a_], Q[b_]] = cof4(RQ, a_, b_)
    p = torch.full((n,), 1 / 64, dtype=N.dtype, device=dev)
    asum = torch.zeros(n, dtype=N.dtype, device=dev); acc = torch.zeros_like(asum)
    for i, j in itertools.combinations(range(6), 2):
        dm = torch.linalg.norm(N[:, i] - N[:, j], dim=1); dp = torch.linalg.norm(N[:, i] + N[:, j], dim=1)
        asum = asum + (math.pi / 2 - 2 * torch.atan2(dm, dp))
        oma_ij = torch.minimum(dm, dp) ** 2 / 2
        r = C[:, i, j]; ra = torch.clamp(r.abs(), max=1.0)
        om = (oma_ij[:, None] + ra[:, None] * S[None]) * (1 + ra[:, None] * Tt[None])     # 1 - t^2 r^2, (n, K)
        rest = [q for q in range(6) if q != i and q != j]
        G = torch.empty(n, len(S), 4, 4, dtype=N.dtype, device=dev)
        omaG = torch.empty(n, len(S), 6, dtype=N.dtype, device=dev); asnG = torch.empty_like(omaG)
        for a_ in range(4):
            G[..., a_, a_] = 1.0
        for p_, (a_, b_) in enumerate(itertools.combinations(range(4), 2)):
            k, l = rest[a_], rest[b_]
            Qkl = tuple(sorted((i, j, k, l)))
            ckl = cQ[Qkl, min(k, l), max(k, l)]; ckk = cQ[Qkl, k, k]; cll = cQ[Qkl, l, l]
            den = torch.sqrt(torch.clamp(ckk * cll, min=1e-300))
            g = -ckl / den
            one_m_g2 = torch.clamp(dQ[Qkl] * om / torch.clamp(ckk * cll, min=1e-300), min=0.0)  # Jacobi identity
            sq = torch.sqrt(one_m_g2)
            G[..., a_, b_] = g; G[..., b_, a_] = g
            asnG[..., p_] = torch.atan2(g, sq)
            omaG[..., p_] = one_m_g2 / (1 + torch.clamp(g.abs(), max=1.0))
        inner = _p4_minus(G, omaG, asnG, LS)                                    # (n, K)
        acc = acc + (W[None] * r[:, None] * inner / (2 * math.pi * torch.sqrt(torch.clamp(om, min=1e-300)))).sum(1)
    return p + asum / (32 * math.pi) + acc


def _normals(R):
    """unit facet normals opposite each ray (cofactors of R without row i, sign towards ray i)."""
    n, d, _ = R.shape
    N = torch.empty_like(R)
    for i in range(d):
        O = torch.cat([R[:, :i], R[:, i + 1:]], 1)
        cols = [torch.linalg.det(torch.cat([O[:, :, :q], O[:, :, q + 1:]], 2)) * (-1) ** q for q in range(d)]
        v = torch.stack(cols, 1)
        sgn = torch.where((v * R[:, i]).sum(1) > 0, 1.0, -1.0).to(R.dtype)
        N[:, i] = v * (sgn / torch.linalg.norm(v, dim=1))[:, None]
    return N


def orth6(R):
    """R (n, 6, 6) torch float64, rows = unit rays. Returns (masses (n,), ops charge)."""
    n = R.shape[0]; dev = R.device
    _, _, (DU, DW), (DM, WM) = _consts(dev)
    out = torch.empty(n, dtype=torch.float64, device=dev)
    dt = torch.linalg.det(R).abs()
    thin = dt < THIN6[0]; w = dt >= WIDE6[0]; mid = ~thin & ~w
    if thin.any():
        out[thin] = _ray(R[thin], dt[thin], DU, DW)
    if mid.any():
        out[mid] = _ray(R[mid], dt[mid], DM, WM)
    if w.any():
        out[w] = _wide(_normals(R[w]))
    nt, nm, nw = int(thin.sum()), int(mid.sum()), int(w.sum())
    return out, ops6(nt, nw, nm)


def ops6(nthin, nwide, nmid=0):
    """m43-style op charge, written in the same spirit as estT d = 4, 5 (multiply-add / sqrt / div / atan2 = 1):
    det 6^3; thin: per node 6x6 matvec + norm + power ~ 50; wide: normals 36 5x5 dets ~ 36*60, C 216, per outer node
    15 4x4 blocks (det + 10 cofactors ~ 11*15), 15 pairs x (6 partial corr ~ 6*8 + inner P4: per inner node det4 40 +
    6 pairs x (cof 15 + 20))."""
    K, Ks = NT[0], NS[0]
    wide = 216 + 36 * 60 + 216 + 15 * 24 + K * (15 * 11 * 15 + 15 * (6 * 8 + Ks * (40 + 6 * 35) + 6 * 20))
    return int(nthin * (216 + NTHIN[0] ** 5 * 50) + nmid * (216 + NMID[0] ** 5 * 50) + nwide * (216 + wide))


def masses6(R, device="cpu", chunk=512):
    tot = 0.0; ops = 0
    for a in range(0, len(R), chunk):
        m, o = orth6(torch.as_tensor(R[a:a + chunk], dtype=torch.float64, device=device)); tot += float(m.sum()); ops += o
    return tot, ops
