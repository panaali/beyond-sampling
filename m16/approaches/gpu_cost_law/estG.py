"""T192: estT_base (verbatim copy of estT_fix/estT.py) with the cone masses deferred and batched.

Every simplicial cone that estT would integrate one at a time is pushed into a buffer; full buffers are integrated
together with the SAME formulas as estT.orthant / estT._orth5 (cofactor normals, atan2 angles, log-node Plackett
integral, thin rule below |det| 1e-6), vectorised in torch float64 on CPU or CUDA. No geometric decision of the recipe
depends on a mass, so cells, splits and the m43.Ops count are unchanged; only the summation order of the error differs.
Ops are charged per simplex exactly as estT charges them (branch thin / Plackett decided by the same |det R|).

usage: from estG import run; run(model, mode, device="cpu"|"cuda") -> dict(acc, ops, sec, cells, ...)
"""
import itertools
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "arc_m43_repro")]
import estT_base as B  # noqa: E402
from m43 import Ops  # noqa: E402

DEV = ["cpu"]
CHUNK = {4: 4096, 5: 1024}           # simplices per batched call (memory ~ CHUNK * K * d^2 * 8 bytes * few)
FLUSH = 200_000                      # simplices buffered per dimension before integration


class _Buf:
    def __init__(self):
        self.R = {}; self.n = {}; self.err = 0.0; self.ops = 0; self.simplices = 0

    def add(self, R):
        d = R.shape[0]
        if d not in self.R:
            self.R[d] = np.empty((FLUSH, d, d)); self.n[d] = 0
        self.R[d][self.n[d]] = R; self.n[d] += 1
        if self.n[d] == FLUSH:
            self.flush(d)

    def flush(self, d=None):
        for e in ([d] if d is not None else list(self.R)):
            n = self.n[e]
            if n:
                m, o = masses(self.R[e][:n], e)
                self.err += m; self.ops += o; self.simplices += n; self.n[e] = 0


BUF = [None]


def _tri_mass(c, ops):
    """estT.mass with orthant() replaced by a push into the buffer (triangulation unchanged)."""
    d = c.R.shape[1]
    if len(c.R) == d:
        BUF[0].add(c.R); return 0.0
    Tm = c.T

    def faces_of(S, Zset, k):
        out = set()
        for i in range(len(c.N)):
            if i in Zset:
                continue
            S2 = tuple(q for q in S if Tm[q, i])
            ops.add(len(S))
            if len(S2) >= k - 1 and S2 not in out:
                Z2 = np.all(Tm[list(S2)], axis=0)
                if d - np.linalg.matrix_rank(c.N[Z2]) == k - 1:
                    out.add(S2)
                ops.add(d ** 3)
        return out

    def tri(S, k):
        if len(S) == k:
            return [S]
        a = S[0]
        Z = set(i for i in range(len(c.N)) if all(Tm[q, i] for q in S))
        res = []
        for F in faces_of(S, Z, k):
            if a in F:
                continue
            for simp in tri(F, k - 1):
                res.append((a,) + simp)
        return res

    for simp in tri(tuple(range(len(c.R))), d):
        BUF[0].add(c.R[list(simp)])
    return 0.0


# ------------------------------------------------------------------------------------------ batched masses
_CACHE = {}


def _consts(d, dev):
    key = (d, dev)
    if key not in _CACHE:
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=dev)
        if d == 4:
            S, Tt, W, DU, DW = B._LOG_S, B._LOG_T, B._LOG_W, B._DU, B._DW
        else:
            S, Tt, W = B._LOG5[0]; DU, DW = B._DU5[0]
        _CACHE[key] = (t(S), t(Tt), t(W), t(DU), t(DW))
    return _CACHE[key]


def _cofactor_normals(R):
    """R (n, d, d) rows = rays. Row i of the result: unit normal of the facet opposite ray i, pointing to ray i
    (estT: cofactors of R without row i, sign by n . R_i)."""
    n, d, _ = R.shape
    N = torch.empty_like(R)
    for i in range(d):
        O = torch.cat([R[:, :i], R[:, i + 1:]], 1)                      # (n, d-1, d)
        cols = [torch.linalg.det(torch.cat([O[:, :, :q], O[:, :, q + 1:]], 2)) * (-1) ** q for q in range(d)]
        v = torch.stack(cols, 1)
        sgn = torch.where((v * R[:, i]).sum(1) > 0, 1.0, -1.0).to(R.dtype)
        N[:, i] = v * (sgn / torch.linalg.norm(v, dim=1))[:, None]
    return N


def _minor(M, rows, cols):
    return torch.linalg.det(M[..., rows, :][..., :, cols])


def _orth4(R):
    n = R.shape[0]; dev = R.device
    S, Tt, W, DU, DW = _consts(4, dev)
    out = torch.empty(n, dtype=torch.float64, device=dev)
    dt = torch.linalg.det(R).abs()
    thin = dt < B.THIN_DET
    if thin.any():
        Rt_ = R[thin]; q = ((DU[None] @ Rt_) ** 2).sum(2)             # (m, 1000)
        out[thin] = dt[thin] / (2 * math.pi ** 2) * (DW[None] / q ** 2).sum(1)
    w = ~thin
    if w.any():
        N = _cofactor_normals(R[w]); C = N @ N.transpose(1, 2)
        p = torch.full((N.shape[0],), 1 / 16, dtype=torch.float64, device=dev)
        I = torch.eye(4, dtype=torch.float64, device=dev)
        Rt = C[:, None] + S[None, :, None, None] * (I - C)[:, None]   # (m, K, 4, 4)
        dR = torch.clamp(torch.linalg.det(Rt), min=0.0)
        acc = torch.zeros_like(p)
        asum = torch.zeros_like(p)
        for i, j in itertools.combinations(range(4), 2):
            dm = torch.linalg.norm(N[:, i] - N[:, j], dim=1); dp = torch.linalg.norm(N[:, i] + N[:, j], dim=1)
            asum = asum + (math.pi / 2 - 2 * torch.atan2(dm, dp))
            oma = torch.minimum(dm, dp) ** 2 / 2
            k, l = [q for q in range(4) if q != i and q != j]
            ck, cl = [q for q in range(4) if q != k], [q for q in range(4) if q != l]
            ckl = (-1) ** (k + l) * _minor(Rt, ck, cl)
            r = C[:, i, j]; ra = torch.clamp(r.abs(), max=1.0)
            om = (oma[:, None] + ra[:, None] * S[None]) * (1 + ra[:, None] * Tt[None])
            apc = torch.atan2(-ckl, torch.sqrt(dR * om))
            acc = acc + (W[None] * r[:, None] * apc / torch.sqrt(torch.clamp(om, min=1e-300))).sum(1)
        out[w] = p + asum / (8 * math.pi) + acc / (4 * math.pi ** 2)
    ops = 144 * n + int(thin.sum()) * 1000 * 30 + int(w.sum()) * (320 + 120 + len(S) * (6 * 34 + 30))
    return out, ops


def _orth5(R):
    n = R.shape[0]; dev = R.device
    S, Tt, W, DU, DW = _consts(5, dev)
    out = torch.empty(n, dtype=torch.float64, device=dev)
    dt = torch.linalg.det(R).abs()
    thin = dt < B.THIN_DET5[0]
    if thin.any():
        Rt_ = R[thin]; q = ((DU[None] @ Rt_) ** 2).sum(2)
        out[thin] = 3 * dt[thin] / (8 * math.pi ** 2) * (DW[None] / q ** 2.5).sum(1)
    w = ~thin
    if w.any():
        N = _cofactor_normals(R[w]); C = N @ N.transpose(1, 2)
        m = N.shape[0]
        I = torch.eye(5, dtype=torch.float64, device=dev)
        Rt = C[:, None] + S[None, :, None, None] * (I - C)[:, None]   # (m, K, 5, 5)
        dQ, cQ = {}, {}
        for Q in itertools.combinations(range(5), 4):
            RQ = Rt[..., list(Q), :][..., :, list(Q)]
            dQ[Q] = torch.clamp(torch.linalg.det(RQ), min=0.0)
            for a_, b_ in itertools.combinations(range(4), 2):
                ra_, rb_ = [q for q in range(4) if q != a_], [q for q in range(4) if q != b_]
                cQ[Q, Q[a_], Q[b_]] = (-1) ** (a_ + b_) * _minor(RQ, ra_, rb_)
        asum = torch.zeros(m, dtype=torch.float64, device=dev); acc = torch.zeros_like(asum)
        for i, j in itertools.combinations(range(5), 2):
            dm = torch.linalg.norm(N[:, i] - N[:, j], dim=1); dp = torch.linalg.norm(N[:, i] + N[:, j], dim=1)
            asum = asum + (math.pi / 2 - 2 * torch.atan2(dm, dp))
            oma = torch.minimum(dm, dp) ** 2 / 2
            rest = [q for q in range(5) if q != i and q != j]
            r = C[:, i, j]; ra = torch.clamp(r.abs(), max=1.0)
            om = (oma[:, None] + ra[:, None] * S[None]) * (1 + ra[:, None] * Tt[None])
            sa = 0.0
            for k, l in itertools.combinations(rest, 2):
                Q = tuple(sorted((i, j, k, l)))
                sa = sa + torch.atan2(-cQ[Q, k, l], torch.sqrt(dQ[Q] * om))
            acc = acc + (W[None] * r[:, None] * sa / torch.sqrt(torch.clamp(om, min=1e-300))).sum(1)
        out[w] = 1 / 32 + asum / (16 * math.pi) + acc / (8 * math.pi ** 2)
    K = len(S)
    ops = 125 * n + int(thin.sum()) * len(DU) * 40 + int(w.sum()) * (1175 + 240 + K * (200 + 420) + K * 10 * 72)
    return out, ops


def _orth3(R):
    Bm = torch.linalg.inv(R.transpose(1, 2)); Sm = Bm @ Bm.transpose(1, 2)
    dg = torch.sqrt(torch.diagonal(Sm, dim1=1, dim2=2)); C = Sm / (dg[:, :, None] * dg[:, None, :])
    a = sum(torch.asin(torch.clamp(C[:, i, j], -1, 1)) for i, j in ((0, 1), (0, 2), (1, 2)))
    return 0.125 + a / (4 * math.pi), (27 + 27 + 9 + 6) * R.shape[0]


def masses(R, d):
    """Summed Gaussian mass of the simplicial cones R (n, d, d) (rows = rays) and the estT op charge."""
    f = {3: _orth3, 4: _orth4, 5: _orth5}[d]
    tot = 0.0; ops = 0
    for a in range(0, len(R), CHUNK.get(d, 8192)):
        Rt = torch.as_tensor(R[a:a + CHUNK.get(d, 8192)], dtype=torch.float64, device=DEV[0])
        m, o = f(Rt); tot += float(m.sum()); ops += o
    return tot, ops


def masses_each(R, d):
    """Per-simplex masses (for validation against estT.orthant)."""
    f = {3: _orth3, 4: _orth4, 5: _orth5}[d]
    return f(torch.as_tensor(R, dtype=torch.float64, device=DEV[0]))[0].cpu().numpy()


B.mass = _tri_mass


def run(model, mode, device="cpu", eps=0.1):
    DEV[0] = device
    for k in B.STATS:
        B.STATS[k] = 0
    BUF[0] = _Buf(); o = Ops(); t0 = time.time()
    acc0 = B.estT(model, o, eps=eps, mode=mode)
    tg = time.time() - t0
    BUF[0].flush()
    if device == "cuda":
        torch.cuda.synchronize()
    assert acc0 == 1.0
    r = dict(acc=1.0 - BUF[0].err, ops=o.n + BUF[0].ops, sec=time.time() - t0, sec_geometry=tg,
             simplices=BUF[0].simplices, **{k: v for k, v in B.STATS.items() if k != "simplices"})
    return r
