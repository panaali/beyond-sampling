"""T192/T221: Gaussian masses of many simplicial cones summed per group (e.g. per cell or per activation region).

Wide cones (|det| >= thin threshold): estT formulas per simplex (estG._orth3/_orth4/_orth5, orth6). Thin cones: sliver
bisection (estG.sliver_split rule, pairwise cos >= thr) with the parent group carried along, then estT's ray-space thin
rule per piece. d = 6 uses the orth6 split: |det| < 1e-2 -> sliver + 6^5 rule, else nested Plackett."""
import math

import numpy as np
import torch

import estG
import estT_base as B
import orth6

THIN = {3: 0.0, 4: B.THIN_DET, 5: B.THIN_DET5[0], 6: orth6.WIDE6[0]}


def sliver_split_ids(R, ids, thr, maxlev=24):
    outR, outI = [], []
    for _ in range(maxlev):
        n, d, _ = R.shape
        G = R @ R.transpose(1, 2); flat = G.reshape(n, -1).clone(); flat[:, ::d + 1] = 2.0
        mn, arg = flat.min(1); bad = mn < thr
        outR.append(R[~bad]); outI.append(ids[~bad])
        if not bad.any():
            return torch.cat(outR), torch.cat(outI)
        Rb = R[bad]; ib = ids[bad]; ab = arg[bad]; a = ab // d; b = ab % d; k = torch.arange(len(Rb), device=R.device)
        m = Rb[k, a] + Rb[k, b]; m = m / torch.linalg.norm(m, dim=1, keepdim=True)
        c1 = Rb.clone(); c1[k, a] = m; c2 = Rb.clone(); c2[k, b] = m
        R = torch.cat([c1, c2]); ids = torch.cat([ib, ib])
    return torch.cat(outR + [R]), torch.cat(outI + [ids])


def _thin_rule(R, d):
    dt = torch.linalg.det(R).abs()
    if d == 6:
        DU, DW = orth6._duffy5(orth6.NTHIN[0], R.device)
        return orth6._ray(R, dt, DU, DW)
    DU, DW = (B._DU, B._DW) if d == 4 else B._DU5[0]
    DU = torch.as_tensor(DU, device=R.device); DW = torch.as_tensor(DW, device=R.device)
    q = ((DU[None] @ R) ** 2).sum(2)
    c = 1 / (2 * math.pi ** 2) if d == 4 else 3 / (8 * math.pi ** 2)
    return c * dt * (DW[None] / q ** (d / 2)).sum(1)


def _wide(R, d):
    if d == 3:
        return estG._orth3(R)[0]
    if d == 6:
        return orth6._wide(orth6._normals(R))
    return (estG._orth4 if d == 4 else estG._orth5)(R)[0]


def grouped(R, gid, ngroups, device="cpu", thr=0.6, chunk=4096, tb=8192):
    """R (n, d, d) numpy unit rays (rows), gid (n,) int -> (ngroups,) float64 summed masses."""
    d = R.shape[1]; out = torch.zeros(ngroups, dtype=torch.float64, device=device)
    old = estG.SLIVER[0]; estG.SLIVER[0] = None
    try:
        for a in range(0, len(R), chunk):
            r = torch.as_tensor(R[a:a + chunk], dtype=torch.float64, device=device)
            g = torch.as_tensor(gid[a:a + chunk], dtype=torch.int64, device=device)
            thin = torch.linalg.det(r).abs() < THIN[d] if d > 3 else torch.zeros(len(r), dtype=torch.bool, device=device)
            if (~thin).any():
                out.index_add_(0, g[~thin], _wide(r[~thin], d))
            if thin.any():
                P, I = sliver_split_ids(r[thin], g[thin], thr)
                for b in range(0, len(P), tb):
                    out.index_add_(0, I[b:b + tb], _thin_rule(P[b:b + tb], d))
    finally:
        estG.SLIVER[0] = old
    return out.cpu().numpy()
