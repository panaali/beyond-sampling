"""T414: input-space branch and bound for E1(t0) within one triple (README pre-registration).
u-boxes in the homogeneity-reduced coordinates y = P u + q (x = g y, g = x_p2 - x_p3 >= t0); CROWN lower bounds of
L_j - L_p3 over the box prune it; surviving boxes are charged the radial upper bound of their slab mass.
usage: python ibab.py p1,p2,p3 OUT.json [CPU_SEC] [K]"""
import json
import math
import os
import sys
import time

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
if os.environ.get("WEIGHTS"):
    _w = np.load(os.environ["WEIGHTS"]); Wi0, Wh0, Wo0 = _w["Wi"], _w["Wh"], _w["Wo"]
else:
    from common import Wi as Wi0, Wh as Wh0, Wo as Wo0  # noqa: E402
D, H, T = 10, 16, 10
T0, U = 2.5, 4.0
TAIL = 2 * 0.5 * math.erfc(U * T0 / math.sqrt(2)) + 8 * 0.5 * math.erfc(U * T0 / 2)   # Phi(-U t0) and Phi(-U t0 / sqrt 2)
dt = th.float64
Wi, Wh, Wo = (th.as_tensor(np.asarray(w, float), dtype=dt) for w in (Wi0, Wh0, Wo0))


def coords(tri):
    """y = P u + q; u = (u_r, u_a, u_l[7]); root box lo/hi"""
    p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]
    P = th.zeros(D, 9, dtype=dt); q = th.zeros(D, dtype=dt)
    P[:, 0] = 1; q[p2] = 1; q[p1] = 1; P[p1, 1] = 1
    for i, l in enumerate(lows):
        P[l, 2 + i] = -1
    lo = th.zeros(9, dtype=dt); hi = th.full((9,), U, dtype=dt); lo[0] = -U
    return P, q, lo, hi


def charge(lo, hi, P, q, pair):
    """radial upper bound on the slab mass of u-boxes (N, 9): vol * 16 (2 pi)^-5 m^-5 Gamma(5, m t0^2 / 2) with m a
    rigorous lower bound of min |y(u)|^2 over the box. For fixed u_r every other coordinate is independent:
    F(u_r) = u_r^2 + (u_r + 1)^2 + (u_r + 1 + clamp(-(u_r + 1), a))^2 + sum_l (u_r - clamp(u_r, l))^2 is convex;
    endpoint minima are exact (F monotone); otherwise bisection brackets u* and the tangent at the bracket's left end bounds min F."""
    alo, ahi, llo, lhi = lo[:, 1], hi[:, 1], lo[:, 2:], hi[:, 2:]

    def F(u):
        ya = u + 1 + th.minimum(th.maximum(-(u + 1), alo), ahi); yl = u[:, None] - th.minimum(th.maximum(u[:, None], llo), lhi)
        return u ** 2 + (u + 1) ** 2 + ya ** 2 + (yl ** 2).sum(1), 2 * u + 2 * (u + 1) + 2 * ya + 2 * yl.sum(1)
    a_, b_ = lo[:, 0].clone(), hi[:, 0].clone()
    for _ in range(60):                                              # bracket the minimiser: F'(a_) < 0 <= F'(b_)
        mid = (a_ + b_) / 2; _, d = F(mid); a_ = th.where(d < 0, mid, a_); b_ = th.where(d < 0, b_, mid)
    fl, dl = F(lo[:, 0]); fh, dh = F(hi[:, 0]); fa, da = F(a_)
    inner = fa + da.clamp(max=0) * (b_ - a_)                         # tangent at a_ lower-bounds F on [a_, b_]
    m = th.where(dl >= 0, fl, th.where(dh <= 0, fh, inner)).clamp_min(0.5)   # monotone cases exact; |y|^2 >= 1/2 always
    w = m * T0 ** 2 / 2
    gam = 24 * th.exp(-w) * (1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24)
    return (hi - lo).prod(1) * 16 * (2 * math.pi) ** -5 * m ** -5 * gam


def concretise(cy, c0, lo, hi, P, q):
    """lower bound of cy . y + c0 over y = P u + q, u in [lo, hi]; also returns the u-coefficients"""
    cu = cy @ P; c = (lo + hi) / 2; r = (hi - lo) / 2
    return c0 + cy @ q + (cu * c[:, None]).sum(-1) - (cu.abs() * r[:, None]).sum(-1), cu


def relax(l, u):
    act = l >= 0; unst = (l < 0) & (u > 0)
    su = th.where(unst, u / (u - l).clamp_min(1e-300), act.to(dt)); bu = th.where(unst, -u * l / (u - l).clamp_min(1e-300), th.zeros_like(u))
    sl = th.where(unst, (u > -l).to(dt), act.to(dt))
    return su, bu, sl


def backward(C, s_top, lo, up):
    """C (N, Q, H) on z_{s_top} (1-indexed step); returns coefficients on y (N, Q, D) and constant (N, Q) such that
    C . z >= cy . y + c0 for all inputs consistent with the bounds"""
    N, Q, _ = C.shape; cy = th.zeros(N, Q, D, dtype=dt); c0 = th.zeros(N, Q, dtype=dt)
    for s in range(s_top, 0, -1):
        cy[:, :, s - 1] = C @ Wi
        if s == 1:
            break
        Ch = C @ Wh; su, bu, sl = relax(lo[s - 1], up[s - 1]); pos = Ch >= 0
        c0 = c0 + th.where(pos, th.zeros_like(Ch), Ch * bu[:, None]).sum(-1)
        C = Ch * th.where(pos, sl[:, None], su[:, None])
    return cy, c0


def margins(lo_u, hi_u, P, q, p3):
    """CROWN lower bounds of L_j - L_p3 (j != p3) over the boxes: (N, 9) values and (N, 9, 9) u-coefficients"""
    N = lo_u.shape[0]; lo, up = {}, {}; I = th.eye(H, dtype=dt)
    for s in range(1, T + 1):
        C = th.cat([I, -I])[None].expand(N, 2 * H, H)
        cy, c0 = backward(C, s, lo, up); b, _ = concretise(cy, c0, lo_u, hi_u, P, q)
        lo[s], up[s] = b[:, :H], -b[:, H:]
    others = [j for j in range(D) if j != p3]
    Dm = (Wo[others] - Wo[p3])[None].expand(N, 9, H)
    su, bu, sl = relax(lo[T], up[T]); pos = Dm >= 0
    c0h = th.where(pos, th.zeros_like(Dm), Dm * bu[:, None]).sum(-1)
    cy, c0 = backward(Dm * th.where(pos, sl[:, None], su[:, None]), T, lo, up)
    return concretise(cy, c0 + c0h, lo_u, hi_u, P, q)


def forward_margin(y, p3):
    h = th.zeros(y.shape[0], H, dtype=dt)
    for t in range(T):
        h = th.relu(h @ Wh.T + y[:, t:t + 1] * Wi[None])
    L = h @ Wo.T
    return L - L[:, p3:p3 + 1]                                     # event iff all <= 0 (j != p3)


def splitdim(lb, cu, lo, hi):
    """pre-registered heuristic: the spec with the largest lower bound; the dim maximising |c_d| * width_d"""
    c = cu[th.arange(len(lb)), lb.argmax(1)]
    return (c.abs() * (hi - lo)).argmax(1)


def run(tri, sec, K, log_every=20):
    P, q, lo0, hi0 = coords(tri); p3 = tri[2]
    pair = (tri[1], tri[2]); LO, HI = lo0[None].clone(), hi0[None].clone(); CH = charge(LO, HI, P, q, pair)
    lb0, cu0 = margins(LO, HI, P, q, p3); SD = splitdim(lb0, cu0, LO, HI)              # split dim stored per open box
    root = float(CH[0]); evals = 0; pruned = 0; curve = []; t0 = time.process_time(); it = 0
    res_lo, res_hi = [], []; rng = np.random.default_rng(414)                  # reservoir of pruned boxes
    while len(LO) and time.process_time() - t0 < sec and len(LO) < 4e6:
        k = min(K, len(LO)); idx = th.topk(CH, k).indices; keep = th.ones(len(LO), dtype=th.bool); keep[idx] = False
        blo, bhi, d = LO[idx], HI[idx], SD[idx]; LO, HI, CH, SD = LO[keep], HI[keep], CH[keep], SD[keep]
        mid = (blo[th.arange(k), d] + bhi[th.arange(k), d]) / 2
        alo, ahi = blo.clone(), bhi.clone(); ahi[th.arange(k), d] = mid
        blo2, bhi2 = blo.clone(), bhi.clone(); blo2[th.arange(k), d] = mid
        clo, chi = th.cat([alo, blo2]), th.cat([ahi, bhi2])
        lb2, cu2 = margins(clo, chi, P, q, p3); evals += len(clo); it += 1
        pr = (lb2 > 0).any(1); pruned += int(pr.sum())
        if pr.any() and len(res_lo) < 200:
            sel = th.nonzero(pr)[:, 0]; sel = sel[th.as_tensor(rng.permutation(len(sel))[:50])]
            res_lo.append(clo[sel]); res_hi.append(chi[sel])
        sv = ~pr; LO, HI = th.cat([LO, clo[sv]]), th.cat([HI, chi[sv]]); CH = th.cat([CH, charge(clo[sv], chi[sv], P, q, pair)])
        SD = th.cat([SD, splitdim(lb2[sv], cu2[sv], clo[sv], chi[sv])])
        if it % log_every == 0 or not len(LO):
            tot = float(CH.sum()) + TAIL; el = time.process_time() - t0
            curve.append(dict(evals=evals, open=len(LO), pruned=pruned, charge=tot, cpu_s=el))
            print("it %d evals %d open %d pruned %d charge %.3e (%.0f evals/s)" % (it, evals, len(LO), pruned, tot, evals / el), flush=True)
    tot = float(CH.sum()) + TAIL
    return dict(tri=tri, root_charge=root, tail=TAIL, final_charge=tot, evals=evals, open=len(LO), pruned=pruned,
                cpu_s=time.process_time() - t0, curve=curve), (th.cat(res_lo) if res_lo else None, th.cat(res_hi) if res_hi else None), (LO, HI, CH)


if __name__ == "__main__":
    th.set_num_threads(1)
    tri = tuple(int(v) for v in sys.argv[1].split(",")); out = sys.argv[2]
    sec = float(sys.argv[3]) if len(sys.argv) > 3 else 7200; K = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    res, (rlo, rhi), (LO, HI, CH) = run(tri, sec, K)
    json.dump(res, open(out, "w"), indent=1)
    th.save(dict(pruned_lo=rlo, pruned_hi=rhi, open_lo=LO, open_hi=HI, open_ch=CH), out.replace(".json", "_boxes.pt"))
