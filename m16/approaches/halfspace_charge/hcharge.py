"""T437: rigorous halfspace ("tilted") charge of a u-box, plus the BaB that uses it (README pre-registration).
Event inside box B lies in Q(B) = B ∩ {a_j.u + b_j <= 0, all j}; for lambda >= 0,
  1[Q] <= 1[B] exp(-(alpha.u + beta)),  alpha = lambda A, beta = lambda.b.
For fixed (u_r, g) the inner coordinates integrate in closed form (Phi differences shifted by alpha_i / g):
  f(r, g) = g (2 pi)^-1 exp(-g^2 q(r)/2 + E(r) + S/(2 g^2)) * D_a * prod_l D_l,
  q = r^2 + (r+1)^2, E = -alpha_r r - beta + alpha_a (r+1) - sum_l alpha_l r, S = alpha_a^2 + sum_l alpha_l^2,
  D_a = Phi(g(r+1+a1) + alpha_a/g) - Phi(g(r+1+a0) + alpha_a/g), D_l = Phi(g(r-l0) - alpha_l/g) - Phi(g(r-l1) - alpha_l/g).
The 2-D integral is bounded by per-rectangle monotone sup bounds (no quadrature error) + an analytic g-tail.
usage: python hcharge.py p1,p2,p3 OUT.json [CPU_SEC] [K] [--resume]"""
import json
import math
import os
import sys
import time

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("WEIGHTS", os.path.join(HERE, "..", "..", "..", "data", "t437_weights.npz"))
import t414_ibab as ib  # noqa: E402  (T414's coords, CROWN, split heuristic; copied unchanged)

dt = th.float64
T0, TAIL = ib.T0, ib.TAIL
NR, NG, WSPAN = int(os.environ.get("NR", 16)), int(os.environ.get("NG", 40)), 80.0
FBBT = os.environ.get("FBBT", "0") == "1"                            # deviation D2: box hull of Q(B)
SPLIT = os.environ.get("SPLIT", "margin")                             # "widest" = deviation D1 (T418's split)
REL = 1e-10                                                          # outward relative inflation (pre-registered)
SLACK = 1e-7
LOG2PI = math.log(2 * math.pi); SQ2 = math.sqrt(2)


def margins_lin(lo_u, hi_u, P, q, p3):
    """T414's CROWN, also returning the linear bounds: margin_j(u) >= A_j . u + b_j on the box (b lowered by SLACK)"""
    N = lo_u.shape[0]; lo, up = {}, {}; I = th.eye(ib.H, dtype=dt)
    for s in range(1, ib.T + 1):
        C = th.cat([I, -I])[None].expand(N, 2 * ib.H, ib.H)
        cy, c0 = ib.backward(C, s, lo, up); b, _ = ib.concretise(cy, c0, lo_u, hi_u, P, q)
        lo[s], up[s] = b[:, :ib.H], -b[:, ib.H:]
    others = [j for j in range(ib.D) if j != p3]
    Dm = (ib.Wo[others] - ib.Wo[p3])[None].expand(N, 9, ib.H)
    su, bu, sl = ib.relax(lo[ib.T], up[ib.T]); pos = Dm >= 0
    c0h = th.where(pos, th.zeros_like(Dm), Dm * bu[:, None]).sum(-1)
    cy, c0 = ib.backward(Dm * th.where(pos, sl[:, None], su[:, None]), ib.T, lo, up)
    lb, A = ib.concretise(cy, c0 + c0h, lo_u, hi_u, P, q)
    b = c0 + c0h + cy @ q
    umax = th.maximum(lo_u.abs(), hi_u.abs())
    b = b - SLACK * (b.abs() + (A.abs() * umax[:, None]).sum(-1) + 1)
    return lb, A, b


def logPhidiff(hi, lo):
    """log(Phi(hi) - Phi(lo)) for hi >= lo (elementwise), stable in both tails, with an absolute rounding allowance"""
    neg = (hi + lo) < 0                                              # use the lower tail when the interval sits left
    a = th.where(neg, -hi, lo); bb = th.where(neg, -lo, hi)            # Phi(hi)-Phi(lo) = Q(a) - Q(bb), a <= bb
    Qa = 0.5 * th.erfc(a / SQ2); Qb = 0.5 * th.erfc(bb / SQ2)
    d = (Qa - Qb).clamp_min(0) + 4e-16 * Qa                          # erfc relative accuracy (A1) + cancellation
    return th.log(d)


def _edges(lo, hi, m):
    """r cell edges (N, NR+1) and g cell edges (N, NG+1); g grid quadratic in w = g^2 m / 2 from w0 to w0 + WSPAN"""
    s = th.linspace(0, 1, NR + 1, dtype=dt)
    re = lo[:, 0:1] + (hi[:, 0:1] - lo[:, 0:1]) * s[None]
    w0 = m * T0 ** 2 / 2
    w = w0[:, None] + WSPAN * (th.linspace(0, 1, NG + 1, dtype=dt) ** 2)[None]
    ge = th.sqrt(2 * w / m[:, None]); ge[:, 0] = T0
    return re, ge


def log_rect_bound(lo, hi, alpha, beta, re, ge):
    """(N, NR, NG) upper bounds of log max f over each rectangle (tilted by alpha, beta)"""
    r0, r1 = re[:, :-1, None], re[:, 1:, None]; g0, g1 = ge[:, None, :-1], ge[:, None, 1:]
    qmin = th.where(r1 < -0.5, 2 * r1 ** 2 + 2 * r1 + 1, th.where(r0 > -0.5, 2 * r0 ** 2 + 2 * r0 + 1, th.full_like(r0, 0.5)))
    ar, aa, al = alpha[:, 0], alpha[:, 1], alpha[:, 2:]
    slope = (-ar + aa - al.sum(1))[:, None, None]; E = th.maximum(slope * r0, slope * r1) - beta[:, None, None] + aa[:, None, None]
    S = (alpha[:, 1:] ** 2).sum(1)[:, None, None]
    out = th.log(g1) - LOG2PI - g0 ** 2 * qmin / 2 + E + S / (2 * g0 ** 2)

    def bil(c):                                                      # sup / inf of g (r + c) over the rectangle
        v = th.stack([g0 * (r0 + c), g0 * (r1 + c), g1 * (r0 + c), g1 * (r1 + c)])
        return v.max(0).values, v.min(0).values

    def kk(k):                                                       # sup / inf of k / g over [g0, g1]
        return th.maximum(k / g0, k / g1), th.minimum(k / g0, k / g1)
    ka_s, ka_i = kk(aa[:, None, None])
    U, _ = bil(1 + hi[:, 1, None, None]); _, L = bil(1 + lo[:, 1, None, None])
    out = out + logPhidiff(U + ka_s, L + ka_i)
    for i in range(7):
        kl_s, kl_i = kk(al[:, i, None, None])
        U, _ = bil(-lo[:, 2 + i, None, None]); _, L = bil(-hi[:, 2 + i, None, None])
        out = out + logPhidiff(U - kl_i, L - kl_s)
    return out


def tail_bound(lo, hi, m, ge):
    """untilted mass with g > G: vol * 16 (2 pi)^-5 m^-5 Gamma(5, m G^2 / 2) (T414's radial bound)"""
    w = m * ge[:, -1] ** 2 / 2
    gam = 24 * th.exp(-w) * (1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24)
    return (hi - lo).prod(1) * 16 * (2 * math.pi) ** -5 * m ** -5 * gam


def _m(lo, hi):
    """T414's rigorous lower bound m of min |y|^2 over the box (its charge() internals)"""
    alo, ahi, llo, lhi = lo[:, 1], hi[:, 1], lo[:, 2:], hi[:, 2:]

    def F(u):
        ya = u + 1 + th.minimum(th.maximum(-(u + 1), alo), ahi); yl = u[:, None] - th.minimum(th.maximum(u[:, None], llo), lhi)
        return u ** 2 + (u + 1) ** 2 + ya ** 2 + (yl ** 2).sum(1), 2 * u + 2 * (u + 1) + 2 * ya + 2 * yl.sum(1)
    a_, b_ = lo[:, 0].clone(), hi[:, 0].clone()
    for _ in range(60):
        mid = (a_ + b_) / 2; _, d = F(mid); a_ = th.where(d < 0, mid, a_); b_ = th.where(d < 0, b_, mid)
    fl, dl = F(lo[:, 0]); fh, dh = F(hi[:, 0]); fa, da = F(a_)
    inner = fa + da.clamp(max=0) * (b_ - a_)
    return th.where(dl >= 0, fl, th.where(dh <= 0, fh, inner)).clamp_min(0.5)


def opt_lambda(lo, hi, A, b, m, steps=int(os.environ.get("LSTEPS", 30))):
    """lambda >= 0 minimising a coarse, non-rigorous midpoint quadrature of the log tilted mass (convex in lambda)"""
    N = lo.shape[0]
    sc = (A.abs() * (hi - lo)[:, None]).sum(-1) + b.abs() + 1e-12          # per-spec scale -> dimensionless lambda'
    s = th.linspace(0, 1, 7, dtype=dt); rm = (s[:-1] + s[1:]) / 2
    rr = (lo[:, 0:1] + (hi[:, 0:1] - lo[:, 0:1]) * rm[None])[:, :, None]
    w0 = m * T0 ** 2 / 2; wm = w0[:, None] + 12 * (((th.arange(10, dtype=dt) + .5) / 10) ** 2)[None]
    gg = th.sqrt(2 * wm / m[:, None])[:, None, :]
    lam = th.zeros(N, 9, dtype=dt, requires_grad=True); opt = th.optim.Adam([lam], lr=0.5)

    def logmass(lmb):
        al = lmb / sc; alpha = (al[:, :, None] * A).sum(1); beta = (al * b).sum(1)
        ar, aa, a7 = alpha[:, 0, None, None], alpha[:, 1, None, None], alpha[:, 2:]
        f = th.log(gg) - gg ** 2 * (rr ** 2 + (rr + 1) ** 2) / 2 - ar * rr - beta[:, None, None] + aa * (rr + 1) + (aa ** 2 + (a7 ** 2).sum(1)[:, None, None]) / (2 * gg ** 2)
        f = f + logPhidiff(gg * (rr + 1 + hi[:, 1, None, None]) + aa / gg, gg * (rr + 1 + lo[:, 1, None, None]) + aa / gg)
        for i in range(7):
            k = a7[:, i, None, None]
            f = f - k * rr + logPhidiff(gg * (rr - lo[:, 2 + i, None, None]) - k / gg, gg * (rr - hi[:, 2 + i, None, None]) - k / gg)
        return th.logsumexp(f.flatten(1), 1)
    for _ in range(steps):
        opt.zero_grad(); v = logmass(lam); v.sum().backward(); opt.step()
        with th.no_grad():
            lam.clamp_(min=0)
    al = th.nan_to_num(lam.detach() / sc, nan=0.0, posinf=0.0)       # any lambda >= 0 is valid; fall back to 0
    al = th.where(th.isfinite(al).all(1, keepdim=True), al, th.zeros_like(al))
    return (al[:, :, None] * A).sum(1), (al * b).sum(1)


def hcharge(lo, hi, A, b, chunk=256):
    """rigorous upper bounds (tilted, untilted) of the slab mass of Q(B) and of B, per box"""
    outs_t, outs_u = [], []
    for i in range(0, lo.shape[0], chunk):
        l, h, a, bb = lo[i:i + chunk], hi[i:i + chunk], A[i:i + chunk], b[i:i + chunk]
        m = _m(l, h); re, ge = _edges(l, h, m)
        alpha, beta = opt_lambda(l, h, a, bb, m)
        area = (re[:, 1:] - re[:, :-1])[:, :, None] * (ge[:, 1:] - ge[:, :-1])[:, None, :]
        lu = log_rect_bound(l, h, th.zeros_like(alpha), th.zeros_like(beta), re, ge)
        lt = log_rect_bound(l, h, alpha, beta, re, ge)
        tail = tail_bound(l, h, m, ge)
        u = (th.exp(lu) * area).sum((1, 2)) + tail
        lt = th.where(th.isnan(lt), lu, lt)                              # a NaN tilted cell falls back to the untilted bound
        t = (th.exp(th.minimum(lt, lu)) * area).sum((1, 2)) + tail
        outs_t.append(t * (1 + REL)); outs_u.append(u * (1 + REL))
    return th.cat(outs_t), th.cat(outs_u)


def fbbt(lo, hi, A, b, passes=3):
    """rigorous box hull of B ∩ {A_j.u + b_j <= 0 for all j}; returns (lo, hi, empty)"""
    lo, hi = lo.clone(), hi.clone()
    for _ in range(passes):
        for j in range(A.shape[1]):
            a = A[:, j]; mn = th.minimum(a * lo, a * hi); S = mn.sum(1, keepdim=True)
            bound = (-b[:, j, None] - (S - mn)) / th.where(a == 0, th.ones_like(a), a)
            bound = bound + 1e-9 * (1 + bound.abs()) * th.where(a > 0, 1.0, -1.0)
            hi = th.where(a > 1e-12, th.minimum(hi, bound), hi); lo = th.where(a < -1e-12, th.maximum(lo, bound), lo)
    return lo, hi, (lo > hi).any(1)


def splitdim(lb, A, lo, hi):
    return (hi - lo).argmax(1) if SPLIT == "widest" else ib.splitdim(lb, A, lo, hi)


def run(tri, sec, K, out, resume=False, log_every=10):
    P, q, lo0, hi0 = ib.coords(tri); p3 = tri[2]
    ck = out.replace(".json", "_ckpt.pt")
    if resume and os.path.exists(ck):
        st = th.load(ck); LO, HI, CH, CU, SD = st["LO"], st["HI"], st["CH"], st["CU"], st["SD"]
        evals, pruned, curve, it, used = st["evals"], st["pruned"], st["curve"], st["it"], st["cpu"]
    else:
        LO, HI = lo0[None].clone(), hi0[None].clone()
        lb0, A0, b0 = margins_lin(LO, HI, P, q, p3); CH, CU = hcharge(LO, HI, A0, b0); SD = splitdim(lb0, A0, LO, HI)
        evals = pruned = it = 0; curve = []; used = 0.0
    t0 = time.process_time() - used; last_ck = time.time()
    while len(LO) and time.process_time() - t0 < sec and len(LO) < 4e6:
        k = min(K, len(LO)); idx = th.topk(CH, k).indices; keep = th.ones(len(LO), dtype=th.bool); keep[idx] = False
        blo, bhi, d = LO[idx], HI[idx], SD[idx]; LO, HI, CH, CU, SD = LO[keep], HI[keep], CH[keep], CU[keep], SD[keep]
        ar = th.arange(k); mid = (blo[ar, d] + bhi[ar, d]) / 2
        alo, ahi = blo.clone(), bhi.clone(); ahi[ar, d] = mid
        blo2, bhi2 = blo.clone(), bhi.clone(); blo2[ar, d] = mid
        clo, chi = th.cat([alo, blo2]), th.cat([ahi, bhi2])
        lb2, A2, b2 = margins_lin(clo, chi, P, q, p3); evals += len(clo); it += 1
        pr = (lb2 > 0).any(1)
        if FBBT:
            clo, chi, emp = fbbt(clo, chi, A2, b2); pr = pr | emp; sv = ~pr
            lb2s, A2s, b2s = margins_lin(clo[sv], chi[sv], P, q, p3)     # CROWN again on the tightened boxes
            pr2 = th.zeros_like(pr); pr2[sv] = (lb2s > 0).any(1); pr = pr | pr2
            lb2[sv], A2[sv], b2[sv] = lb2s, A2s, b2s
        pruned += int(pr.sum()); sv = ~pr
        ct, cu = hcharge(clo[sv], chi[sv], A2[sv], b2[sv])
        LO, HI = th.cat([LO, clo[sv]]), th.cat([HI, chi[sv]]); CH = th.cat([CH, ct]); CU = th.cat([CU, cu])
        SD = th.cat([SD, splitdim(lb2[sv], A2[sv], clo[sv], chi[sv])])
        if it % log_every == 0 or not len(LO):
            el = time.process_time() - t0
            curve.append(dict(evals=evals, open=len(LO), pruned=pruned, charge=float(CH.sum()) + TAIL,
                              untilted=float(CU.sum()) + TAIL, cpu_s=el))
            print("it %d evals %d open %d pruned %d charge %.3e untilted %.3e (%.0f evals/s)" % (
                it, evals, len(LO), pruned, curve[-1]["charge"], curve[-1]["untilted"], evals / el), flush=True)
        if time.time() - last_ck > 600:
            th.save(dict(LO=LO, HI=HI, CH=CH, CU=CU, SD=SD, evals=evals, pruned=pruned, curve=curve, it=it,
                         cpu=time.process_time() - t0), ck); last_ck = time.time()
    res = dict(tri=tri, tail=TAIL, final_charge=float(CH.sum()) + TAIL, final_untilted=float(CU.sum()) + TAIL,
               evals=evals, open=len(LO), pruned=pruned, cpu_s=time.process_time() - t0, curve=curve)
    th.save(dict(LO=LO, HI=HI, CH=CH, CU=CU, SD=SD, evals=evals, pruned=pruned, curve=curve, it=it,
                 cpu=time.process_time() - t0), ck)
    return res


if __name__ == "__main__":
    th.set_num_threads(1)
    tri = tuple(int(v) for v in sys.argv[1].split(",")); out = sys.argv[2]
    sec = float(sys.argv[3]) if len(sys.argv) > 3 else 7200; K = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    res = run(tri, sec, K, out, resume="--resume" in sys.argv)
    json.dump(res, open(out, "w"), indent=1)
