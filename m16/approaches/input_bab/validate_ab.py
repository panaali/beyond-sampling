"""T414 validation (a) charge >= MC slab mass (2e6 samples) on 20 random boxes (formula checked at t0 = 0.3, where the slab is common;
the formula is the same for every t0) and root charge at t0 = 2.5 >= 5.6e-8; (c) CROWN lower bounds <= sampled margins
on random boxes. usage: python validate_ab.py p1,p2,p3"""
import sys

import numpy as np
import torch as th

import ibab

th.set_num_threads(1)
tri = tuple(int(v) for v in sys.argv[1].split(",")); p1, p2, p3 = tri; P, q, lo0, hi0 = ibab.coords(tri)
rng = np.random.default_rng(4140)
print("root charge t0=2.5: %.3e (must be >= 5.6e-8)" % float(ibab.charge(lo0[None], hi0[None], P, q, (p2, p3))[0]))
# (a1) the radial formula (Jacobian g^9, Gamma(5)) vs direct x-sampling of the slab; (a2) charge >= exact box mass
def fmass(s, t0):                                                        # integral_{t0}^inf g^9 phi_10(g y) dg, |y|^2 = s
    w = s * t0 ** 2 / 2
    return 16 * (2 * np.pi) ** -5 * s ** -5 * 24 * np.exp(-w) * (1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24)
def umass(lo, hi, t0, n=400000):                                         # vol x E_u f(|y(u)|^2), uniform u in the box
    uu = lo + (hi - lo) * th.as_tensor(rng.uniform(0, 1, (n, 9))); s = ((uu @ P.T + q) ** 2).sum(1).numpy()
    v = float((hi - lo).prod()); f = fmass(s, t0); return v * f.mean(), v * f.std() / n ** .5
lows = [i for i in range(10) if i not in tri]; ok = True
for t0 in (0.3, 1.0):
    hits = 0; n = 0
    for _ in range(5):
        x = rng.standard_normal((1000000, 10)); g = x[:, p2] - x[:, p3]
        hits += int(((g >= t0) & (x[:, p1] >= x[:, p2]) & (x[:, lows].max(1) <= x[:, p3]) & (np.abs(x[:, p3]) <= ibab.U * g)
                     & (x[:, p1] - x[:, p2] <= ibab.U * g) & ((x[:, p3:p3 + 1] - x[:, lows]).max(1) <= ibab.U * g)).sum()); n += 1000000
    mc = hits / n; se = (mc / n) ** .5; um, us_ = umass(lo0, hi0, t0, 2000000)
    good = abs(um - mc) <= 3 * (se ** 2 + us_ ** 2) ** .5; ok &= good
    print("(a1) t0 %.1f slab-in-root-box mass: x-MC %.4e +- %.1e, u-integral %.4e +- %.1e %s" % (t0, mc, se, um, us_, "ok" if good else "MISMATCH"))
um, us_ = umass(lo0, hi0, 2.5, 2000000); print("(a1) t0 2.5 slab mass by u-integral %.3e +- %.1e (DRI: 5.6e-8)" % (um, us_))
for b in range(20):
    c = th.as_tensor(rng.uniform([-1.5] + [0] * 8, [0.5] + [1.5] * 8)); w = th.as_tensor(rng.uniform(0.02, 0.5, 9))
    lo, hi = (c - w).clone(), c + w; lo[1:] = lo[1:].clamp(min=0)
    m_, s_ = umass(lo, hi, 2.5, 100000); ch = float(ibab.charge(lo[None], hi[None], P, q, (p2, p3))[0])
    good = ch >= m_ - 3 * s_; ok &= good
    if b < 6 or not good:
        print("(a2) box %d exact mass %.3e charge %.3e ratio %.2f %s" % (b, m_, ch, ch / m_, "" if good else "FAIL"))
print("(a) PASS" if ok else "(a) FAIL")
# (c) CROWN soundness on random boxes (any t0; the margin does not depend on g)
ok = True
for b in range(30):
    c = th.as_tensor(rng.uniform([-2] + [0] * 8, [2] + [2] * 8)); w = th.as_tensor(rng.uniform(0.01, 1.0, 9))
    lo, hi = (c - w).clone(), c + w; lo[1:] = lo[1:].clamp(min=0)
    lb, _ = ibab.margins(lo[None], hi[None], P, q, p3)
    uu = lo + (hi - lo) * th.as_tensor(rng.uniform(0, 1, (20000, 9)))
    m = ibab.forward_margin(uu @ P.T + q, p3); m = th.cat([m[:, :p3], m[:, p3 + 1:]], 1)
    viol = (lb[0][None] > m.min(0).values[None] + 1e-9).any(); ok &= not bool(viol)
    if b < 5 or viol:
        print("box %d lb max %.3f sampled min of best spec %.3f %s" % (b, float(lb.max()), float(m.min(0).values[lb[0].argmax()]), "VIOLATION" if viol else ""))
print("(c) CROWN lb <= sampled margins on 30 boxes:", "PASS" if ok else "FAIL")
