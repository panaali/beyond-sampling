"""T437 validation (a), (b): rigorous charges vs T414's D1 near-exact box mass and vs a u-MC of the slab mass of Q(B).
usage: python validate.py CKPT.pt [NBOX] [NS]"""
import math
import sys

import torch as th

import hcharge as hc
import t414_ibab as ib
import t414_qcharge as qc

th.set_num_threads(1)
P, q, lo0, hi0 = ib.coords((3, 7, 8))
R, Ru = (lo0[None], hi0[None])
lb, A, b = hc.margins_lin(R, Ru, P, q, 8)
for nr, ng in [(16, 40), (64, 160)]:
    hc.NR, hc.NG = nr, ng; t, u = hc.hcharge(R, Ru, A, b)
    print("(a) root box NR %d NG %d: rigorous untilted %.4e tilted %.4e, slab mass 5.5975e-08, ratio %.1f" % (nr, ng, u, t, float(u) / 5.5975e-8))
hc.NR, hc.NG = 16, 40
st = th.load(sys.argv[1]); nb = int(sys.argv[2]) if len(sys.argv) > 2 else 50; ns = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
g = th.Generator().manual_seed(4370); idx = th.randperm(len(st["LO"]), generator=g)[:nb]
LO, HI = st["LO"][idx], st["HI"][idx]
lb, A, b = hc.margins_lin(LO, HI, P, q, 8); t, u = hc.hcharge(LO, HI, A, b)
Q = qc.qmass(LO, HI, 2.5, hc._m(LO, HI))
bad_a = int((u < Q * (1 - 1e-3)).sum()); bad_b = 0; ratios = []
for i in range(nb):
    uu = LO[i] + (HI[i] - LO[i]) * th.rand(ns, 9, generator=g, dtype=th.float64)
    y = uu @ P.T + q; m = (y ** 2).sum(1); w = m * 2.5 ** 2 / 2
    f = 16 * (2 * math.pi) ** -5 * m ** -5 * 24 * th.exp(-w) * (1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24)
    f = f * ((uu @ A[i].T + b[i]) <= 0).all(1)
    vol = float((HI[i] - LO[i]).prod()); est = vol * float(f.mean()); se = vol * float(f.std()) / math.sqrt(ns)
    bad_b += int(float(t[i]) < est - 3 * se); ratios.append(float(t[i]) / max(est, 1e-300))
r = th.tensor(ratios, dtype=th.float64); qd = th.tensor([0, .5, 1.], dtype=th.float64)
print("(a) %d random open boxes: rigorous untilted < D1 near-exact mass on %d; ratio untilted/D1 quantiles %s" % (
    nb, bad_a, [round(float(v), 3) for v in th.quantile(u / Q, qd)]))
print("(b) tilted charge < MC(Q(B)) - 3 s.e. on %d of %d; ratio charge/MC quantiles %s" % (bad_b, nb, [round(float(v), 3) for v in th.quantile(r, qd)]))
