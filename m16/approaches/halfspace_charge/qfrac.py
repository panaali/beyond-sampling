"""T437 D2 scale estimate (float, not a certificate): on NB uniform random open boxes of a checkpoint,
sum qmass(B) * (weighted-MC fraction of B's slab mass inside Q(B)) vs sum qmass(B).
usage: python qfrac.py CKPT.pt [NB] [NS]"""
import sys

import torch as th

import hcharge as hc
import t414_ibab as ib
import t414_qcharge as qc

th.set_num_threads(1)
st = th.load(sys.argv[1]); nb = int(sys.argv[2]) if len(sys.argv) > 2 else 5000; ns = int(sys.argv[3]) if len(sys.argv) > 3 else 500
P, q, _, _ = ib.coords((3, 7, 8)); g = th.Generator().manual_seed(4371)
idx = th.randperm(len(st["LO"]), generator=g)[:nb]; LO, HI = st["LO"][idx], st["HI"][idx]
qm, fr = [], []
for i in range(0, nb, 250):
    lo, hi = LO[i:i + 250], HI[i:i + 250]; _, A, b = hc.margins_lin(lo, hi, P, q, 8)
    qm.append(qc.qmass(lo, hi, 2.5, hc._m(lo, hi)))
    u = lo[:, None] + (hi - lo)[:, None] * th.rand(len(lo), ns, 9, generator=g, dtype=th.float64)
    y = u @ P.T + q; m = (y ** 2).sum(-1); w = m * 2.5 ** 2 / 2
    wt = m ** -5 * th.exp(-w) * (1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24)
    inq = ((u @ A.transpose(1, 2) + b[:, None]) <= 0).all(-1)
    fr.append((wt * inq).sum(1) / wt.sum(1))
qm, fr = th.cat(qm), th.nan_to_num(th.cat(fr)); scale = len(st["LO"]) / nb
print("open boxes %d, sampled %d: est. sum qmass(B) %.3e, est. sum qmass(B) frac_Q %.3e, ratio %.4f, boxes with frac_Q = 0: %.3f" % (
    len(st["LO"]), nb, scale * float(qm.sum()), scale * float((qm * fr).sum()), float((qm * fr).sum() / qm.sum()), float((fr == 0).double().mean())))
