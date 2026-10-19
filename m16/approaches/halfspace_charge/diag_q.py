"""T437 diagnostic: weighted-MC fraction of each open box's slab mass inside Q(B) (all CROWN halfspaces <= 0),
and inside the best single halfspace; u uniform in the box, weight = int_{g>=t0} g^9 phi_10(g y(u)) dg (exact radial).
usage: python diag_q.py CKPT.pt [NBOX] [NS]"""
import math
import sys

import torch as th

import hcharge as hc
import t414_ibab as ib

th.set_num_threads(1)
st = th.load(sys.argv[1]); nb = int(sys.argv[2]) if len(sys.argv) > 2 else 200; ns = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
P, q, _, _ = ib.coords((3, 7, 8))
idx = th.topk(st["CU"], nb).indices; LO, HI = st["LO"][idx], st["HI"][idx]
lb, A, b = hc.margins_lin(LO, HI, P, q, 8)
g = th.Generator().manual_seed(437)
fr_all, fr_one, ev = [], [], 0
for i in range(nb):
    u = LO[i] + (HI[i] - LO[i]) * th.rand(ns, 9, generator=g, dtype=th.float64)
    y = u @ P.T + q; m = (y ** 2).sum(1); wv = m * 2.5 ** 2 / 2
    w = m ** -5 * th.exp(-wv) * (1 + wv + wv ** 2 / 2 + wv ** 3 / 6 + wv ** 4 / 24)
    lin = u @ A[i].T + b[i]                                       # (ns, 9)
    fr_all.append(float((w * (lin <= 0).all(1)).sum() / w.sum()))
    fr_one.append(float(min((w * (lin[:, j] <= 0)).sum() / w.sum() for j in range(9))))
    mg = ib.forward_margin(y, 8); mg = th.cat([mg[:, :8], mg[:, 9:]], 1)
    ev += int((mg <= 0).all(1).sum())
    assert (lin <= mg + 1e-6 * (1 + mg.abs())).all(), "CROWN halfspace above a sampled margin"
fa, fo = th.tensor(fr_all), th.tensor(fr_one); qs = th.tensor([0, .1, .5, .9, 1.])
print("boxes", nb, "samples/box", ns, "sampled events", ev)
print("fraction in Q(B), quantiles", [round(float(v), 4) for v in th.quantile(fa, qs)], "mass-weighted", float((fa * st["CU"][idx]).sum() / st["CU"][idx].sum()))
print("best single halfspace, quantiles", [round(float(v), 4) for v in th.quantile(fo, qs)])
c = (LO + HI) / 2; r = (HI - LO) / 2
ub = (A * c[:, None]).sum(-1) + b + (A.abs() * r[:, None]).sum(-1)       # max of each linear bound over the box
qd = qs.double()
print("max over specs of the CROWN lower bound's MIN over the box, quantiles", [round(float(v), 2) for v in th.quantile(lb.max(1).values, qd)])
print("max over specs of the CROWN lower bound's MAX over the box, quantiles", [round(float(v), 2) for v in th.quantile(ub.max(1).values, qd)])
print("boxes where some halfspace cuts the box (max > 0):", int((ub.max(1).values > 0).sum()))
