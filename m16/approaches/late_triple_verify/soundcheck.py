"""T418: soundness of certified boxes. Rebuild D2's certified tree boxes (or a lbab run's pruned reservoir), recompute
the library lower bounds, sample uniform points in each box with the reference forward on raw x (net.margins_x, not
UNet), and check (i) no sampled point is an event, (ii) sampled margin_j >= lb_j for every j (all specs).
usage: python soundcheck.py d2 D2.json METHOD [pts]   |   python soundcheck.py run RUN_open.pt METHOD [pts]"""
import json
import sys

import numpy as np
import torch as th

import net
import lbab

th.set_num_threads(1)
tri = (3, 7, 8); kind, path, meth = sys.argv[1:4]; pts = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
rng = np.random.default_rng(4183); A, c = net.ymap(tri)
if kind == "d2":
    R = json.load(open(path)); n = R["n"]
    x = net.slab_sample(tri, n, np.random.default_rng(4182)); u, _ = net.x_to_u(x, tri); u = th.as_tensor(u)
    lo0, hi0 = net.root(tri); depth = th.as_tensor(R["depth"]); lo = lo0[None].repeat(n, 1); hi = hi0[None].repeat(n, 1); ar = th.arange(n)
    for k in range(int(depth.max())):
        act = depth > k; w = hi - lo; d = th.argmax(w + 1e-12 * th.arange(9, 0, -1, dtype=net.dt), 1)
        mid = (lo[ar, d] + hi[ar, d]) / 2; right = u[ar, d] >= mid
        sr = act & right; sl = act & ~right
        lo[ar[sr], d[sr]] = mid[sr]; hi[ar[sl], d[sl]] = mid[sl]
    assert ((u >= lo) & (u <= hi)).all()
else:
    S = th.load(path); lo, hi = S["pruned_lo"], S["pruned_hi"]
B = lbab.Bounder(tri); lb = th.cat([B(lo[i:i + 64], hi[i:i + 64], meth)[0] for i in range(0, len(lo), 64)])
cert = (lb > 0).any(1); viol = 0; evs = 0; worst = np.inf
for i in range(len(lo)):
    uu = lo[i] + (hi[i] - lo[i]) * th.as_tensor(rng.uniform(size=(pts, 9))); y = uu @ A.T + c
    g = 2.5 + rng.exponential(1.0, size=(pts, 1))                                   # any g > 0 (homogeneity)
    m = net.margins_x(y * th.as_tensor(g), 8) / th.as_tensor(g)
    evs += int((m <= 0).all(1).sum()); gap = (m.min(0).values - lb[i]).min().item(); worst = min(worst, gap); viol += gap < -1e-7
print(dict(kind=kind, method=meth, boxes=len(lo), certified=int(cert.sum()), points=pts * len(lo), events=evs,
           bound_violations=int(viol), worst_gap=worst, PASS=bool(evs == 0 and viol == 0)), flush=True)
