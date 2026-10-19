"""T418 deviation D2: dyadic-leaf covering estimate. For exact slab points, walk down the bisection tree of the root
u-box with a fixed rule (split the widest dim, ties by lowest index, at the midpoint), and find the coarsest tree box
containing the point that the bounder certifies (binary search on depth; certification is checked at each probe). The
leaves a BaB with this split rule needs are >= those boxes, so N_leaf ~ E_slab[1 / (vol(leaf) p_u)] is the box count of
an ideal-order BaB (every split along the path, no wasted splits elsewhere). Tree nodes <= 2 x leaves.
usage: python dyadic.py p1,p2,p3 OUT.json [n] [method]"""
import json
import math
import sys
import time

import numpy as np
import torch as th

import net
import lbab

th.set_num_threads(1)
tri = tuple(int(v) for v in sys.argv[1].split(",")); out = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200; meth = sys.argv[4] if len(sys.argv) > 4 else "alpha-CROWN"
x = net.slab_sample(tri, n, np.random.default_rng(4182)); u, g = net.x_to_u(x, tri); u = th.as_tensor(u)
M = net.UNet(tri)
with th.no_grad():
    ev = (M(u) <= 0).all(1)
lo0, hi0 = net.root(tri); inside = ((u >= lo0) & (u <= hi0)).all(1)
B = lbab.Bounder(tri)


def box_at(depth):
    """tree box at a given depth containing each point (N, 9)"""
    lo = lo0[None].repeat(n, 1); hi = hi0[None].repeat(n, 1); ar = th.arange(n)
    for _ in range(depth):
        w = hi - lo; d = th.argmax(w + 1e-12 * th.arange(9, 0, -1, dtype=net.dt), 1)       # widest, lowest index on ties
        mid = (lo[ar, d] + hi[ar, d]) / 2; right = u[ar, d] >= mid
        lo[ar[right], d[right]] = mid[right]; hi[ar[~right], d[~right]] = mid[~right]
    return lo, hi


def cert(depth_vec):
    ok = th.zeros(n, dtype=th.bool)
    for dpt in depth_vec.unique().tolist():
        sel = th.nonzero(depth_vec == dpt)[:, 0]; lo, hi = box_at(int(dpt)); lo, hi = lo[sel], hi[sel]
        for i in range(0, len(sel), 64):
            ok[sel[i:i + 64]] = (B(lo[i:i + 64], hi[i:i + 64], meth)[0] > 0).any(1)
    return ok


t = time.process_time()
a = th.zeros(n, dtype=th.long); b = th.full((n,), 90, dtype=th.long)          # find the smallest certified depth in (a, b]
okb = cert(b)
while (b - a > 1).any():
    m = (a + b) // 2; c = cert(m); b = th.where(c & (b - a > 1), m, b); a = th.where(~c & (b - a > 1), m, a)
depth = th.where(okb, b, th.full_like(b, -1)).numpy()
vol0 = float((hi0 - lo0).prod()); vol = vol0 * 2.0 ** -depth.astype(float)
s2 = (x / g[:, None]) ** 2; s2 = s2.sum(1); ww = s2 * net.T0 ** 2 / 2
f = 16 * (2 * math.pi) ** -5 * s2 ** -5 * 24 * np.exp(-ww) * (1 + ww + ww ** 2 / 2 + ww ** 3 / 6 + ww ** 4 / 24)
Z = float(net.quad_mass(lo0[None], hi0[None], nr=200, nq=16, ng=64)); pu = f / Z
good = (~ev).numpy() & inside.numpy() & (depth >= 0); keep = good & (pu >= np.quantile(pu[good], .002))
est = float(np.mean(np.where(keep, 1 / (np.where(depth >= 0, vol, 1) * pu), 0)))
R = dict(n=n, method=meth, events=int(ev.sum()), outside_root=int((~inside).sum()), uncertified_at_90=int((depth < 0).sum()),
         depth_median=float(np.median(depth[good])), depth_p90=float(np.quantile(depth[good], .9)),
         leaves_est=est, nodes_est=2 * est, cpu_s=time.process_time() - t, depth=depth.tolist())
print({k: v for k, v in R.items() if k != "depth"}, flush=True); json.dump(R, open(out, "w"))
