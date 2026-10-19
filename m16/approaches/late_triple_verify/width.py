"""T418 E2: certified u-cube half-width around exact slab points of a triple, per bounder, and covering estimates.
usage: python width.py p1,p2,p3 OUT.json [parts: C,A,abcin,abcrelu] [n]"""
import json
import math
import os
import sys
import time

import numpy as np
import torch as th

import net
import lbab

th.set_num_threads(1)
tri = tuple(int(v) for v in sys.argv[1].split(",")); out = sys.argv[2]
parts = sys.argv[3].split(",") if len(sys.argv) > 3 else ["C", "A"]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 200
R = json.load(open(out)) if os.path.exists(out) else {}
x = net.slab_sample(tri, n, np.random.default_rng(418)); u, g = net.x_to_u(x, tri); u = th.as_tensor(u)
M = net.UNet(tri)
with th.no_grad():
    mg = M(u)
ev = (mg <= 0).all(1); ne = ~ev
R["points"] = dict(n=n, events=int(ev.sum()), best_margin_median=float(mg.max(1).values.median()))
B = lbab.Bounder(tri)


def cube(w, idx):
    lo = u[idx] - w[:, None]; hi = u[idx] + w[:, None]; lo[:, 1:] = lo[:, 1:].clamp(min=0); return lo, hi


def bisect(cert, idx, lo_w, hi_w, steps):
    ok_lo = cert(lo_w, idx)
    for _ in range(steps):
        mid = (lo_w * hi_w).sqrt(); c = cert(mid, idx); lo_w = th.where(c, mid, lo_w); hi_w = th.where(c, hi_w, mid)
    return th.where(ok_lo, lo_w, th.zeros_like(lo_w))


# slab density in u (radial integral at the point) for T414's covering estimate
s2 = (th.as_tensor(x) / th.as_tensor(g)[:, None]).pow(2).sum(1).numpy(); ww = s2 * net.T0 ** 2 / 2
f = 16 * (2 * math.pi) ** -5 * s2 ** -5 * 24 * np.exp(-ww) * (1 + ww + ww ** 2 / 2 + ww ** 3 / 6 + ww ** 4 / 24)
Z = float(net.quad_mass(*[t[None] for t in net.root(tri)], nr=200, nq=16, ng=64)); pu = f / Z


def cover(w, idx, mult=1.0):
    w = np.asarray(w); m = ne.numpy()[idx] & (w > 0); p = pu[idx]; keep = m & (p >= np.quantile(p, .002))
    est = float(np.mean(np.where(keep, 1 / ((2 * np.where(w > 0, w, 1)) ** 9 * p), 0))) * mult
    return dict(median_w=float(np.median(w[ne.numpy()[idx]])), p10_w=float(np.quantile(w[ne.numpy()[idx]], .1)),
                uncertifiable=int(((w == 0) & ne.numpy()[idx]).sum()), cover_ideal=est, cover_halved=est * 2 ** 9)


allidx = np.arange(n)
for name, meth in [("C", "CROWN"), ("A", "alpha-CROWN")]:
    if name not in parts:
        continue
    t = time.process_time()

    def cert(w, idx, meth=meth):
        r = []
        for i in range(0, len(idx), 50):
            lo, hi = cube(w[i:i + 50], idx[i:i + 50]); r.append((B(lo, hi, meth)[0] > 0).any(1))
        return th.cat(r)
    w = bisect(cert, allidx, th.full((n,), 1e-6, dtype=net.dt), th.full((n,), 2.0, dtype=net.dt), 20 if name == "C" else 14)
    R[name] = dict(**cover(w.numpy(), allidx), cpu_s=time.process_time() - t, w=w.tolist())
    print(name, {k: v for k, v in R[name].items() if k != "w"}, flush=True); json.dump(R, open(out, "w"))

for name, mode, m_, steps in [("abcin", "input", 60, 6), ("abcrelu", "relu", 20, 4)]:
    if name not in parts:
        continue
    wA = th.as_tensor(R["A"]["w"][:m_], dtype=net.dt).clamp_min(1e-4); idx = allidx[:m_]
    abc = lbab.ABC(tri, 10.0, mode); th.set_default_dtype(th.float64); doms = {}; t = time.process_time()
    order = th.argsort(mg, 1, descending=True)

    def cert(w, idx):
        r = []
        for k, i in enumerate(idx):
            lo, hi = cube(w[k:k + 1], idx[k:k + 1]); v, d, _ = abc(lo[0], hi[0], order[i]); r.append(v)
            if v:
                doms[int(i)] = d                                                 # domains of the last certified cube
        return th.as_tensor(r)
    lo_w = wA.clone(); hi_w = 4 * wA
    for _ in range(steps):
        mid = (lo_w * hi_w).sqrt(); c = cert(mid, idx); lo_w = th.where(c, mid, lo_w); hi_w = th.where(c, hi_w, mid)
    dm = float(np.mean([doms.get(int(i), 1) for i in idx])) if doms else 1.0
    R[name] = dict(**cover(lo_w.numpy(), idx), gain_vs_A_median=float((lo_w / wA).median()), domains_mean=dm,
                   cover_ideal_x_domains=cover(lo_w.numpy(), idx)["cover_ideal"] * max(dm, 1),
                   cpu_s=time.process_time() - t, w=lo_w.tolist())
    R[name + "_A_same_points"] = cover(wA.numpy(), idx)
    print(name, {k: v for k, v in R[name].items() if k != "w"}, R[name + "_A_same_points"], flush=True); json.dump(R, open(out, "w"))
