"""T455 validation on a cbab checkpoint.
V1: rmass.upper (float) >= rmass.upper_arb (arb ball upper endpoint), same grid, on n random open boxes.
V2: rmass.upper >= MC of the box mass - 3 s.e.: u uniform in the box, exact radial integral in g:
    mass = vol * E_u[(2 pi)^-5 16 |y|^-10 Gamma(5, |y|^2 t0^2 / 2)]  (independent of the (r, g) quadrature).
V3: soundness of pruned boxes (reservoir): uniform points, margins by the reference forward on raw x at random g > 0;
    no events, and sampled margins >= the alpha-CROWN lower bound recomputed on the box.
usage: python validate.py p1,p2,p3 CKPT.pt OUT.json [v1,v2,v3] [n]"""
import json
import math
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "late_triple_verify"))
import net    # noqa: E402
import rmass  # noqa: E402

th.set_num_threads(1)
tri = tuple(int(v) for v in sys.argv[1].split(",")); S = th.load(sys.argv[2]); out = sys.argv[3]
parts = sys.argv[4].split(",") if len(sys.argv) > 4 else ["v1", "v2", "v3"]; n = int(sys.argv[5]) if len(sys.argv) > 5 else 200
R = json.load(open(out)) if os.path.exists(out) else {}
rng = np.random.default_rng(4550); A, c = net.ymap(tri); p3 = tri[2]
lo, hi = S["lo"], S["hi"]; sel = th.as_tensor(rng.choice(len(lo), min(n, len(lo)), replace=False)); l, h = lo[sel], hi[sel]
m = net.minsq(l, h); up = rmass.upper(l, h, 48, 96, m)

if "v1" in parts:
    bad = 0; worst = np.inf
    for i in range(len(l)):
        a = rmass.upper_arb(l[i].tolist(), h[i].tolist(), 48, 96, float(m[i]))
        bad += float(up[i]) < a; worst = min(worst, (float(up[i]) - a) / a)
    R["V1"] = dict(n=len(l), float_below_arb=int(bad), min_rel_gap=worst, PASS=bool(bad == 0)); print("V1", R["V1"], flush=True)

if "v2" in parts:
    viol = 0; ratios = []
    for i in range(len(l)):
        u = l[i] + (h[i] - l[i]) * th.as_tensor(rng.uniform(size=(20000, 9))); y = u @ A.T + c; s2 = (y * y).sum(1)
        w = s2 * net.T0 ** 2 / 2
        f = (2 * math.pi) ** -5 * 16 * s2 ** -5 * 24 * th.exp(-w) * (1 + w + w ** 2 / 2 + w ** 3 / 6 + w ** 4 / 24)
        vol = float((h[i] - l[i]).prod()); mc = vol * float(f.mean()); se = vol * float(f.std()) / math.sqrt(len(f))
        viol += float(up[i]) < mc - 3 * se; ratios.append(float(up[i]) / mc if mc > 0 else np.inf)
    r = np.array(ratios); R["V2"] = dict(n=len(l), violations=int(viol), ratio_median=float(np.median(r)),
                                         ratio_min=float(r.min()), PASS=bool(viol == 0)); print("V2", R["V2"], flush=True)

if "v3" in parts:
    import lbab
    B = lbab.Bounder(tri)
    rl = th.cat([x[0] for x in S["res"]]); rh = th.cat([x[1] for x in S["res"]]); pts = 500; ev = 0; viol = 0; worst = np.inf
    for i0 in range(0, len(rl), 64):
        lb = B(rl[i0:i0 + 64], rh[i0:i0 + 64], "alpha-CROWN")[0]; lbc = B(rl[i0:i0 + 64], rh[i0:i0 + 64], "CROWN")[0]
        lb = th.maximum(lb, lbc)
        for j in range(len(lb)):
            i = i0 + j; u = rl[i] + (rh[i] - rl[i]) * th.as_tensor(rng.uniform(size=(pts, 9))); y = u @ A.T + c
            g = th.as_tensor(2.5 + rng.exponential(1.0, size=(pts, 1))); mg = net.margins_x(y * g, p3) / g
            ev += int((mg <= 0).all(1).sum()); gap = float((mg.min(0).values - lb[j]).min()); worst = min(worst, gap)
            viol += gap < -1e-7
    R["V3"] = dict(boxes=len(rl), points=pts * len(rl), events=ev, bound_violations=int(viol), worst_gap=worst,
                   PASS=bool(ev == 0 and viol == 0)); print("V3", R["V3"], flush=True)
json.dump(R, open(out, "w"), indent=1)
