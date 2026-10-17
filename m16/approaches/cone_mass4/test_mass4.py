"""T109 validation: mass4 vs the 30-digit mpmath reference (independent all-correlation Plackett path) on random and
adversarial thin cones; also 1e8-sample MC on a few. usage: python test_mass4.py [n_per_family] [mc]"""
import json
import math
import os
import sys
import time

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import mass4 as M  # noqa: E402
import reference as RF  # noqa: E402

NPF = int(sys.argv[1]) if len(sys.argv) > 1 else 40
DO_MC = len(sys.argv) > 2 and sys.argv[2] == "mc"
rng = np.random.default_rng(109)
NODES = (3, 4, 5, 6, 8, 12)


def normals_from_rays(R):
    return np.linalg.inv(R.T)


def fam(name):
    out = []
    for _ in range(NPF):
        if name == "gauss_normals":
            N = rng.standard_normal((4, 4))
        elif name == "gauss_rays":
            N = normals_from_rays(rng.standard_normal((4, 4)))
        elif name.startswith("flat_rays"):          # rays nearly in a hyperplane (thin, small det)
            eps = 10.0 ** -float(name.split("_")[-1])
            R = rng.standard_normal((4, 4)); R[:, 3] *= eps
            N = normals_from_rays(R @ np.linalg.qr(rng.standard_normal((4, 4)))[0])
        elif name.startswith("close_rays"):         # two rays nearly parallel
            eps = 10.0 ** -float(name.split("_")[-1])
            R = rng.standard_normal((4, 4)); R[1] = R[0] + eps * rng.standard_normal(4)
            N = normals_from_rays(R)
        elif name.startswith("needle"):             # all 4 rays in a small cap (tiny mass)
            eps = 10.0 ** -float(name.split("_")[-1])
            c = rng.standard_normal(4)
            R = c + eps * rng.standard_normal((4, 4))
            N = normals_from_rays(R)
        elif name.startswith("close_normals"):      # two facet normals nearly parallel (rho -> 1)
            eps = 10.0 ** -float(name.split("_")[-1])
            N = rng.standard_normal((4, 4)); N[1] = N[0] + eps * rng.standard_normal(4)
        elif name.startswith("opp_normals"):        # two facet normals nearly opposite (rho -> -1)
            eps = 10.0 ** -float(name.split("_")[-1])
            N = rng.standard_normal((4, 4)); N[1] = -N[0] + eps * rng.standard_normal(4)
        elif name.startswith("flat_normals"):       # normals nearly dependent (det C -> 0)
            eps = 10.0 ** -float(name.split("_")[-1])
            N = rng.standard_normal((4, 4)); N[3] = N[:3].T @ rng.standard_normal(3) + eps * rng.standard_normal(4)
        out.append(N)
    return out


FAMS = ["gauss_normals", "gauss_rays"] + [f"{f}_{e}" for f in ("flat_rays", "close_rays", "needle", "close_normals",
                                                                 "opp_normals", "flat_normals") for e in (2, 4, 6, 8)]
res = {}
t0 = time.time()
for fname in FAMS:
    errs = {n: [] for n in NODES}; ops = []; refs = []
    for N in fam(fname):
        C = M.corr_from_normals(N, M.Ops())
        r = float(RF.ref_plackett(N))
        refs.append(r)
        for n in NODES:
            o = M.Ops()
            p = M.mass4(N, o, n)
            errs[n].append(abs(p - r))
            if n == 8:
                ops.append(o.n)
    res[fname] = dict(max_abs_err={n: float(max(v)) for n, v in errs.items()},
                      ref_range=[float(min(refs)), float(max(refs))], ops_n8=int(np.mean(ops)))
    print(fname, {n: "%.1e" % max(v) for n, v in errs.items()}, "mass in [%.1e, %.1e]" % (min(refs), max(refs)),
          round(time.time() - t0), flush=True)
json.dump(res, open(f"{D}/test_mass4.json", "w"), indent=1)
if DO_MC:
    mc = []
    for k in range(4):
        N = rng.standard_normal((4, 4)); C = M.corr_from_normals(N, M.Ops())
        hit = 0; tot = 0
        for _ in range(100):
            x = rng.standard_normal((1_000_000, 4)); hit += int(((x @ N.T) >= 0).all(1).sum()); tot += 1_000_000
        p = hit / tot; se = math.sqrt(p * (1 - p) / tot)
        mc.append(dict(mc=p, se=se, ref=float(RF.ref_plackett(N)), mass4=M.mass4(N), z=(M.mass4(N) - p) / se))
        print(mc[-1], flush=True)
    json.dump(mc, open(f"{D}/test_mc.json", "w"), indent=1)
