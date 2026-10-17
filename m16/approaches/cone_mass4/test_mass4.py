"""T109 validation: mass4 / mass4_rays vs the 30-digit mpmath reference (independent all-correlation Plackett path,
built from the defining rays or normals in 30 digits) on random and adversarial families; estT.orthant (T87's rule)
on the same cones for comparison; 1e8-sample MC on 4 cases.
usage: python test_mass4.py [n_per_family] [mc]"""
import json
import math
import os
import sys
import time

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
sys.path.insert(0, os.path.join(D, "..", "understand_m43"))
import mass4 as M  # noqa: E402
import reference as RF  # noqa: E402

try:
    import estT  # noqa: E402
except Exception as ex:                                   # pragma: no cover
    estT = None
    print("estT not importable:", ex)

NPF = int(sys.argv[1]) if len(sys.argv) > 1 else 12
DO_MC = len(sys.argv) > 2 and sys.argv[2] == "mc"
rng = np.random.default_rng(109)


def make(name):
    """-> ('rays', R) or ('normals', N)"""
    fam, e = (name.rsplit("_", 1) + ["0"])[:2] if name[-1].isdigit() else (name, "0")
    eps = 10.0 ** -float(e)
    if fam == "gauss_normals":
        return "normals", rng.standard_normal((4, 4))
    if fam == "gauss_rays":
        return "rays", rng.standard_normal((4, 4))
    if fam == "cap_rays":                                  # estT-like: rays in a random cap
        c = rng.standard_normal(4); return "rays", c + rng.uniform(0.05, 2) * rng.standard_normal((4, 4))
    if fam == "flat_rays":                                 # rays nearly in a hyperplane
        R = rng.standard_normal((4, 4)); R[:, 3] *= eps
        return "rays", R @ np.linalg.qr(rng.standard_normal((4, 4)))[0]
    if fam == "close_rays":                                # two rays nearly parallel
        R = rng.standard_normal((4, 4)); R[1] = R[0] + eps * rng.standard_normal(4); return "rays", R
    if fam == "needle_rays":                               # all rays in a tiny cap
        c = rng.standard_normal(4); return "rays", c + eps * rng.standard_normal((4, 4))
    if fam == "sliver_rays":                               # 3 rays nearly collinear + 1 generic
        R = rng.standard_normal((4, 4)); R[1] = R[0] + eps * rng.standard_normal(4)
        R[2] = R[0] + eps * rng.standard_normal(4); return "rays", R
    if fam == "close_normals":
        N = rng.standard_normal((4, 4)); N[1] = N[0] + eps * rng.standard_normal(4); return "normals", N
    if fam == "opp_normals":
        N = rng.standard_normal((4, 4)); N[1] = -N[0] + eps * rng.standard_normal(4); return "normals", N
    if fam == "flat_normals":
        N = rng.standard_normal((4, 4)); N[3] = N[:3].T @ rng.standard_normal(3) + eps * rng.standard_normal(4)
        return "normals", N
    raise ValueError(name)


FAMS = ["gauss_normals", "gauss_rays", "cap_rays"] + [f"{f}_{e}" for f in (
    "flat_rays", "close_rays", "needle_rays", "sliver_rays", "close_normals", "opp_normals", "flat_normals")
    for e in (2, 4, 6, 8)]
res = {}
t0 = time.time()
for fname in FAMS:
    E, Eest, O, refs = [], [], [], []
    for _ in range(NPF):
        kind, X = make(fname)
        if kind == "rays":
            r = float(RF.ref_rays(X)); o = M.Ops(); p = M.mass4_rays(X, o); R = X
        else:
            r = float(RF.ref_plackett(N=X)); o = M.Ops(); p = M.mass4(X, o)
            R = np.linalg.inv(X).T                       # rays of {N x >= 0}
        E.append(abs(p - r)); O.append(o.n); refs.append(r)
        if estT is not None:
            try:
                Eest.append(abs(estT.orthant(R, estT.Ops()) - r))
            except Exception:
                Eest.append(float("nan"))
    res[fname] = dict(max_abs_err=float(max(E)), mean_ops=float(np.mean(O)), max_ops=int(max(O)),
                      estT_max_abs_err=float(np.nanmax(Eest)) if Eest else None,
                      mass_range=[float(min(refs)), float(max(refs))])
    print(f"{fname:16s} mass4 max err {max(E):.1e}  ops mean {np.mean(O):.0f} max {max(O)} | estT max err "
          f"{res[fname]['estT_max_abs_err']:.1e} | mass [{min(refs):.1e},{max(refs):.1e}]  {time.time() - t0:.0f}s",
          flush=True)
json.dump(res, open(f"{D}/test_mass4.json", "w"), indent=1)
if DO_MC:
    mc = []
    for k in range(4):
        N = rng.standard_normal((4, 4))
        hit = tot = 0
        for _ in range(100):
            x = rng.standard_normal((1_000_000, 4)); hit += int(((x @ N.T) >= 0).all(1).sum()); tot += 1_000_000
        p = hit / tot; se = math.sqrt(p * (1 - p) / tot); m4 = M.mass4(N)
        mc.append(dict(mc=p, se=se, ref=float(RF.ref_plackett(N=N)), mass4=m4, z=(m4 - p) / se))
        print(mc[-1], flush=True)
    json.dump(mc, open(f"{D}/test_mc.json", "w"), indent=1)
