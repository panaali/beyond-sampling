"""DEV-model diagnostic for the hybrid (not scored): for each tau, the exact part's cost, mass and wrong mass, and the
walk's per-unit-cost variance with and without E (fixed Sobol set of n points per stratum, rule seed 31415).
usage: python diag_hybrid.py T model taus... [--n 2048]"""
import json
import sys
import time

import numpy as np

sys.path.append("../..")
import leaderboard as L  # noqa: E402
import hybrid as Hy  # noqa: E402
import mwalk  # noqa: E402

FE = Hy.FE
T = int(sys.argv[1]); name = sys.argv[2]
args = sys.argv[3:]
n = 2048
if "--n" in args:
    i = args.index("--n"); n = int(args[i + 1]); args = args[:i] + args[i + 2:]
taus = [float(x) for x in args]
ms = L.models(16, T, dev=name.startswith("dev"))
m = ms[name]


def walk_stats(keys):
    out = []
    for j, (kind, h) in enumerate(FE.strata(T)):
        U = FE.C.sobol(T, n, FE.RULE_SEED + 101 * j)
        FE.W.walk = lambda *a, **k: mwalk.walk(*a, excl=keys, **k)
        v, c = FE.run_points(m, T, kind, h, U)
        FE.W.walk = Hy.FROZEN_WALK
        M = FE.stratum_mass(T, kind, h)
        out.append(dict(kind=kind, h=h, M=M, mean=M * v.mean(), sd=M * v.std(), c=c.mean()))
    # Neyman-optimal variance x cost: (sum_h M_h sd_h sqrt(c_h))^2
    vc = sum(s["sd"] * np.sqrt(s["c"]) for s in out) ** 2
    return dict(err=sum(s["mean"] for s in out), varcost=vc, strata=out)


base = walk_stats(frozenset())
print(json.dumps(dict(name=name, T=T, tau=0, walk_err=base["err"], varcost=base["varcost"])), flush=True)
for tau in taus:
    t0 = time.process_time()
    ex = Hy.exact_part(m, tau)
    w = walk_stats(ex["keys"])
    print(json.dumps(dict(name=name, T=T, tau=tau, cells=len(ex["keys"]), massE=ex["mass"], wrongE=ex["wrong"],
                          errE=ex["err"], fpe=ex["fpe"], visited=ex["visited"], kept=ex["kept"],
                          walk_rest=w["err"], varcost=w["varcost"], var_ratio=w["varcost"] / base["varcost"],
                          parts={k: float(v) for k, v in ex["parts"].items()}, cpu=time.process_time() - t0)),
          flush=True)
