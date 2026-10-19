"""T450 runner: unmodified estT_fix run with every orthant() call also evaluated under the candidate t-rules.

usage: OMP_NUM_THREADS=1 python run.py H T SEED MODE [--sub RULE] -> runs/H{H}T{T}_{SEED}_{MODE}[_sub-RULE].json
  MODE exact | recipe108.  --sub RULE: the gate run, with RULE substituted into the engine (no wrapper).
The error is a plain sum of orthant() values (estT._cell_error -> mass -> orthant), so for each candidate
    acc_c = 1 - sum orthant_c,   ops_c = ops_ref - sum ops(orthant_ref) + sum ops(orthant_c).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); A = os.path.join(HERE, "..")
sys.path[:0] = [HERE, os.path.join(A, "understand_m43"), os.path.join(A, "arc_m43_repro")]
assert hashlib.sha256(open(os.path.join(HERE, "estT.py"), "rb").read()).hexdigest()[:12] == "a88f62c41a94"
import estT  # noqa: E402
from m43 import Ops  # noqa: E402
from zoo44 import zoo  # noqa: E402


def graded(n, p=3):
    x, w = np.polynomial.legendre.leggauss(n); s = (x + 1) / 2; w = w / 2
    S = (1 - s) ** p                                                    # 1 - t
    return S, 1 - S, w * p * (1 - s) ** (p - 1)


def logrule(n):
    x, w = np.polynomial.legendre.leggauss(n); v = 35.0 * (x + 1); S = np.exp(-v)
    return S, 1 - S, 35.0 * w * S


RULES = {"G12": graded(12), "G16": graded(16), "G24": graded(24), "L24": logrule(24), "L32": logrule(32)}  # pre-registered
POSTHOC = {"G32": graded(32), "G48": graded(48), "G64": graded(64), "P4_32": graded(32, 4), "P4_48": graded(48, 4),
           "P5_32": graded(32, 5), "P5_48": graded(48, 5), "L48": logrule(48), "L64": logrule(64), "L96": logrule(96)}
if os.environ.get("T450_POSTHOC"):                    # post hoc (added after every pre-registered rule failed)
    RULES.update({k: POSTHOC[k] for k in os.environ["T450_POSTHOC"].split(",")})
if os.environ.get("T450_ONLY"):                       # (8,5): G24 (primary) + chosen post-hoc rules only (runtime)
    RULES = {k: {**RULES, **POSTHOC}[k] for k in os.environ["T450_ONLY"].split(",")}
REF = (estT._LOG_S, estT._LOG_T, estT._LOG_W)
REF5 = estT._LOG5[0]
assert np.array_equal(REF[0], logrule(128)[0]) and np.array_equal(REF5[2], logrule(128)[2])


def use(rule):
    S, Tt, W = rule
    estT._LOG_S, estT._LOG_T, estT._LOG_W = S, Tt, W
    estT._LOG5[0] = (S, Tt, W)


def restore():
    estT._LOG_S, estT._LOG_T, estT._LOG_W = REF
    estT._LOG5[0] = REF5


orth0 = estT.orthant
ACC = {k: dict(sum=0.0, ops=0, maxd=0.0, sumabs=0.0) for k in RULES}
REFO = dict(sum=0.0, ops=0, n=0, nthin=0)


def orth_wrap(Rays, ops):
    o = Ops(); v = orth0(Rays, o); ops.add(o.n)
    REFO["sum"] += v; REFO["ops"] += o.n; REFO["n"] += 1
    for k, rule in RULES.items():
        use(rule); oc = Ops(); vc = orth0(Rays, oc); restore()
        a = ACC[k]; a["sum"] += vc; a["ops"] += oc.n
        d = abs(vc - v); a["maxd"] = max(a["maxd"], d); a["sumabs"] += d
    return v


if __name__ == "__main__":
    H, T, s, mode = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    sub = sys.argv[sys.argv.index("--sub") + 1] if "--sub" in sys.argv else None
    m = zoo(H, T, s)
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    out = os.path.join(HERE, "runs", f"H{H}T{T}_{s}_{mode}" + (f"_sub-{sub}" if sub else "") + ".json")
    if sub:
        use({**RULES, **POSTHOC}[sub])
    else:
        estT.orthant = orth_wrap
    o = Ops(); t0 = time.process_time(); err = {}
    acc = estT.estT(m, o, mode=mode, info=err)
    r = dict(H=H, T=T, seed=s, mode=mode, sub=sub, acc=acc, ops=o.n, cpu=time.process_time() - t0, err=err["err"],
             stats={k: v for k, v in estT.STATS.items()})
    if not sub:
        r["ref_orth"] = REFO
        r["cand"] = {k: dict(acc=1 - a["sum"], dacc=(1 - a["sum"]) - acc, ops=o.n - REFO["ops"] + a["ops"],
                             max_dsimplex=a["maxd"], sum_absd=a["sumabs"]) for k, a in ACC.items()}
        r["err_resum_check"] = REFO["sum"] - err["err"]
    json.dump(r, open(out, "w"), indent=1)
    print(os.path.basename(out), f"acc {acc:.12f} ops {o.n:.3e} cpu {r['cpu']:.0f}s", flush=True)
    if not sub:
        for k, c in r["cand"].items():
            print(f"  {k}: dacc {c['dacc']:+.2e} ops ratio {c['ops'] / o.n:.3f} max dsimplex {c['max_dsimplex']:.1e}")
