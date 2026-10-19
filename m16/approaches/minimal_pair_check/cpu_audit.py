"""T379 CPU-time audit of T10's frozen estimator (not scored; the estimator is unchanged).
1. CPU time of est vs plain sampling at the same cost (single thread), 16k and 64k.
2. cProfile: where the time goes (walk's Python loop vs numpy kernels).
3. Vectorised-walk estimate: count walk() iterations and row-cells during est (by wrapping walk for this audit only),
   time the SAME walk code on one large batch (one call, all faces at once) to get the marginal CPU s per row-cell
   and per charged FPE, and compare with sampling's CPU s per FPE.
usage: python cpu_audit.py MODEL_SEED   -> cpu_audit_<seed>.json"""
import cProfile
import json
import os
import pstats
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

from frozen_check import E  # noqa: E402
import models as M  # noqa: E402
import score as S  # noqa: E402

seed = int(sys.argv[1]); m = M.load(seed); out = {}
orig_walk = E.W.walk
stats = {}


def counted(*a, **k):
    t0 = time.process_time(); mass, cost, cells = orig_walk(*a, **k)
    stats["calls"] += 1; stats["iters"] += int(cells.max()) if len(cells) else 0
    stats["rowcells"] += int(cells.sum()); stats["faces"] += len(cells); stats["walk_s"] += time.process_time() - t0
    return mass, cost, cells


for b in S.BUDGETS:
    spf = S.sampling_sec_per_fpe(m, b)
    t0 = time.process_time(); _, c = E.est(m, b); cpu = time.process_time() - t0
    stats.update(calls=0, iters=0, rowcells=0, faces=0, walk_s=0.0)
    E.W.walk = counted
    E.est(m, b)
    E.W.walk = orig_walk
    prof = cProfile.Profile(); prof.enable(); E.est(m, b); prof.disable()
    ps = pstats.Stats(prof); tot = ps.total_tt
    top = sorted(((v[2], f"{os.path.basename(k[0])}:{k[2]}") for k, v in ps.stats.items()), reverse=True)[:12]
    out[b] = dict(cost=c, cpu_s=cpu, sampling_s_per_fpe=spf, time_ratio=cpu / (c * spf), walk=dict(stats),
                  profile_total_s=tot, profile_top_tottime=[(round(t, 4), n) for t, n in top])
    print(b, json.dumps(out[b]), flush=True)

# marginal cost of the same walk code on one large batch (no per-call overhead), using the estimator's own face law
T = 5; U = E.C.sobol(T, 1 << 16, 12345)
row = {}
for kind, h in E.strata(T):
    y, a, bb, tau, wt = E.face_points_strat(T, kind, h, U)
    r = np.linalg.norm(y, axis=1)
    t0 = time.process_time()
    F, c, cells = E.W.walk(m, y / r[:, None], a, bb, np.minimum(tau / r, 1e8), kind, cdf=E.hom.tcdf(T))
    dt = time.process_time() - t0
    row[f"{kind}{h}"] = dict(faces=len(y), rowcells=int(cells.sum()), fpe=float(c.sum()), cpu_s=dt,
                             s_per_fpe=dt / c.sum(), iters=int(cells.max()))
    print(kind, h, row[f"{kind}{h}"], flush=True)
spf_big = S.sampling_sec_per_fpe(m, 1 << 16, reps=4)
fpe = sum(v["fpe"] for v in row.values()); cpu = sum(v["cpu_s"] for v in row.values())
out["large_batch"] = dict(strata=row, s_per_fpe=cpu / fpe, sampling_s_per_fpe=spf_big,
                          time_ratio_vectorised=(cpu / fpe) / spf_big)
print("large batch: walk s/FPE", cpu / fpe, "sampling s/FPE", spf_big, "ratio", (cpu / fpe) / spf_big)
json.dump(out, open(f"cpu_audit_{seed}.json", "w"), indent=1)
