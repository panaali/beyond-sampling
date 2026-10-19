"""Wall-time benchmark: compiled walk est vs optimised samplers, at equal MSE.
usage: python bench.py T  -> bench_T{T}.json. One thread (BLAS pinned to 1 by the caller's env)."""
import json
import sys
import time

import numpy as np

import models as Mo
import sampler as S
import walk_nb as WN

E = Mo.frozen()
C = E.C
frozen_walk = E.W.walk
T = int(sys.argv[1])
BUDGETS = (16000, 64000)
REP = 7
H = 16
step_f = 2 * H * H + 2 * H
FPE_FLOPS = T * step_f + 2 * H * T          # one length-T sequence: T recurrent steps + readout

# --- instrumentation: time spent inside the compiled walk
WT = [0.0]
def timed_walk(*a, **k):
    t0 = time.perf_counter(); r = WN.walk(*a, **k); WT[0] += time.perf_counter() - t0
    return r

# --- model-free precompute (Sobol rules, face-point maps): memoised by identity of the (cached) rule slice
orig_sobol, orig_fps = C.sobol, E.face_points_strat
memo_s, memo_f = {}, {}
def sobol_memo(nd, n, seed):
    k = (nd, n, seed)
    if k not in memo_s:
        memo_s[k] = orig_sobol(nd, n, seed)
    return memo_s[k]
def fps_memo(T_, kind, h, U):
    k = (T_, kind, h, U.__array_interface__["data"][0], len(U))
    if k not in memo_f:
        memo_f[k] = orig_fps(T_, kind, h, U)
    return memo_f[k]
def precompute(on):
    C.sobol = sobol_memo if on else orig_sobol
    E.face_points_strat = fps_memo if on else orig_fps


def time_est(m, B, mode, pre):
    """median wall time of est over REP warm repeats; walk-internal time; counters."""
    if mode == "numpy":
        E.W.walk = frozen_walk
    else:
        WN.MODE["fast"] = mode.startswith("fast"); WN.MODE["fm"] = mode == "fastfm"; E.W.walk = timed_walk
    precompute(pre)
    E.est(m, B)                                   # warm (JIT, memo)
    ts, ws = [], []
    WN.FLOPS[:] = 0
    reps = 3 if mode == "numpy" else REP
    for _ in range(reps):
        WT[0] = 0.0; t0 = time.perf_counter(); e, c = E.est(m, B); ts.append(time.perf_counter() - t0); ws.append(WT[0])
    precompute(False); E.W.walk = frozen_walk; WN.MODE["fm"] = False
    i = int(np.argsort(ts)[len(ts) // 2])
    return {"t": ts[i], "t_walk": ws[i], "est": e, "cost": c, "flops_exec": (WN.FLOPS / reps).tolist()}


def time_samp(m, n, kind):
    rng = np.random.default_rng(7)
    if kind == "compiled_pre":
        x = rng.standard_normal((T, n)); f = lambda: S.est_compiled(m, n, None, x=x)
    elif kind == "compiledfm_pre":
        x = rng.standard_normal((T, n)); f = lambda: S.count_correct_fm(*m, x) / n
    elif kind == "compiled_rng":
        f = lambda: S.est_compiled(m, n, rng)
    else:
        f = lambda: S.est_numpy(m, n, rng)
    f(); ts = []
    for _ in range(REP):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


models = Mo.groups(T); truths = Mo.truths()
out = {"T": T, "models": {}, "sampler": {}}
for name, m in models.items():
    row = {}
    for B in BUDGETS:
        row[B] = {f"{mode}{'_pre' if pre else ''}": time_est(m, B, mode, pre)
                  for mode, pre in (("fast", False), ("fast", True), ("fastfm", False),
                                                                ("fastfm", True), ("repro", False), ("numpy", False))}
        r = row[B]
        print(name, B, " ".join(f"{k} {v['t'] * 1e3:.1f}ms(walk {v['t_walk'] * 1e3:.1f})" for k, v in r.items()), flush=True)
    sm = {}
    for kind in ("compiled_rng", "compiled_pre", "compiledfm_pre", "numpy"):
        sm[kind] = {str(n): time_samp(m, n, kind) for n in (16000, 64000, 1 << 20)}
    print(name, "sampler s/FPE at 2^20:", {k: v[str(1 << 20)] / (1 << 20) for k, v in sm.items()}, flush=True)
    out["models"][name] = {"est": row, "sampler": sm, "truth": truths.get(f"{name}:T{T}")}
json.dump(out, open(f"{Mo.HERE}/bench_T{T}.json", "w"), indent=1)
