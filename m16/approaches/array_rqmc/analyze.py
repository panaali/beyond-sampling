"""Compute-adjusted comparison from results/*.json.

Cost model, in forward-pass equivalents (one forward pass = 10 x (16x16 + 16) + 10x16 = 2880 multiply-adds):
  mc, strat            : 1.0 per sample (strat: model evals only; rejection draws not counted, as in stratified.py)
  9 chain steps        : 2448/2880 = 0.85
  exact last step (CMC): charged 1.0 (op count of an incremental implementation is ~2,050 MACs ~ 0.71:
                         Wh h, 16 breakpoints, sort of 18, 19 intervals x (rank-1 logit update + 10-way
                         feasibility + ndtr)); the vectorised numpy code here does more work (dense recompute)
  sorting (array-RQMC) : per sorted step (8 steps) f + 2 log2 n ops (importance function + comparisons/permute),
                         f = 0 for m1/m2/gap keys, 30 for the linear value function; 2-D batch sort counts both keys.
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FP = 2880.0


def cost(name, n):
    lg = np.log2(n)
    base = {"mc": 1.0, "strat": 1.0}.get(name)
    if base is not None:
        return base
    c = 0.85 + 1.0
    if name.startswith("arr"):
        f = 30 if name.endswith("V") else 0
        keys = 2 if name.startswith("arr2") else 1
        c += 8 * (f + keys * 2 * lg) / FP
    return c


rows = {}
for fn in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
    if "truth" in fn:
        continue
    d = json.load(open(fn)); tr = np.array(d["truth"])[:, None]
    for key, E in d["res"].items():
        name, n = key.split("|"); n = int(n); E = np.array(E)
        if E.shape[1] < 4:
            continue
        rows[(d["set"], name, n)] = dict(var=E.var(1, ddof=1).mean(), mse=((E - tr) ** 2).mean(),
                                         p=tr.mean(), reps=E.shape[1] * E.shape[0])

print(f"{'set':6s} {'method':10s} {'n':>6s} {'cost/n':>6s} {'RMSE':>9s} {'n*var':>8s} "
      f"{'gain vs MC (evals)':>18s} {'gain vs MC (cost)':>17s} {'gain vs strat (cost)':>20s}")
for (s, name, n), r in sorted(rows.items()):
    trs = json.load(open(glob.glob(os.path.join(HERE, "results", f"{s}_*.json"))[0]))["truth"]
    pq = np.mean([t * (1 - t) for t in trs])           # per-sample variance of plain sampling
    C = cost(name, n)
    g_eval = pq / (n * r["var"])                         # MSE ratio at equal number of chains / model calls
    g_cost = g_eval / C
    sv = rows.get((s, "strat", n))
    g_st = (sv["var"] * 1.0) / (r["var"] * C) if sv else float("nan")
    print(f"{s:6s} {name:10s} {n:6d} {C:6.2f} {np.sqrt(r['mse']):9.2e} {n*r['var']:8.5f} {g_eval:18.2f} {g_cost:17.2f} {g_st:20.2f}")

print("\nempirical convergence: slope of log(var) vs log(n)  (MC = -1)")
for s in sorted({k[0] for k in rows}):
    for name in sorted({k[1] for k in rows if k[0] == s}):
        ns = sorted(k[2] for k in rows if k[0] == s and k[1] == name)
        if len(ns) >= 2:
            v = [rows[(s, name, n)]["var"] for n in ns]
            b = np.polyfit(np.log(ns), np.log(v), 1)[0]
            print(f"  {s:6s} {name:10s} n={ns[0]}..{ns[-1]}: var ~ n^{b:.2f}  (RMSE ~ n^{b/2:.2f})")
