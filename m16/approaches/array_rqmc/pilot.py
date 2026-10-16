"""Quick look (original model): per-chain variance of CMC vs indicator; RMSE of RQMC / Array-RQMC variants."""
import sys, time, numpy as np
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/array_rqmc")
from common import Wi, Wh, Wo
from core import *
M = (Wi, Wh, Wo)
rng = np.random.default_rng(1)
S = init(400_000); [step(M, S, rng.standard_normal(400_000)) for _ in range(9)]
y = last_step_value(M, S); p = y.mean()
print(f"CMC per-chain var {y.var():.5f} vs indicator {p*(1-p):.5f} -> x{p*(1-p)/y.var():.2f}")
V = ValueFn.fit(M)
methods = {"mc+cmc": lambda m, n, r: est_mc(m, n, r), "rqmc9+cmc": est_rqmc,
           "arr m1": lambda m, n, r: est_array(m, n, r, sort_m1),
           "arr gap": lambda m, n, r: est_array(m, n, r, sort_gap),
           "arr V": lambda m, n, r: est_array(m, n, r, V),
           "arr2 (m1,V)": lambda m, n, r: est_array(m, n, r, V, sort2=sort_m1)}
R = 24
for n in (2**10, 2**12, 2**14):
    for name, f in methods.items():
        t0 = time.time(); e = np.array([f(M, n, np.random.default_rng(10_000 + r)) for r in range(R)])
        print(f"n={n:6d} {name:12s} sd={e.std(ddof=1):.2e} n*var={n*e.var(ddof=1):.5f} mean-truth={e.mean()-0.953179:+.1e} ({time.time()-t0:.1f}s)", flush=True)
