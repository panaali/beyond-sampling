"""DEV-model check (MC reference, diagnosis only): is the K=4 unconditional estimate on M_16,10 (-0.37 pp on orig)
a property of the method or a coincidence? Runs ARC basic propagation on DEV models (never scored).
Usage: python dev_check.py T K mode n_models"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "m16")
import numpy as np
import arc_kprop as A
import leaderboard as LB

T, K, mode, nm = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
dev = LB.models(16, T, dev=True)
names = [n for n in dev if n.startswith("dev_s0.0025")][: nm // 2] + [n for n in dev if n.startswith("dev_s0.02")][: nm - nm // 2]
rng = np.random.default_rng(7)
for n in names:
    m = dev[n]
    x = rng.standard_normal((1_000_000, T)); p = LB.correct(m, x).mean()
    t0 = time.time()
    e, raw, _ = (A.estimate_uncond if mode == "uncond" else A.estimate_cell)(m, K)
    print(f"{n:16s} MC acc {p:.4f} (se {np.sqrt(p*(1-p)/1e6)*100:.3f} pp)  ARC K={K} {mode}: {e:.4f}  bias {100*(e-p):+.2f} pp  "
          f"cost {raw/LB.fpe_flops(16, T):.0f} FPE  {time.time()-t0:.0f}s", flush=True)
