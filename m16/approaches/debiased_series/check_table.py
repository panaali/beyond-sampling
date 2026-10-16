"""Check level 0: tabulated E[g] vs brute-force mean of g on fresh inputs (model-free given tau)."""
import sys, os, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_eval import noisy_models, T
from common import Wi, Wh, Wo
from ds import build_table, grid, HI0, W
from surrogate import thresholds, g
for it in (3, 5):
    H = build_table(it)
    for name, m in [("orig", (Wi, Wh, Wo)), ("n05_1", noisy_models(6, .005)[1])]:
        t23, t12, _ = thresholds(m, it, hi0=HI0)
        i = lambda t: np.rint(t / (HI0 / 2 ** it) - 0.5).astype(int)
        Eg = H[i(t23), i(t12)].mean()
        rng = np.random.default_rng(555); vals = []
        for _ in range(8):
            vals.append(g(rng.standard_normal((500_000, T)), t23, t12, 0.0, W).mean())
        print(f"iters={it} {name}: table {Eg:.5f}  brute {np.mean(vals):.5f} +- {np.std(vals)/np.sqrt(8):.5f}", flush=True)
