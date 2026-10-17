"""Independent truth check (T74 item 1): own forward pass + plain MC with 1e7 samples per model vs exact.py.
usage: OMP_NUM_THREADS=1 python truth_mc.py   (writes truth_mc.json; ~8 CPU-min)"""
import json, os, time
import numpy as np
from common_review import base_model, zoo, noisy, mc_truth
from exact import exact_acc

N = 10_000_000
b = base_model()
cases = [("base", b)] + [(f"zoo{s}", zoo(s)) for s in range(1, 5)]
cases += [(f"s0.05_seed10_{i}", m) for i, m in enumerate(noisy(b, 12, 0.05, 10))]
cases += [(f"s0.10_seed20_{i}", m) for i, m in enumerate(noisy(b, 6, 0.10, 20))]
cases += [(f"s0.02_seed30_{i}", m) for i, m in enumerate(noisy(b, 6, 0.02, 30))]
rows = []
for j, (name, m) in enumerate(cases):
    t = time.time(); mc = mc_truth(m, N, 1000 + j); ex = exact_acc(m)
    se = np.sqrt(mc * (1 - mc) / N); z = (ex - mc) / se
    rows.append(dict(name=name, mc=mc, exact=ex, se=se, z=z))
    print(f"{name:20s} MC {mc:.6f} +- {se:.6f}  exact {ex:.6f}  z {z:+.2f}  ({time.time()-t:.0f}s)", flush=True)
zs = np.array([r["z"] for r in rows])
print(f"n={len(zs)} mean z {zs.mean():+.2f}, rms z {np.sqrt((zs**2).mean()):.2f}, max |z| {np.abs(zs).max():.2f}")
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "truth_mc.json"), "w"), indent=1)
