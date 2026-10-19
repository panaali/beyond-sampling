"""L = 32 truths by plain MC: N fresh language samples (chunks of 50k), saves every per-sequence max p_illegal
> 1e-3 (float32) and N. usage: python mc32.py MODEL N SEED -> data/t446/<MODEL>_L32_mc_<SEED>.npz"""
import sys
import time

import numpy as np

from est import phi_all, sample_law
from s3lib import DATA, Dyck2, load

name, N, seed = sys.argv[1], int(float(sys.argv[2])), int(sys.argv[3])
model, _ = load(name)
lang = Dyck2(L=32)
rng = np.random.default_rng(seed)
keep = []
t0 = time.time()
for i in range(0, N, 50000):
    y = sample_law(min(50000, N - i), 32, rng)
    m = phi_all(model, lang, y).max(1)
    keep.append(m[m > 1e-3].astype(np.float32))
    if i % 5_000_000 == 0:
        print(name, i, f"{time.time() - t0:.0f}s", flush=True)
np.savez(f"{DATA}/{name}_L32_mc_{seed}.npz", maxes=np.concatenate(keep), N=N, seconds=time.time() - t0)
print(name, "done", f"{time.time() - t0:.0f}s")
