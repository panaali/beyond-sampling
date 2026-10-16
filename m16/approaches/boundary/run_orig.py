import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from estimator import est_boundary
M = (Wi, Wh, Wo)
b = int(sys.argv[1]); seed0 = int(sys.argv[2]); reps = int(sys.argv[3])
for r in range(reps):
    t = time.time()
    a, u, terms, ns, cost = est_boundary(M, b, np.random.default_rng(seed0 + r), details=True)
    print(f"{a:.6f} {u:.0f} {time.time()-t:.1f}s", {k: round(v, 5) for k, v in terms.items()}, ns, flush=True)
