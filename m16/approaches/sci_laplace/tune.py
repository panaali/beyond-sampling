"""Per-face cost and variance of the hybrid walk vs k0 (original model; scrambles vary)."""
import sys, time
import numpy as np
from lqt import *
model = get_model(sys.argv[1] if len(sys.argv) > 1 else "orig")
for k0 in (0.02, 0.04, 0.08):
    t = time.time()
    a = [e23(model, 4, s, k0) for s in range(6)]
    b = [e12(model, 16, s, k0) for s in range(6)]
    print("k0", k0, "e23 %.5f sd %.5f cost/face %.2f | e12 %.5f sd %.5f cost/face %.2f  (%.0fs)" % (
        np.mean([x[0] for x in a]), np.std([x[0] for x in a]), np.mean([x[1] for x in a]) / 1440,
        np.mean([x[0] for x in b]), np.std([x[0] for x in b]), np.mean([x[1] for x in b]) / 720, time.time() - t), flush=True)
