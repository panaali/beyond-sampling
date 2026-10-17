"""High-precision MC truth for one model (adjudicates FX vs T87's exact at (4,4)). usage: python mc_big.py H T seed n"""
import sys
import time

import numpy as np

from zoo import forward, zoo_model

H, T, s, n = map(int, sys.argv[1:5])
m = zoo_model(H, T, s); rng = np.random.default_rng(12345 + s)
ok = 0; done = 0; t0 = time.time(); chunk = 2_000_000
while done < n:
    x = rng.standard_normal((chunk, T))
    ok += int((forward(m, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum()); done += chunk
p = ok / done
print(f"M_{H},{T} seed {s}: MC {done:.2e} samples: {p:.8f} +- {np.sqrt(p * (1 - p) / done):.1e} ({time.time() - t0:.0f} s)")
