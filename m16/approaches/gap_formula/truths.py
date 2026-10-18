"""Ground-truth accuracies of the scored models: 2e7 fresh samples each (seed 269000 + model index)."""
import json
import sys
import numpy as np
from gfcommon import MODELS, predict, T

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000
out = {}
for i, (name, m) in enumerate(MODELS.items()):
    rng = np.random.default_rng(269000 + i)
    ok = 0
    for _ in range(N // 1_000_000):
        x = rng.standard_normal((1_000_000, T))
        ok += int((predict(m, x) == np.argsort(x, 1)[:, -2]).sum())
    a = ok / N
    out[name] = {"acc": a, "se": float(np.sqrt(a * (1 - a) / N)), "N": N}
    print(name, out[name], flush=True)
json.dump(out, open("results/truths.json", "w"), indent=1)
