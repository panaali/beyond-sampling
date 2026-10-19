"""Independent plain-sampling truths for zoo(16,T) seeds (own RNG stream). Appends to truths.json."""
import json, os, sys
import numpy as np
import core as C
T = int(sys.argv[1]); n = int(sys.argv[2]); seeds = [int(s) for s in sys.argv[3].split(",")]
path = "truths.json"
for s in seeds:
    m = C.zoo(16, T, s); rng = np.random.default_rng(424242 + 100 * T + s); ok = 0; ch = 500_000
    for _ in range(n // ch):
        x = rng.standard_normal((ch, T)); ok += int((C.forward(m, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
    p = ok / n
    d = json.load(open(path)) if os.path.exists(path) else {}
    d[f"16:{T}:{s}"] = dict(acc=p, se=float(np.sqrt(p * (1 - p) / n)), n=n, ok=ok)
    json.dump(d, open(path, "w"), indent=1); print(T, s, d[f"16:{T}:{s}"], flush=True)
