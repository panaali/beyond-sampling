"""Select the v5b spread variant on LB3 DEV models only (noisy copies drawn with seed 1; never scored).
Criterion fixed before running: smallest RMS bias over the 24 DEV models at 131,072 nodes."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/lb3/m16")
import leaderboard as L
import v5
dev = L.models(dev=True)
out = {}
truth = {}
for n, m in dev.items():                      # DEV truths: 4M plain samples each (seed 274000+i), model-free code
    rng = np.random.default_rng(274000 + len(truth))
    ok = 0
    for _ in range(8):
        x = rng.standard_normal((500_000, 10))
        ok += int((v5.forward_argmax(m, x) == np.argsort(x, 1)[:, -2]).sum())
    truth[n] = ok / 4_000_000
variants = {"v5a": lambda m: v5.est(m, 131072)[0]}
for rv, nl in ((1, 7), (1, 3), (1, 1), (2, 3), (2, 7)):
    variants[f"spread_r{rv}_n{nl}"] = (lambda rv, nl: lambda m: v5.est_spread(m, 131072 * (1 + rv), rivals=rv, n_low=nl)[0])(rv, nl)
for k, f in variants.items():
    b = {n: f(m) - truth[n] for n, m in dev.items()}
    out[k] = {"rms_bias": float(np.sqrt(np.mean(np.square(list(b.values()))))), "bias": b}
    print(k, "RMS %.4f  by sigma:" % out[k]["rms_bias"],
          {s: round(float(np.sqrt(np.mean([v ** 2 for n, v in b.items() if n.startswith(f'dev_s{s}_')]))), 4) for s in L.SIGMAS}, flush=True)
json.dump({"truths": truth, "variants": out}, open("results/dev_select.json", "w"), indent=1)
