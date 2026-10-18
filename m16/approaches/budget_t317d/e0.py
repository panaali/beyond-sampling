"""T324 extra path step: closure c with the TRUE v4 value at the latest free position (T304 Stage A's closure c), on the
same paired inputs as budget.py (seed 324000 + index). E1 (K = 1 node v4) -> E0 (true v4) -> truth."""
import json, os, sys
import numpy as np
sys.argv = sys.argv[:1]
from budget import Point, verdict, mu, dev2, forward, N, T, HERE  # noqa: E402
res = json.load(open(os.path.join(HERE, "results.json"))); out = {}
for idx, (name, model) in enumerate(dev2()):
    rng = np.random.default_rng(324000 + idx); x = rng.standard_normal((N, T)); P = Point(x)
    ok = (forward(model, x).argmax(1) == P.tri[:, 1]).astype(float)
    v4 = -np.sort(-x, 1)[:, 3]
    v = verdict(model, P, P.v2 + P.g12, P.v2, P.v2 - P.g23, v4, mu(v4))
    out[name] = dict(mean=float(v.mean()), paired_bias_pp=float(100 * (v - ok).mean()), mc_ok=float(ok.mean()))
    assert abs(out[name]["mc_ok"] - res[name]["mc_ok"]) < 1e-12           # same inputs as budget.py
    print(idx, name, out[name], flush=True)
json.dump(out, open(os.path.join(HERE, "results_e0.json"), "w"), indent=1)
tru = np.array([res[n]["truth"] for n in out]); e = 100 * (np.array([out[n]["mean"] for n in out]) - tru)
print("E0 (closure c, true v4): bias %+.3f rms %.3f" % (e.mean(), np.sqrt((e ** 2).mean())))
