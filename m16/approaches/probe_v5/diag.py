"""Where does probe v2 go wrong? True error decomposition vs v2-style probe predictions, per model."""
import json
import sys
import numpy as np
sys.path.insert(0, "/home/user/lb3/m16")
import leaderboard as L
from probes import TRIPLES, canonical, bisect, cdf, REL23, REL12, forward

ms = L.models()
out = {}
for name in sys.argv[1:]:
    m = ms[name]
    x = np.random.default_rng(7).standard_normal((1_000_000, 10))
    idx = np.argsort(-x, 1)
    pred = forward(m, x)
    rank = np.argsort(idx, 1)[np.arange(len(x)), pred]           # 0 = max
    blind = (x > 0).sum(1) <= 1
    err = rank != 1
    true = {"blind": float((err & blind).mean()), "pick1": float((err & ~blind & (rank == 0)).mean()),
            "pick3": float((err & ~blind & (rank == 2)).mean()), "pick4+": float((err & ~blind & (rank >= 3)).mean()),
            "acc": float(1 - err.mean()), "P_err_blind": float(err[blind].mean())}
    X = canonical(TRIPLES, 0)
    a, b, c = TRIPLES[:, 0], TRIPLES[:, 1], TRIPLES[:, 2]
    r = np.arange(720)

    def set23(Xs, g):
        Y = Xs.copy(); Y[r, c] = Y[r, b] * (1 - g); return Y

    def set12(Xs, g):
        Y = Xs.copy(); Y[r, a] = Y[r, b] / (1 - g); return Y
    t23, e1 = bisect(m, X, set23, c, 1e-6, 1.0, 12)
    t12, e2 = bisect(m, X, set12, a, 1e-6, 0.95, 12)
    probe = {"blind": 11 / 1024 * 8 / 9, "pick3": float((1 - 11 / 1024) * cdf(REL23, t23).mean()),
             "pick1": float((1 - 11 / 1024) * cdf(REL12, t12).mean()),
             "n_t23_at_hi": int((t23 >= 1.0).sum()), "n_t12_at_hi": int((t12 >= 0.95).sum()), "evals": e1 + e2}
    probe["acc"] = 1 - probe["blind"] - probe["pick3"] - probe["pick1"]
    out[name] = {"true": true, "probe_v2": probe}
    print(name, "TRUE", {k: round(v, 4) for k, v in true.items()}, "\n   PROBE", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in probe.items()}, flush=True)
json.dump(out, open("results/diag.json", "w"), indent=1)
