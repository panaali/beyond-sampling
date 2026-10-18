"""T356 (DIAGNOSTIC SAMPLING): closure c's oracle floors on DEV3, i.e. E1 (node v4, budget.closure_c) and E0 (true v4,
T304's closure c), paired, 2e5 exact-law inputs per model (seed 356000 + index), vs DEV3 1e8 truths."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.argv = sys.argv[:1]
import budget as B  # noqa: E402
D = os.path.join(HERE, "../../../../data/oracle_floor"); z = np.load(os.path.join(D, "dev3_weights.npz"))
tr = json.load(open(os.path.join(D, "dev3_truths.json")))["truths"]; rows = {}
for k, nm in enumerate(z["names"]):
    m = tuple(z[a][k].astype(np.float64) for a in ("Wi", "Wh", "Wo")); x = np.random.default_rng(356000 + k).standard_normal((B.N, 10)); P = B.Point(x)
    v4 = -np.sort(-x, 1)[:, 3]
    rows[str(nm)] = dict(truth=tr[str(nm)]["acc"], E1=float(B.closure_c(m, P).mean()), E0=float(B.verdict(m, P, P.v2 + P.g12, P.v2, P.v2 - P.g23, v4, B.mu(v4)).mean()))
p = np.array([r["truth"] for r in rows.values()]); pq = float((p * (1 - p)).mean()); S = {"mean_pq": pq}
for k in ("E1", "E0"):
    e = 100 * (np.array([r[k] for r in rows.values()]) - p); S[k] = dict(bias=float(e.mean()), sd=float(e.std(ddof=1)), rms=float(np.sqrt((e ** 2).mean())))
json.dump(dict(per_model=rows, summary=S), open(os.path.join(HERE, "floors_dev3.json"), "w"), indent=1); print(S)
