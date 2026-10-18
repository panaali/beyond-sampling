"""T341: floors, slopes, parity split, paired bootstrap vs the reverse spread; verdict."""
import json
import numpy as np

R = json.load(open("results/os_all.json")); names = sorted(R); V = list(R[names[0]]["b"])
tr = np.array([R[n]["truth"] for n in names]) * 100
B = np.array([[R[n]["b"][v] for v in V] for n in names]) * 100
out = {"n": len(names), "floors": {v: {"rms": float(np.sqrt(np.mean(B[:, j] ** 2))), "mean": float(B[:, j].mean()),
                                       "slope": float(np.polyfit(tr, B[:, j], 1)[0])} for j, v in enumerate(V)}}
P = {k: np.array([R[n]["spread_by_lastparity"][k] for n in names]) * 100 for k in R[names[0]]["spread_by_lastparity"]}
out["spread_by_last_free_parity_mean"] = {k: float(v.mean()) for k, v in P.items()}
rng = np.random.default_rng(341); j0 = V.index("s-"); bs = {v: [] for v in V}
for _ in range(10000):
    i = rng.integers(0, len(names), len(names)); r0 = np.sqrt(np.mean(B[i, j0] ** 2))
    for j, v in enumerate(V):
        bs[v].append(np.sqrt(np.mean(B[i, j] ** 2)) - r0)
out["vs_s-"] = {v: [out["floors"][v]["rms"] - out["floors"]["s-"]["rms"], np.percentile(bs[v], [2.5, 97.5]).tolist()]
                for v in V if v != "s-"}
out["go"] = bool(any(out["vs_s-"][v][1][1] < 0 for v in ("ii", "iv")))
json.dump(out, open("results/summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
