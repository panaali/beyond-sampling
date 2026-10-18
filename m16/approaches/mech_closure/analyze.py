"""T338: floors (RMS over models of b_m, raw and noise-corrected), mean, slope; nodes; paired bootstrap; verdict."""
import json
import numpy as np

R = json.load(open("results/mc_all.json")); names = sorted(R)
keys = list(R[names[0]]["b"]); tr = np.array([R[n]["truth"] for n in names]) * 100
B = np.array([[R[n]["b"][k] for k in keys] for n in names]) * 100
SE = np.array([[R[n]["se"][k] for k in keys] for n in names]) * 100
out = {"n": len(names), "variants": {}}
for c, k in enumerate(keys):
    b = B[:, c]
    out["variants"][k] = {"rms": float(np.sqrt(np.mean(b ** 2))),
                          "rms_nc": float(np.sqrt(max(np.mean(b ** 2) - np.mean(SE[:, c] ** 2), 0))),
                          "mean": float(b.mean()), "slope": float(np.polyfit(tr, b, 1)[0]), "max_abs": float(np.abs(b).max())}
for v in ("mech", "late", "adj", "mech_nobudget"):
    nodes = np.array([max(R[n]["nodes"][r][v][ln] for r in ("nb", "b") for ln in R[n]["nodes"][r][v]) for n in names])
    out["variants"].setdefault(f"{v}_oracle", {})["max_nodes_line"] = {"max": int(nodes.max()), "mean": float(nodes.mean())}
    out["variants"][f"{v}_oracle"]["n_alloc_mean"] = float(np.mean([R[n]["n_alloc"]["nb"][v] + R[n]["n_alloc"]["b"][v] for n in names]))
    out["variants"][f"{v}_oracle"]["pos_hist"] = np.sum([np.array(R[n]["alloc_pos_hist"]["nb"][v]) + np.array(R[n]["alloc_pos_hist"]["b"][v]) for n in names], 0).tolist()
rng = np.random.default_rng(338); ci = {}
col = {k: i for i, k in enumerate(keys)}
pairs = [("mech_oracle", "late_oracle"), ("mech_oracle", "c_oracle"), ("mech_oracle", "adj_oracle")]
bs = {p: [] for p in pairs}
for _ in range(10000):
    i = rng.integers(0, len(names), len(names))
    rms = lambda k: np.sqrt(np.mean(B[i, col[k]] ** 2))
    for a, b in pairs:
        bs[(a, b)].append(rms(a) - rms(b))
out["bootstrap"] = {f"{a} - {b}": [float(np.sqrt(np.mean(B[:, col[a]] ** 2)) - np.sqrt(np.mean(B[:, col[b]] ** 2))),
                                    np.percentile(bs[(a, b)], [2.5, 97.5]).tolist()] for a, b in pairs}
m = out["variants"]["mech_oracle"]
go = m["rms"] <= .40 and m["max_nodes_line"]["max"] <= 3660
mech = out["bootstrap"]["mech_oracle - late_oracle"][1][1] < 0
out["verdict"] = {"go": bool(go), "mechanism_beats_recency": bool(mech)}
json.dump(out, open("results/summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
