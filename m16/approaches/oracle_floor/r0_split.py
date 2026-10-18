"""R0 oracle-floor stats by region from the per-skeleton rows (results/oracle[_lb3]/*.npy; cols v1 v2 v3 q R0 lin tri).
Stage A gates (T296 pre-registration). usage: python r0_split.py dev2|lb3"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); which = sys.argv[1]
D = os.path.join(HERE, "results", "oracle_lb3" if which == "lb3" else "oracle")
REG = ("blind", "band23", "band12", "tail")
rows = {}
for f in sorted(glob.glob(os.path.join(D, "*.npy"))):
    n = os.path.basename(f)[:-4]
    if n == "target":
        continue
    R = np.load(f).astype(np.float64); g12, g23 = R[:, 0] - R[:, 1], R[:, 1] - R[:, 2]
    reg = np.where(R[:, 1] <= 0, 0, np.where(np.minimum(g12, g23) >= .05, 3, np.where(g23 <= g12, 1, 2)))
    d = R[:, 4] - R[:, 3]
    rows[n] = dict(b=100 * d.mean(), se=100 * d.std() / np.sqrt(len(d)), **{r: 100 * d[reg == i].sum() / len(d) for i, r in enumerate(REG)},
                   **{"se_" + r: 100 * np.sqrt(((d * (reg == i)) ** 2).mean() - (d * (reg == i)).mean() ** 2) / np.sqrt(len(d)) for i, r in enumerate(REG)})
grp = lambda n: "orig" if n == "orig" else "seed" if n.startswith("seed") else n.rsplit("_", 1)[0] if n.startswith("s0") else n.split("_")[0]


def gates(names):
    b = np.array([rows[n]["b"] for n in names]); rms = float(np.sqrt(np.mean(b ** 2)))
    G = {}
    for n in names:
        G.setdefault(grp(n), []).append(rows[n]["b"])
    gm = {g: float(np.mean(v)) for g, v in G.items()}
    reg_mean = {r: float(np.mean([rows[n][r] for n in names])) for r in REG}
    reg_rms = {r: float(np.sqrt(np.mean([rows[n][r] ** 2 for n in names]))) for r in REG}
    se = np.array([rows[n]["se"] for n in names])
    ok = dict(i=rms <= .55, ii=all(abs(v) <= .30 for v in gm.values()), iii=all(abs(v) <= rms / 2 for v in reg_mean.values()))
    return dict(n=len(names), rms_pp=rms, rms_noise_corrected_pp=float(np.sqrt(max(np.mean(b ** 2 - se ** 2), 0))), mean_pp=float(b.mean()),
                group_mean_pp=gm, region_mean_pp=reg_mean, region_rms_pp=reg_rms, tests=ok, GO=all(ok.values()))


names = list(rows)
out = {"all": gates(names), "per_model": rows}
if which == "lb3":
    out["retrain_like_subset"] = gates([n for n in names if grp(n) in ("orig", "seed", "s0.0025", "s0.005")])
json.dump(out, open(os.path.join(HERE, f"results/r0_{which}.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_model"}, indent=1))
if which == "lb3":
    for n in names:
        r = rows[n]; print("%-10s b %+.2f +- %.2f | blind %+.3f b23 %+.2f b12 %+.2f tail %+.2f" % (n, r["b"], r["se"], *[r[q] for q in REG]))
