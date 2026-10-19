"""Gain = mean p(1-p)/cost / MSE over rule seeds (MSE incl. bias, minus truth se^2); bootstrap 95% CI over rules.
Truths: T=5 from truths.json (48M samples, own stream); T=4 from ht_seeds/scan_seeds.json (16-23M samples)."""
import json, sys
import numpy as np
T = int(sys.argv[1])
raw = json.load(open(f"mse_T{T}.json"))
if T == 5:
    tr = {s: (d["acc"], d["se"]) for k, d in json.load(open("truths.json")).items() for s in [int(k.split(":")[2])] if k.startswith(f"16:{T}:")}
else:
    ht = json.load(open("../ht_seeds/scan_seeds.json"))
    tr = {s: (ht[f"16:{T}:{s}"]["truth"]["acc"], ht[f"16:{T}:{s}"]["truth"]["se"]) for s in range(5)}
rows = {}
print(f"T={T}: gain [95% CI]   (bias z = mean err / its s.e.)")
for key, d in raw.items():
    s, v, b = key.split("|"); s = int(s)
    if s not in tr: continue
    p, se = tr[s]; e = np.array(d["est"]); c = np.array(d["cost"])
    base = np.mean(p * (1 - p) / c)
    g = lambda ii: base / max(np.mean((e[ii] - p) ** 2) - se ** 2, 1e-30)
    rng = np.random.default_rng(0); bs = [g(rng.integers(0, len(e), len(e))) for _ in range(2000)]
    z = (e.mean() - p) / np.sqrt(e.var() / len(e) + se ** 2)
    rows[key] = dict(gain=g(np.arange(len(e))), ci=list(np.percentile(bs, [2.5, 97.5])), z=z, cost=c.mean())
for v in ("plain", "radial", "strata", "frozen"):
    for b in sorted({k.split("|")[2] for k in rows}, key=int):
        cells = [rows.get(f"{s}|{v}|{b}") for s in range(5)]
        print(f"{v:7s} {b:>6s}: " + " | ".join(f"{r['gain']:6.1f} [{r['ci'][0]:5.1f},{r['ci'][1]:6.1f}] z{r['z']:+.1f}" if r else "   -   " for r in cells))
json.dump(rows, open(f"mse_T{T}_gains.json", "w"), indent=1)
