"""T272 part (1) analysis: kink density vs T and vs E33's per-model Sobol gains / tail shares (H=16, seeds 0-4).
usage: python analyse_kinks.py > kinks_analysis.out"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
K = json.load(open(os.path.join(HERE, "results", "kinks_2000.json")))
E33 = {(r["H"], r["T"], r["seed"]): r for r in json.load(open(os.path.join(HERE, "..", "ht_seeds", "table.json")))}


def dens(r, key="density"):
    Z = np.array(r["Z"]); w = Z / Z.sum()
    if key == "density":
        return float(w[0] * r["23"]["density"] + w[1] * r["12"]["density"])
    if key == "jumps":
        return float(w[0] * r["23"]["jumps"] / r["23"]["length"] + w[1] * r["12"]["jumps"] / r["12"]["length"])
    if key == "kinkF":                                  # kinks weighted by the local band value, per unit length
        return float(w[0] * r["23"]["kink_F"] / r["23"]["length"] + w[1] * r["12"]["kink_F"] / r["12"]["length"])


rows = []
for cell, r in K.items():
    if cell == "orig":
        continue
    H, T, s = (int(v) for v in cell.split(":"))
    e = E33[(H, T, s)]
    rows.append(dict(T=T, seed=s, dens=dens(r), jumps=dens(r, "jumps"), kinkF=dens(r, "kinkF"),
                     gain=e["sobol"]["best"], tail=e["sobol"]["tail"], acc=e["acc"]))
print("T  seed  kinks/unit  jumps/unit  kinkF/unit  gain   tail")
for x in sorted(rows, key=lambda x: (x["T"], x["seed"])):
    print(f"{x['T']:2d} {x['seed']:4d} {x['dens']:10.1f} {x['jumps']:10.2f} {x['kinkF']:10.4f} {x['gain']:7.1f} {x['tail']:.3f}")
print("\ngeometric means by T:")
Ts = sorted({x["T"] for x in rows}); gm = {}
for T in Ts:
    xs = [x for x in rows if x["T"] == T]
    gm[T] = {k: float(np.exp(np.mean(np.log([max(x[k], 1e-12) for x in xs])))) for k in ("dens", "jumps", "kinkF", "gain", "tail")}
    print(T, {k: round(v, 3) for k, v in gm[T].items()})
print("target M_16,10:", round(dens(K["orig"]), 1), "kinks/unit;", round(dens(K["orig"], "jumps"), 2), "jumps/unit")
ratio = gm[10]["dens"] / gm[3]["dens"]
print(f"\n(a) kink density T=10 / T=3 = {ratio:.2f} (support needs >= 10)")
# within-T (T = 4, 5, 6, 8): centre log values per T, Spearman over the pooled 20 models
sel = [x for x in rows if x["T"] in (4, 5, 6, 8)]
def centred(k):
    v = np.array([np.log(max(x[k], 1e-12)) for x in sel]); Tv = np.array([x["T"] for x in sel])
    for T in set(Tv):
        v[Tv == T] -= v[Tv == T].mean()
    return v
g = centred("gain")
res = {}
for k in ("dens", "jumps", "kinkF", "tail"):
    rho, p = spearmanr(centred(k), g); res[k] = (float(rho), float(p))
    print(f"(b) within-T Spearman rho(log {k}, log gain) = {rho:+.2f} (p = {p:.2f}, n = {len(sel)})")
rho_all = spearmanr(np.log([x["dens"] for x in rows]), np.log([x["gain"] for x in rows]))[0]
rho_all_tail = spearmanr(np.log([x["tail"] for x in rows]), np.log([x["gain"] for x in rows]))[0]
print(f"across all T: rho(kinks, gain) = {rho_all:+.2f}; rho(tail, gain) = {rho_all_tail:+.2f}")
json.dump(dict(rows=rows, gm={str(k): v for k, v in gm.items()}, ratio_10_3=ratio, within_T=res,
               rho_all=[rho_all, rho_all_tail]), open(os.path.join(HERE, "results", "kinks_analysis.json"), "w"), indent=1)
