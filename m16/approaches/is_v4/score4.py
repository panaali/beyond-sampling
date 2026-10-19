"""T456 scoring (pre-registered line in README): needs est_<I>.json and truth_<I>.json for I = 0..14."""
import json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "panel.json"))); rows = []
for i, pt in enumerate(P):
    e = json.load(open(os.path.join(HERE, f"est_{i}.json"))); t = json.load(open(os.path.join(HERE, f"truth_{i}.json")))
    tr, p = t["truth"], e["p"]; z = (p - tr) / math.sqrt(e["se"] ** 2 + t["se"] ** 2) if (e["se"] or t["se"]) else float("inf")
    ok_truth = t["hits"] >= 100
    rows.append(dict(i=i, model=pt["model"], t0=pt["t0"], truth=tr, se_truth=t["se"], truth_hits=t["hits"], est=p, se=e["se"],
                     ratio=p / tr if tr > 0 else float("nan"), z=z, flag=e["flag"], gate_maxw=e["gate_maxw"],
                     gate_capshare=e["gate_capshare"], n_capped=e["n_capped"], raw_ok=ok_truth and abs(z) <= 2,
                     honest=ok_truth and (abs(z) <= 2 or e["flag"]), within2=0.5 <= p / tr <= 2 if tr > 0 else False,
                     hits=e["hits"], C=e["C"], C_draws=e["C_draws"], F=(p - tr) ** 2 * e["C"] / (tr * (1 - tr)),
                     F_draws=(p - tr) ** 2 * e["C_draws"] / (tr * (1 - tr)), F_pref=e["Z"], new_desc=e["n_new_descents"],
                     tri_before=len(e["triples_before"]), tri_after=len(e["triples_after"])))
e2 = np.mean([(x["est"] - x["truth"]) ** 2 for x in rows]); pq = np.mean([x["truth"] * (1 - x["truth"]) for x in rows])
C = np.mean([x["C"] for x in rows]); Cd = np.mean([x["C_draws"] for x in rows])
S = dict(n=len(rows), honest=sum(x["honest"] for x in rows), raw_z_le2=sum(x["raw_ok"] for x in rows),
         flags=sum(x["flag"] for x in rows), line_pass=sum(x["honest"] for x in rows) >= 14, within2=sum(x["within2"] for x in rows),
         F_panel=e2 * max(C, 1e3) / pq, F_panel_draws=e2 * max(Cd, 1e3) / pq, F_le1=sum(x["F"] <= 1 for x in rows),
         beats_pref=sum(x["F"] < x["F_pref"] for x in rows), beats_pref_draws=sum(x["F_draws"] < 0.19 + x["F_pref"] for x in rows),
         C_mean=C, C_max=max(x["C"] for x in rows), median_ratio=float(np.median([x["ratio"] for x in rows])))
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, "score4.json"), "w"), indent=1, default=float)
print("| # | model | t0 | truth (hits) | IS est +- se | ratio | z | flag (maxw, capshare, n_capped) | new desc. | triples before/after | C | F | F_pref |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in rows:
    print(f"| {x['i']} | {x['model']} | {x['t0']} | {x['truth']:.3e} ({x['truth_hits']}) | {x['est']:.3e} +- {x['se']:.1e} | {x['ratio']:.2f} | "
          f"{x['z']:+.1f} | {'FLAG' if x['flag'] else '-'} ({x['gate_maxw']:.3f}, {x['gate_capshare']:.2f}, {x['n_capped']}) | {x['new_desc']} | "
          f"{x['tri_before']}/{x['tri_after']} | {x['C']:.2e} | {x['F']:.2g} | {x['F_pref']:.2g} |")
print(S)
