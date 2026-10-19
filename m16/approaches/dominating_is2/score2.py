"""T453 scoring (run only after all 15 t403_<I>.json estimates are pushed): reads T403's truths from
claude/t403-rare-tie-check results/score.json (rows, same order), and gap_rates.json (P(g23 >= t0), 1e8 draws)."""
import json, math, os, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
T403 = json.loads(subprocess.check_output(["git", "show", "origin/claude/t403-rare-tie-check:m16/approaches/rare_tie_check/results/score.json"], cwd=HERE))["rows"]
G = json.load(open(os.path.join(HERE, "gap_rates.json")))
rows = []
for i, t in enumerate(T403):
    r = json.load(open(os.path.join(HERE, f"t403_{i}.json"))); assert r["model"] == t["model"] and r["t0"] == t["t0"]
    p, tr = r["p"], t["truth"]; e = p - tr; q = G[str(t["t0"])]
    rows.append(dict(i=i, model=t["model"], t0=t["t0"], stratum=t["stratum"], truth=tr, se_truth=t["se"], est=p, se=r["se"],
                     ratio=p / tr, within2=0.5 <= p / tr <= 2, z=e / math.sqrt(r["se"] ** 2 + t["se"] ** 2), hits=r["hits"],
                     C=r["C"], C_draws=r["C_draws"], F=e ** 2 * r["C"] / (tr * (1 - tr)), F_se=r["se"] ** 2 * r["C"] / (tr * (1 - tr)),
                     F_draws=e ** 2 * r["C_draws"] / (tr * (1 - tr)), F_pref=q, F_pref_draws=0.19 + q,
                     F_walk=t["F"], ratio_walk=t["ratio"], tie_truth=t["tie_truth"], n_points=r["n_points"]))
e2 = np.mean([(x["est"] - x["truth"]) ** 2 for x in rows]); pq = np.mean([x["truth"] * (1 - x["truth"]) for x in rows])
C = np.mean([x["C"] for x in rows]); Cd = np.mean([x["C_draws"] for x in rows])
summ = dict(n=len(rows), within2=sum(x["within2"] for x in rows), z_le2=sum(abs(x["z"]) <= 2 for x in rows),
            line_pass=sum(x["within2"] for x in rows) >= 12 and sum(abs(x["z"]) <= 2 for x in rows) >= 13,
            F_panel=e2 * max(C, 1e3) / pq, F_panel_draws=e2 * max(Cd, 1e3) / pq,
            F_point_le1=sum(x["F"] <= 1 for x in rows), F_point_le10=sum(x["F"] <= 10 for x in rows),
            beats_pref=sum(x["F"] < x["F_pref"] for x in rows), beats_pref_draws=sum(x["F_draws"] < x["F_pref_draws"] for x in rows),
            beats_walk=sum(x["F"] < x["F_walk"] for x in rows), median_ratio=float(np.median([x["ratio"] for x in rows])))
json.dump(dict(summary=summ, rows=rows), open(os.path.join(HERE, "score2.json"), "w"), indent=1, default=float)
print("| # | model | t0 | stratum | truth | IS est +- se | ratio | z | hits | C | F | F (draws) | F_pref | F_walk |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in rows:
    print(f"| {x['i']} | {x['model']} | {x['t0']} | {x['stratum'][0]} | {x['truth']:.3e} | {x['est']:.3e} +- {x['se']:.1e} | {x['ratio']:.2f} | "
          f"{x['z']:+.1f} | {x['hits']} | {x['C']:.2e} | {x['F']:.2g} | {x['F_draws']:.2g} | {x['F_pref']:.2g} | {x['F_walk']:.2g} |")
print(summ)
