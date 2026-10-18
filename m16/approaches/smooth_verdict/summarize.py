"""T319 summary: RMS / mean / slope / region split per closure x verdict, and the pre-registered go/stop."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results.json")))
tr = json.load(open(os.path.join(HERE, "../../../data/oracle_floor/dev2_truths.json")))["truths"]
names = list(r); acc = 100 * np.array([tr[n]["acc"] for n in names])
S = {"n_models": len(names)}
for c in ("R0", "top4"):
    for v in ("hard", "smooth", "smoothall"):
        k = f"{c}_{v}"; b = np.array([r[n][k]["b"] for n in names]); se = np.array([r[n][k]["se"] for n in names])
        S[k] = dict(rms=float(np.sqrt(np.mean(b ** 2))), rms_nc=float(np.sqrt(max(np.mean(b ** 2 - se ** 2), 0))), mean=float(b.mean()),
                    sd=float(b.std(ddof=1)), slope=float(np.polyfit(acc, b, 1)[0]), mean_se=float(se.mean()),
                    region_mean={q: float(np.mean([r[n][k][q] for n in names])) for q in ("blind", "band23", "band12", "tail")},
                    region_rms={q: float(np.sqrt(np.mean([r[n][k][q] ** 2 for n in names]))) for q in ("blind", "band23", "band12", "tail")})
    h, s = S[f"{c}_hard"], S[f"{c}_smooth"]
    S[f"{c}_verdict"] = dict(rms_ratio=s["rms"] / h["rms"], go=bool(s["rms"] <= .7 * h["rms"] and abs(s["slope"]) <= abs(h["slope"]) + .01),
                            fable_mean_halved=bool(abs(s["mean"]) <= abs(h["mean"]) / 2), fable_stop_lt10=bool(s["rms"] > .9 * h["rms"]))
S["overall_go"] = S["R0_verdict"]["go"] or S["top4_verdict"]["go"]
json.dump(S, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
for k, v in S.items():
    print(k, v if not isinstance(v, dict) else {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
