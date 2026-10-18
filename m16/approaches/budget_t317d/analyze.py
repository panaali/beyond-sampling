"""T324 summary: per variant bias / RMS / MSE vs truth (1e8), one-at-a-time and cumulative shares, checks."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results.json"))); names = list(r)
tru = np.array([r[n]["truth"] for n in names])
def stats(vals):
    e = 100 * (np.asarray(vals) - tru); return dict(bias=float(e.mean()), rms=float(np.sqrt((e ** 2).mean())), mse=float((e ** 2).mean()), sd=float(e.std(ddof=1)))
S = {"n": len(names)}
S["est_d"] = stats([r[n]["est_d"] for n in names])
keys = [k for k in r[names[0]] if isinstance(r[names[0]][k], dict)]
for k in keys:
    S[k] = stats([r[n][k]["mean"] for n in names])
S["truth_mc_check"] = stats([r[n]["mc_ok"] for n in names])
# consistency of the pointwise form with cont_d.est
z = np.array([(r[n]["V1"]["mean"] - r[n]["est_d"]) / r[n]["V1"]["se"] for n in names])
d = np.array([100 * (r[n]["V1"]["mean"] - r[n]["est_d"]) for n in names]); se = np.array([100 * r[n]["V1"]["se"] for n in names])
S["check_pointwise_vs_est"] = dict(max_abs_z=float(np.abs(z).max()), rms_diff_pp=float(np.sqrt((d ** 2).mean())), mean_se_pp=float(se.mean()),
                                    passed=bool(np.abs(z).max() <= 4 and np.sqrt((d ** 2).mean()) <= 2 * se.mean()))
S["identity_E1_max"] = max(r[n]["identity_E1"] for n in names)
V1, E1 = S["V1"], S["E1_direct"]
one = {a: dict(d_bias=S["off_" + a]["bias"] - V1["bias"], d_rms=S["off_" + a]["rms"] - V1["rms"], d_mse=S["off_" + a]["mse"] - V1["mse"],
               share_bias=(V1["bias"] - S["off_" + a]["bias"]) / V1["bias"], share_mse=(V1["mse"] - S["off_" + a]["mse"]) / V1["mse"])
       for a in ("tmax", "blind", "mulin", "product", "transverse")}
path = ["V1"] + [k for k in keys if k.startswith("path")]
cum = []
for a, b in zip(path[:-1], path[1:]):
    cum.append(dict(step=b, bias=S[b]["bias"], rms=S[b]["rms"], d_bias=S[b]["bias"] - S[a]["bias"], d_mse=S[b]["mse"] - S[a]["mse"],
                    share_bias=(S[a]["bias"] - S[b]["bias"]) / V1["bias"], share_mse=(S[a]["mse"] - S[b]["mse"]) / V1["mse"]))
cum.append(dict(step="truth (closure c off)", bias=0.0, rms=0.0, share_bias=E1["bias"] / V1["bias"], share_mse=E1["mse"] / V1["mse"]))
S["one_at_a_time"] = one; S["cumulative"] = cum
S["redteam_claim"] = dict(mse_share_from_approx=(V1["mse"] - E1["mse"]) / V1["mse"], bias_V1=V1["bias"], bias_E1=E1["bias"],
                          supported=bool((V1["mse"] - E1["mse"]) >= .6 * V1["mse"] and abs(E1["bias"]) < abs(V1["bias"]) / 2))
json.dump(S, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
for k in ["est_d", "V1"] + ["off_" + a for a in one] + path[1:] + ["E1_direct", "truth_mc_check"]:
    print("%-22s bias %+.3f  rms %.3f  sd %.3f" % (k, S[k]["bias"], S[k]["rms"], S[k]["sd"]))
print(json.dumps({k: S[k] for k in ("check_pointwise_vs_est", "identity_E1_max", "one_at_a_time", "cumulative", "redteam_claim")}, indent=1))
