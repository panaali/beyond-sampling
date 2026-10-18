"""Summary of T292: oracle floor b_m, observer F table, ANAT-80. Reads results/{oracle,anat,obs_*}/.
Truths: own 2e7-sample numpy MC (anat.py) unless ../../gpu_cost_law-style truths.json is given as argv[1]."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda d: {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "results", d, "*.json"))}
orc, ana = R("oracle"), R("anat")
names = sorted([n for n in orc if n != "target"], key=lambda s: (s.split("_")[0], int(s.split("_s")[-1])))
if len(sys.argv) > 1:
    tj = json.load(open(sys.argv[1]))["truths"]; truth = {n: tj[n]["acc"] for n in names}; tsrc = sys.argv[1]
else:
    truth = {n: ana[n]["acc"] for n in names}; tsrc = "own 2e7 MC (anat.py)"
p = np.array([truth[n] for n in names]); pq = np.mean(p * (1 - p))
out = {"truth_source": tsrc, "n_models": len(names), "acc_mean": p.mean(), "acc_sd": p.std(ddof=1), "acc_min": p.min(), "acc_max": p.max(), "mean_pq": pq}
slope = lambda y: float(np.polyfit(100 * p, y, 1)[0])
REG = ("blind", "band23", "band12", "tail")

# ---- 1. oracle floor
b = np.array([orc[n]["lin"]["bias_pp"] for n in names]); se = np.array([orc[n]["lin"]["se_pp"] for n in names])
parts = {r: np.array([orc[n]["lin"][f"bias_{r}_pp"] for n in names]) for r in REG}
o = dict(rms_pp=float(np.sqrt(np.mean(b ** 2))), rms_noise_corrected_pp=float(np.sqrt(max(np.mean(b ** 2 - se ** 2), 0))),
         mean_pp=float(b.mean()), sd_pp=float(b.std(ddof=1)), slope_pp_per_pp=slope(b), median_abs_pp=float(np.median(np.abs(b))),
         mean_se_pp=float(se.mean()), min_pp=float(b.min()), max_pp=float(b.max()),
         n_abs_le_040=int((np.abs(b) <= .4).sum()), n_abs_gt_080=int((np.abs(b) > .8).sum()),
         by_region={r: dict(mean_pp=float(parts[r].mean()), rms_pp=float(np.sqrt(np.mean(parts[r] ** 2))), slope=slope(parts[r])) for r in REG},
         target=orc["target"]["lin"])
G1 = o["rms_pp"] <= .40 and abs(o["mean_pp"]) <= .25 and abs(o["slope_pp_per_pp"]) <= .05
o["outcome"] = "G1" if G1 else ("DEAD" if o["rms_pp"] > .80 else "neither")
out["oracle"] = o

# ---- 2. observer
out["observer"] = {}
for cfg in ("111", "112", "222"):
    ob = R("obs_" + cfg)
    if not all(n in ob for n in names):
        continue
    e = np.array([ob[n]["est"] - truth[n] for n in names]) * 100
    C = np.mean([ob[n]["C_fpe"] for n in names]); C0 = np.mean([ob[n]["C_fpe_estpy"] for n in names])
    F = np.mean((e / 100) ** 2) * max(C, 1000) / pq
    out["observer"][cfg] = dict(rms_pp=float(np.sqrt(np.mean(e ** 2))), mean_pp=float(e.mean()), sd_pp=float(e.std(ddof=1)),
                                slope_pp_per_pp=slope(e), C_fpe=float(C), F=float(F), F_estpy_cost=float(np.mean((e / 100) ** 2) * max(C0, 1000) / pq),
                                sampling_rms_at_C_pp=float(100 * np.sqrt(pq / C)), corr_e_with_b=float(np.corrcoef(e, b)[0, 1]),
                                target_bias_pp=float(100 * (ob["target"]["est"] - ana["target"]["acc"])) if "target" in ob and "target" in ana else None)

# ---- 3. ANAT-80
if all(n in ana for n in names):
    E = {r: np.array([ana[n]["err_pp"][r] for n in names]) for r in REG}
    err = 100 - 100 * p
    tail2 = np.array([ana[n]["tail_pp"]["0.2"] for n in names]); tail1 = np.array([ana[n]["tail_pp"]["0.1"] for n in names])
    rk = np.array([np.array(ana[n]["wrong_rank_pp"]).sum(0) for n in names])     # (80,10) by rank of chosen input
    gross, gross_err, thin, x9 = [], [], [], []
    for n in names:
        z = np.load(os.path.join(HERE, "results/anat", n + ".npz"))
        gt, ge = z["gtot"][5:].sum(0), z["ger"][5:].sum(0)           # min gap >= .2, non-blind, per triple
        valid = gt >= 30; rate = np.where(valid, ge / np.maximum(gt, 1), 0)
        g = valid & (rate > .2); gross.append(int(g.sum())); thin.append(int(((gt > 0) & ~valid).sum()))
        gross_err.append(100 * ge[g].sum() / ana[n]["n"])
        x9.append(100 * z["er"].sum(2) / ana[n]["n"])                   # (4 regions, 3 x9 cases)
    gross, gross_err, x9 = np.array(gross), np.array(gross_err), np.array(x9)
    var = err.var()
    share = {r: float(np.cov(E[r], err, bias=True)[0, 1] / var) for r in REG}   # sums to 1: variance attribution
    a = dict(err_mean_pp={r: float(E[r].mean()) for r in REG}, err_sd_pp={r: float(E[r].std(ddof=1)) for r in REG},
             cross_model_variance_share=share, slope_on_acc={r: slope(E[r]) for r in REG},
             tail_ge_01_pp=dict(mean=float(tail1.mean()), sd=float(tail1.std(ddof=1)), share_of_var=float(np.cov(tail1, err, bias=True)[0, 1] / var)),
             tail_ge_02_pp=dict(mean=float(tail2.mean()), sd=float(tail2.std(ddof=1)), share_of_var=float(np.cov(tail2, err, bias=True)[0, 1] / var)),
             gross_triples=dict(mean=float(gross.mean()), median=float(np.median(gross)), max=int(gross.max()),
                                n_models_with_any=int((gross > 0).sum()), corr_with_err=float(np.corrcoef(gross, err)[0, 1]),
                                err_in_gross_pp_mean=float(gross_err.mean()), err_in_gross_share_of_var=float(np.cov(gross_err, err, bias=True)[0, 1] / var),
                                thin_triples_mean=float(np.mean(thin))),
             x9_case_err_pp_mean=dict(zip(("max", "2nd", "below"), x9.sum(1).mean(0).round(3).tolist())),
             x9_case_share_of_var=dict(zip(("max", "2nd", "below"), [float(np.cov(x9.sum(1)[:, c], err, bias=True)[0, 1] / var) for c in range(3)])),
             wrong_rank_pp_mean=rk.mean(0).round(3).tolist(),
             rank4plus_pp=dict(mean=float(rk[:, 3:].sum(1).mean()), share_of_var=float(np.cov(rk[:, 3:].sum(1), err, bias=True)[0, 1] / var)))
    # OLS of accuracy on the parts (standardised coefficients) and bias-part vs anatomy-part correlation
    X = np.c_[np.ones(len(names)), *[E[r] for r in REG]]
    a["ols_acc_on_parts_coef"] = dict(zip(("const",) + REG, np.linalg.lstsq(X, 100 * p, rcond=None)[0].round(3).tolist()))
    a["corr_lin_tail_bias_vs_tail_err"] = float(np.corrcoef(parts["tail"], E["tail"])[0, 1])
    a["lin_tail_bias_over_tail_err_mean"] = float(np.mean(parts["tail"] / E["tail"]))
    out["anat"] = a
    rows = [dict(name=n, acc=truth[n], b_pp=b[i], b_se=se[i], **{f"b_{r}": parts[r][i] for r in REG},
                 **{f"err_{r}": E[r][i] for r in REG}, tail02=tail2[i], gross=int(gross[i]),
                 **{f"obs{c}_e": 100 * (R("obs_" + c)[n]["est"] - truth[n]) for c in out["observer"]}) for i, n in enumerate(names)]
    json.dump(rows, open(os.path.join(HERE, "results/per_model.json"), "w"), indent=1)
json.dump(out, open(os.path.join(HERE, "results/summary.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
