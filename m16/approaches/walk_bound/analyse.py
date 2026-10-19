"""T449 analysis: actual error vs the candidate bounds (a) Lipschitz x W1 floor, (b) HK variation growth,
(c) oracle cell-enclosure width; -> analysis.json, printed table (analysis.log)."""
import glob
import json
import os

import numpy as np
from scipy.special import gamma

import lib

OUT = os.path.join(lib.HERE, "out")
rows = lambda p: [json.loads(l) for l in open(os.path.join(OUT, p + ".jsonl"))]
BUD = (64_000, 1_000_000, 4_000_000)
T429_RMS = {64_000: 1.02e-3, 1_000_000: 2.02e-4, 4_000_000: 9.0e-5}     # DEV3, 100 models (T429 README)
D = 9                                                                      # within-configuration dimension
VD = np.pi ** (D / 2) / gamma(D / 2 + 1)
res = {}

# 1. actual error
err = rows("err"); _, tr = lib.dev3_models(0)
res["err"] = {}
for B in BUD:
    e_ref = np.array([r[f"est_{B}"] - r["ref"] for r in err])
    e_tr = np.array([r[f"est_{B}"] - tr["truths"][r["model"]]["acc"] for r in err if r["model"] in tr["truths"]])
    res["err"][B] = dict(rms_vs_ref=float(np.sqrt((e_ref ** 2).mean())), max_vs_ref=float(np.abs(e_ref).max()),
                         rms_vs_truth_dev3=float(np.sqrt((e_tr ** 2).mean())), t429_rms=T429_RMS[B],
                         per_model={r["model"]: float(x) for r, x in zip(err, e_ref)})
ref_tr = np.array([r["ref"] - tr["truths"][r["model"]]["acc"] for r in err if r["model"] in tr["truths"]])
res["ref_vs_truth_dev3"] = [float(x) for x in ref_tr]

# 2. modulus of continuity
mod = rows("mod")
res["mod"] = {}
for j in range(5):
    rr = [r for r in mod if r["stratum"] == j]
    sl = np.concatenate([np.load(os.path.join(OUT, f"slope_{r['model']}_{j}.npy"))[:, 0] for r in rr])
    x = np.sort(sl)[::-1]; n = len(x); kk = max(10, n // 100)
    hill = 1 / np.mean(np.log(x[:kk] / x[kk]))                             # Hill tail index of the FD slope
    med = {dl: float(np.median([r["d"][dl]["abs"][0] for r in rr])) for dl in rr[0]["d"]}
    q99 = {dl: float(np.median([r["d"][dl]["abs"][2] for r in rr])) for dl in rr[0]["d"]}
    mx = {dl: float(max(r["d"][dl]["abs"][3] for r in rr)) for dl in rr[0]["d"]}
    fit = lambda q: float(np.polyfit(np.log10(list(map(float, q))), np.log10(list(q.values())), 1)[0])
    res["mod"][j] = dict(M=rr[0]["M"], L_lb=float(x[0]), slope_mean=float(sl.mean()), slope_med=float(np.median(sl)),
                         hill_tail=float(hill), median_abs=med, q99_abs=q99, max_abs=mx,
                         hoelder_median=fit(med), hoelder_q99=fit(q99), hoelder_max=fit(mx),
                         frac_g_pos=float(np.mean([r["frac_pos"] for r in rr])),
                         cost_per_line=float(np.mean([r["cost_per_line"] for r in rr])),
                         adapt_gain=float(sl.mean() / np.mean(sl ** (D / (D + 1))) ** ((D + 1) / D)))

# 3. (a) Lipschitz x W1 floor: lines per stratum <= B (each line costs >= 1 FPE), split over configurations
res["a"] = {}
ncfg = [len(lib.C.configs(10, k)) for k, _ in lib.STRATA]
for B in BUD:
    w1 = [D / (D + 1) * (B / ncfg[j] * VD) ** (-1 / D) for j in range(5)]
    b_sup = sum(res["mod"][j]["M"] * res["mod"][j]["L_lb"] * w1[j] for j in range(5))
    b_mean = sum(res["mod"][j]["M"] * res["mod"][j]["slope_mean"] * w1[j] for j in range(5))
    res["a"][B] = dict(W1_floor=w1, bound_floor=b_sup, ratio=b_sup / res["err"][B]["rms_vs_ref"],
                       mean_slope_heuristic=b_mean, ratio_heuristic=b_mean / res["err"][B]["rms_vs_ref"])

# 3a'. weight-derived switch-point sensitivity rho = |c_perp| / |c . d|
sens = sorted(glob.glob(os.path.join(OUT, "sens_*.npz")))
if sens:
    res["sens"] = {}
    for j in range(5):
        rho = np.concatenate([np.load(f)[f"rho_end_{j}"] for f in sens])
        kw = np.concatenate([np.load(f)[f"k_end_{j}"] for f in sens])
        rg = np.concatenate([np.load(f)[f"rho_gate_{j}"] for f in sens])
        chk = max(float(np.load(f)[f"check_{j}"]) for f in sens)
        x = np.sort(rho)[::-1]; kk = max(10, len(x) // 50)
        xg = np.sort(rg)[::-1]; kg = max(10, len(xg) // 50)
        res["sens"][j] = dict(n_end=len(rho), rho_end_q=[float(np.quantile(rho, p)) for p in (.5, .9, .99, 1)],
                              krho_max=float((kw * rho).max()), hill_end=float(1 / np.mean(np.log(x[:kk] / x[kk]))),
                              rho_gate_q=[float(np.quantile(rg, p)) for p in (.5, .9, .99, 1)],
                              hill_gate=float(1 / np.mean(np.log(xg[:kg] / xg[kg]))), walk_check=chk)

# 4. (b) Vitali variation growth
vit = rows("vitali")
res["b"] = []
for r in vit:
    ns = np.array(list(map(int, r["V3"]))); v3 = np.array([v[0] for v in r["V3"].values()])
    v2 = np.array([v[1] for v in r["V3"].values()])
    res["b"].append(dict(model=r["model"], stratum=r["stratum"], anchor=r["anchor"], box=r["box"],
                         V3=[float(x) for x in v3], exp3_last=float(np.polyfit(np.log(ns[-3:]), np.log(v3[-3:]), 1)[0]),
                         exp2_last=float(np.polyfit(np.log(ns[-3:]), np.log(v2[-3:]), 1)[0])))

# 5. (c) oracle enclosure width (sum over strata, mean over models), half-width compared with the error
orc = rows("oracle")
ks = list(orc[0]["k"])
res["c"] = {}
for k in ks:
    per_model = {}
    for r in orc:
        per_model.setdefault(r["model"], [0, 0, 0, 0])
        v = r["k"][k]
        per_model[r["model"]][0] += v["width"]; per_model[r["model"]][1] += v["lo"]
        per_model[r["model"]][2] += v["ncells"]; per_model[r["model"]][3] += v["frac_zero"] / 5
    W = np.array([v[0] for v in per_model.values()]); Wlo = np.array([v[1] for v in per_model.values()])
    res["c"][k] = dict(ncells=float(np.mean([v[2] for v in per_model.values()])), width_mean=float(W.mean()),
                       width_min=float(W.min()), width_lo_min=float(Wlo.min()),
                       halfwidth_ratio_1e6=float(W.min() / 2 / res["err"][1_000_000]["rms_vs_ref"]),
                       frac_zero_range=float(np.mean([v[3] for v in per_model.values()])))
nc = np.array([res["c"][k]["ncells"] for k in ks[-3:]]); wm = np.array([res["c"][k]["width_mean"] for k in ks[-3:]])
beta = float(np.polyfit(np.log(nc), np.log(wm), 1)[0])
res["c_fit"] = dict(beta=beta)
for B in BUD:
    tgt = 2 * 10 * res["err"][B]["rms_vs_ref"]                             # width for a 10x half-width
    res["c_fit"][B] = dict(target_width=tgt, cells_needed=float(nc[-1] * (tgt / wm[-1]) ** (1 / beta)),
                           cells_needed_d9=float(nc[-1] * (tgt / wm[-1]) ** (-D)))

json.dump(res, open(os.path.join(lib.HERE, "analysis.json"), "w"), indent=1, default=str)

# printed summary
print("ACTUAL ERROR (10 models: zoo seeds 0-4, dev3_s500-504), vs ref rule 27182 at 1.6e7 FPE")
for B in BUD:
    e = res["err"][B]
    print(f"  B={B:>8}: RMS {e['rms_vs_ref']:.2e} (max {e['max_vs_ref']:.2e}); DEV3 5 vs 1e8 truth {e['rms_vs_truth_dev3']:.2e};"
          f" T429 DEV3-100 {e['t429_rms']:.1e}")
print("  ref - truth (dev3):", " ".join(f"{x:+.1e}" for x in res["ref_vs_truth_dev3"]))
print("MODULUS (pooled over 10 models x 2048 nodes): stratum M | L_lb max slope | mean | median | Hill tail | Hoelder med/q99/max | adapt gain")
for j, m in res["mod"].items():
    print(f"  {j} M={m['M']:.3g} L_lb={m['L_lb']:.3g} mean={m['slope_mean']:.3g} med={m['slope_med']:.3g}"
          f" hill={m['hill_tail']:.2f} hoelder={m['hoelder_median']:.2f}/{m['hoelder_q99']:.2f}/{m['hoelder_max']:.2f}"
          f" gain={m['adapt_gain']:.2f} g>0 frac={m['frac_g_pos']:.3f} FPE/line={m['cost_per_line']:.1f}")
print("(a) LIPSCHITZ x W1 FLOOR (rigorous lower bound on this bound's best case)")
for B in BUD:
    a = res["a"][B]
    print(f"  B={B:>8}: bound >= {a['bound_floor']:.3g} ({a['ratio']:.2e}x error); with mean slope (not a bound)"
          f" {a['mean_slope_heuristic']:.3g} ({a['ratio_heuristic']:.0f}x)")
if sens:
    print("(a') SWITCH-POINT SENSITIVITY rho = |c_perp|/|c.d|: endpoints q50/q90/q99/max | Hill | gates q50/q99/max | Hill | walk check")
    for j, s in res["sens"].items():
        print(f"  {j} n={s['n_end']} end {'/'.join(f'{x:.2g}' for x in s['rho_end_q'])} hill={s['hill_end']:.2f}"
              f" | gate {s['rho_gate_q'][0]:.2g}/{s['rho_gate_q'][2]:.2g}/{s['rho_gate_q'][3]:.2g} hill={s['hill_gate']:.2f}"
              f" | max k*rho={s['krho_max']:.3g} | check {s['walk_check']:.1e}")
print("(b) VITALI 3-D VARIATION growth exponent (V3 ~ n^e over n=16..64; HK finite needs e -> 0)")
for r in res["b"]:
    print(f"  {r['model']:>10} s{r['stratum']} a{r['anchor']} box{r['box']}: V3 {r['V3'][0]:.2g}->{r['V3'][-1]:.3g}"
          f" e3={r['exp3_last']:.2f} (2-D e2={r['exp2_last']:.2f})")
print("(c) ORACLE ENCLOSURE WIDTH (lower bound for any enclosure), sum over strata")
for k in ks:
    c = res["c"][k]
    print(f"  cells {c['ncells']:.1e}: width mean {c['width_mean']:.3g} min-model {c['width_min']:.3g}"
          f" (CI-lo {c['width_lo_min']:.3g}); half-width / error(1e6) >= {c['halfwidth_ratio_1e6']:.0f}x; zero-range cells {c['frac_zero_range']:.3f}")
print(f"  fit width ~ cells^{beta:.3f} (last 3 levels)")
for B in BUD:
    f = res["c_fit"][B]
    print(f"  B={B:>8}: 10x needs width {f['target_width']:.1e}: cells ~{f['cells_needed']:.1e} (fit) / {f['cells_needed_d9']:.1e} (N^-1/9)")
