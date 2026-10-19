"""Apply the frozen T377 analysis (README) -> results/summary377.json."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
T24 = ROOT.parent
frag = json.load(open(ROOT / "fragility.json"))
audit = json.load(open(ROOT / "results" / "audit.json"))
a24 = json.load(open(T24 / "results" / "audit.json"))
FRAG_NAME = {8: "M8_s4(default)", 16: "M16_s2(default)", 32: "M32_s0(default)"}

# accounts: (model, H, gap, r2 list, code)
acc = []
for c, r in a24.items():
    s = json.load(open(T24 / "results" / f"score_{c}.json"))
    acc.append({"code": c, "model": FRAG_NAME[r["H"]], "H": r["H"], "gap": s["gap_recovered"], "budget": 80})
arm = []
for c, r in audit.items():
    s = json.load(open(ROOT / "results" / f"score_{c}.json"))
    rec = {"code": c, "model": r["model"], "H": r["H"], "gap": s["gap_recovered"], "budget": r["budget"],
           "r2": s["r2"], "n_r2_ge_0.8": int(sum(v >= .8 for v in s["r2"])), "model_acc": s["model_acc"],
           "replacement_acc": s["replacement_acc"], "one_neuron_mean_drop": s["model_acc"] - float(np.mean(s["acc_replacing_one_neuron"]))}
    (arm if r["budget"] != 80 else acc).append(rec)

models = defaultdict(list)
for a in acc:
    models[a["model"]].append(a["gap"])
tab = {m: {"H": frag[m]["H"], "F90": frag[m]["F90"], "F80": frag[m]["F80"], "S1": frag[m]["S1"],
           "g_bar": float(np.mean(g)), "gaps": g, "n_accounts": len(g)} for m, g in models.items()}


def loo(names, cols):
    y = np.array([tab[m]["g_bar"] for m in names])
    X = np.column_stack([np.ones(len(names))] + [np.array([f(tab[m]) for m in names]) for f in cols])
    err = []
    for k in range(len(names)):
        msk = np.arange(len(names)) != k
        beta = np.linalg.lstsq(X[msk], y[msk], rcond=None)[0]
        err.append(y[k] - X[k] @ beta)
    full = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.sqrt(np.mean(np.square(err)))), [float(b) for b in full]


size = lambda t: np.log2(t["H"])
fr = lambda t: t["F90"]
out = {"models": tab, "loo": {}}
for label, names in (("excl_M8", [m for m in tab if tab[m]["H"] != 8]), ("all8", list(tab))):
    ra, ba = loo(names, [size]); rb, bb = loo(names, [fr]); rc, bc = loo(names, [size, fr])
    v = "fragility dominates" if rb <= 0.7 * ra else "size dominates" if ra <= 0.7 * rb else "not separable"
    out["loo"][label] = {"n": len(names), "RMSE_size": ra, "RMSE_F": rb, "RMSE_both": rc, "coef_size": ba,
                         "coef_F": bb, "coef_both": bc, "verdict": v}
within = {}
for H in (16, 32):
    ms = [m for m in tab if tab[m]["H"] == H]
    Fs, gs = [tab[m]["F90"] for m in ms], [tab[m]["g_bar"] for m in ms]
    within[H] = {"models": ms, "F_spread": float(max(Fs) - min(Fs)), "spearman_F_g": float(spearmanr(Fs, gs)[0]),
                 "underpowered": bool(max(Fs) - min(Fs) < 0.05)}
out["within_size"] = within
out["primary_verdict"] = ("underpowered" if any(w["underpowered"] for w in within.values())
                          else out["loo"]["excl_M8"]["verdict"])

rng = np.random.default_rng(3771)
def eff_slope(sel):
    e = np.array([a["gap"] / frag[a["model"]]["F90"] for a in sel]); x = np.log2([a["H"] for a in sel])
    sl = np.polyfit(x, e, 1)[0]; bs = []
    for _ in range(10_000):
        k = rng.integers(0, len(sel), len(sel))
        if len(set(x[k])) > 1:
            bs.append(np.polyfit(x[k], e[k], 1)[0])
    return {"slope_per_doubling": float(sl), "CI95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "mean_e_by_H": {int(h): float(e[x == np.log2(h)].mean()) for h in sorted(set(a["H"] for a in sel))}}
out["efficiency"] = {"all": eff_slope(acc), "excl_M8": eff_slope([a for a in acc if a["H"] != 8])}

# labels (new models only)
J = ROOT / "judging"; key = json.load(open(J / "key.json"))
PV = {"SAME": 1.0, "PARTIAL": 0.5, "DIFFERENT": 0.0}
def parse(kind):
    o = {1: {}, 2: {}}
    for p in sorted((J / "raw").glob(f"{kind}_*_j*.txt")):
        jj = int(re.search(r"_j(\d)", p.name).group(1))
        for line in p.read_text().splitlines():
            parts = [s.strip().strip("*`") for s in line.split("|")]
            if len(parts) >= 2 and re.fullmatch(r"[A-Z0-9]{5}", parts[0]):
                o[jj][parts[0]] = [s.upper() for s in parts[1:]]
    return o
rr, pr = parse("role"), parse("pair")
ri = [it for b in key["role"] for it in b]; pi = [it for b in key["pair"] for it in b]
rid = {(it["code"], it["neuron"]): it["id"] for it in ri}
out["missing"] = {"role": sum(it["id"] not in rr[j] for it in ri for j in (1, 2)),
                  "pair": sum(it["id"] not in pr[j] for it in pi for j in (1, 2))}
lab = {}
for m in sorted({it["model"] for it in pi}):
    sc = lambda it: np.mean([PV[pr[j][it["id"]][0]] for j in (1, 2) if it["id"] in pr[j] and pr[j][it["id"]][0] in PV])
    Sm = float(np.mean([sc(it) for it in pi if it["model"] == m and it["kind"] == "matched"]))
    Sc = float(np.mean([sc(it) for it in pi if it["model"] == m and it["kind"] == "control"]))
    codes = [c for c in audit if audit[c]["model"] == m]
    leg = []
    for c in codes:
        r2 = next(a["r2"] for a in acc if a["code"] == c)
        for j in (1, 2):
            leg.append(np.mean([(rr[j].get(rid[(c, i)], ["D"])[0] in ("A", "B")) and r2[i] >= .8 for i in range(len(r2))]))
    g = tab[m]["g_bar"]
    lab[m] = {"S_matched": Sm, "S_control": Sc, "D": Sm - Sc, "legible_judge": float(np.mean(leg)), "g_bar": g,
              "dissociation": bool(Sm - Sc >= .5 and np.mean(leg) >= .75 and g < .6)}
out["labels"] = lab
m16 = [m for m in lab if lab[m] and m.startswith("M16")]
out["label_replication"] = {"n_M16_dissociated": sum(lab[m]["dissociation"] for m in m16), "of": len(m16),
                            "replicates": sum(lab[m]["dissociation"] for m in m16) >= 2}
out["arm160"] = {"accounts": arm, "T24_80call_M32_gaps": [a["gap"] for a in acc if a["model"] == "M32_s0(default)"]}
out["new_accounts"] = [a for a in acc if "r2" in a]
json.dump(out, open(ROOT / "results" / "summary377.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("models", "loo", "within_size", "primary_verdict", "efficiency", "labels",
                                      "label_replication", "missing")}, indent=1))
print("arm160:", [(a["code"], round(a["gap"], 4)) for a in arm], "vs", out["arm160"]["T24_80call_M32_gaps"])
