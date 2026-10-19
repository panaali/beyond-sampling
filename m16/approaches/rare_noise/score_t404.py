"""T404 scoring (pre-registered). usage: python score_t404.py a|b -> results/t404_score_{a|b}.json and stdout.
F = r^2 / s^2, r = |est/truth - 1|, s = min(1, sqrt((1 - p) / (p C))) at the estimator's own executed C (FPE)."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import BASES, SIGMAS  # noqa: E402

ok = lambda x: .5 <= x <= 2


def load(pat):
    d = {}
    for fn in glob.glob(os.path.join(HERE, "results", pat)):
        d.update(json.load(open(fn)))
    return d


def F(e, p, C):
    s2 = min(1.0, (1 - p) / (p * C)); return (e / p - 1) ** 2 / s2


def parse(nm):
    return (nm.split("@")[0], float(nm.split("@")[1].split("#")[0])) if "@" in nm else (nm, 0.0)


def summ(rs, key="ratio"):
    x = np.array([r[key] for r in rs]); f = np.array([r["F"] for r in rs])
    return dict(n=len(rs), within2=int(sum(ok(v) for v in x)), frac2=float(np.mean([ok(v) for v in x])), median=float(np.median(x)),
                mean_log10=float(np.mean(np.log10(np.maximum(x, 1e-300)))), n_high=int((x > 1).sum()), n_low=int((x < 1).sum()),
                F_le1=int((f <= 1).sum()), F_le10=int((f <= 10).sum()), F_median=float(np.median(f)))


def breakdown(rows):
    noisy = [r for r in rows if r["sigma"] > 0]
    return dict(all_noisy=summ(noisy), clean=summ([r for r in rows if r["sigma"] == 0]),
                by_sigma={str(s): summ([r for r in noisy if r["sigma"] == s]) for s in SIGMAS},
                by_base={b: summ([r for r in noisy if r["base"] == b]) for b in BASES},
                by_base_sigma={f"{b}@{s:g}": summ([r for r in noisy if r["base"] == b and r["sigma"] == s]) for b in BASES for s in SIGMAS})


part = sys.argv[1]
if part == "a":
    S = json.load(open(os.path.join(HERE, "results", "score.json"))); A = load("t404_a_*.json"); G = load("t388_est_*.json"); rows = []
    for r in S["rows"]:
        e = A[r["model"]]["est"][str(r["t0"])]; C = A[r["model"]]["C_fpe"]
        rows.append(dict(model=r["model"], base=r["base"], sigma=r["sigma"], t0=r["t0"], truth=r["truth"], tie=r["tie_share"], est=e, ratio=e / r["truth"],
                         F=F(e, r["truth"], C), C=C, C_k3=r["C_fpe"], C_t388=G[r["model"]]["C_fpe"],
                         d_vs_t388=e - G[r["model"]]["est"][str(r["t0"])]))
    out = breakdown(rows); out["rows"] = rows
    out["verdict"] = "GO" if out["all_noisy"]["within2"] >= 140 else "NO GO"
    cm = {r["model"]: r for r in rows}
    out["C_over_k3"] = dict(median=float(np.median([r["C"] / r["C_k3"] for r in cm.values()])), min=min(r["C"] / r["C_k3"] for r in cm.values()), max=max(r["C"] / r["C_k3"] for r in cm.values()))
    out["C_over_t388"] = dict(median=float(np.median([r["C"] / r["C_t388"] for r in cm.values()])), min=min(r["C"] / r["C_t388"] for r in cm.values()), max=max(r["C"] / r["C_t388"] for r in cm.values()))
    out["C_range"] = [min(r["C"] for r in cm.values()), max(r["C"] for r in cm.values())]
    out["max_abs_d_vs_t388"] = max(abs(r["d_vs_t388"]) for r in rows); out["bound"] = 7.037785093518071e-08
    out["misses"] = [r for r in rows if r["sigma"] > 0 and not ok(r["ratio"])]
else:
    tr = {json.load(open(f))["t"]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "results", "t404_truth_*.json"))}
    out = {}
    for lab, pat in (("K3", "t404_k3e2_*.json"), ("A", "t404_ae2_*.json")):
        E = load(pat); rows = []
        for nm in sorted(E):
            b, s = parse(nm)
            for t in sorted(tr):
                T = tr[t]; k = T["hits"][nm]; p = T["p_cond"] * k / T["n"]; e = E[nm]["est"][str(t)]; C = E[nm]["C_fpe"]
                br = T["by_rank"][nm]; wrong = max(k, 1)
                rows.append(dict(model=nm, base=b, sigma=s, t=t, truth=p, hits=k, rel_se=1 / np.sqrt(max(k, 1)), est=e, ratio=e / p, F=F(e, p, C), C=C,
                                 share_rank1=br[0] / wrong, share_rank3=br[2] / wrong, share_rank4plus=sum(br[3:]) / wrong))
        o = breakdown(rows); o["rows"] = rows; o["verdict"] = "GO" if o["all_noisy"]["frac2"] >= .8 else "NO GO"
        o["by_t"] = {str(t): summ([r for r in rows if r["sigma"] > 0 and r["t"] == t]) for t in sorted(tr)}
        o["C_range"] = [min(r["C"] for r in rows), max(r["C"] for r in rows)]; out[lab] = o
json.dump(out, open(os.path.join(HERE, "results", f"t404_score_{part}.json"), "w"), indent=1)


def show(o, pre=""):
    for k in ("all_noisy", "clean"):
        print(pre, k, o[k])
    for k, v in o["by_sigma"].items():
        print(pre, "sigma", k, v)
    for k, v in o["by_base"].items():
        print(pre, "base", k, v)
    for k in ("by_t", "C_range", "C_over_k3", "C_over_t388", "max_abs_d_vs_t388", "verdict"):
        if k in o:
            print(pre, k, o[k])


if part == "a":
    show(out, "a")
    for r in out["misses"]:
        print("miss", r["model"], r["t0"], round(r["ratio"], 3), round(r["tie"], 3))
else:
    for lab in out:
        show(out[lab], lab)
