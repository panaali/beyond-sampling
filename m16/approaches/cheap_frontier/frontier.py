"""Score every (estimator, setting) on a panel: C, bias, RMS, F per group, F_max, and the (C, F_max) Pareto set.
Sources: results/sweep_raw.jsonl (this folder), LB3 raw rows of cone_enum / propagation / branch_bound
(results/leaderboard_v3_raw.jsonl, cost converted 6000 -> 5948 FLOPs/FPE), probe_v5 LB3 raw (ext/), the T274b cone
baseline bias (ext/, cost = 180 x 11 FPE, integration uncounted -> F is a lower bound).
usage: python frontier.py [lb3|dev2] -> results/frontier_<panel>.json, printed tables"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf  # noqa: E402

# step labels per family: D = all steps deductive (or declared deductive approximations); I = has an inductive step
LABEL = {
    "rank_exact": lambda c: "D" if c.get("stride", 1) == 1 else "I",
    "observer": lambda c: "D" if c.get("stride", 1) == 1 else "I",
    "probe_bisect": lambda c: "D" if c.get("stride", 1) == 1 else "I",
    "reliability": lambda c: "I",          # every rung uses a fixed subset of chambers (one rest arrangement + stride)
    "cone_enum": lambda c: "D", "propagation": lambda c: "D",
    "t274b_cone": lambda c: "D",
    "branch_bound": lambda c: "I", "probe_v5a": lambda c: "I", "probe_v5q": lambda c: "I", "probe_v5b": lambda c: "I",
}
FAM = {"rx": "rank_exact", "obs": "observer", "probe": "probe_bisect", "rel": "reliability", "tcone": "t274b_cone"}


def cfgstr(c):
    return ",".join(f"{k}={v}" for k, v in sorted(c.items()))


def collect(panel="lb3"):
    pts = {}
    p = os.path.join(cf.HERE, "results", "sweep_raw.jsonl" if panel == "lb3" else f"sweep_raw_{panel}.jsonl")
    for line in open(p):
        d = json.loads(line)
        if d["fam"] in FAM:
            fam, cfg = FAM[d["fam"]], cfgstr(d["cfg"])
        else:                                 # lb / v5: name the row after the estimator
            c = dict(d["cfg"]); fam = c.pop("name", None) or "probe_" + c.pop("which"); cfg = cfgstr(c)
        pts.setdefault((fam, cfg), dict(label=LABEL[fam](d["cfg"]), rows=[]))["rows"].append(d)
    if panel != "lb3":
        return pts
    for line in open(os.path.join(cf.ROOT, "results", "leaderboard_v3_raw.jsonl")):
        d = json.loads(line)
        if d["v"] == 3 and d["H"] == 16 and d["T"] == 10 and d["name"] in ("cone_enum", "propagation", "branch_bound") \
                and d["budget"] in (1000, 4000):
            key = (d["name"], f"budget={d['budget']}")
            pts.setdefault(key, dict(label=LABEL[d["name"]]({}), rows=[]))["rows"].append(
                dict(model=d["model"], est=d["est"], cost_fpe=d["cost"] * 6000 / cf.FPE))
    for line in open(os.path.join(cf.HERE, "ext", "probe_v5_lb3_raw.jsonl")):
        d = json.loads(line)
        if d["budget"] in (1000, 4000):
            key = (d["name"], f"budget={d['budget']}")
            pts.setdefault(key, dict(label=LABEL[d["name"]]({}), rows=[]))["rows"].append(
                dict(model=d["model"], est=d["est"], cost_fpe=d["cost"] * 6000 / cf.FPE))
    cb = json.load(open(os.path.join(cf.HERE, "ext", "t274b_cone_bias.json"))); tr = cf.lb3_truths()
    for m in (0, 1, 2):
        key = ("t274b_cone", f"m={m}")
        pts[key] = dict(label="D", rows=[dict(model=k, est=tr[k]["acc"] + v[f"cone_m{m}"],
                                                cost_fpe=180 * 11 * (1 + 0 * m)) for k, v in cb.items()],
                        note="paired bias from 400k samples; integration cost uncounted")
    return pts


def pareto(items, x, y):
    s = sorted(items, key=lambda r: (r[x], r[y])); out = []; best = np.inf
    for r in s:
        if r[y] < best:
            out.append(r); best = r[y]
    return out


def main(panel="lb3"):
    truth = cf.lb3_truths() if panel == "lb3" else cf.dev2_truths()
    groups = cf.GROUPS if panel == "lb3" else cf.DEV2_GROUPS
    pts = collect(panel); table = []
    for (fam, cfg), v in pts.items():
        sc = cf.score(v["rows"], truth, groups)
        if len(sc) < len(groups):
            continue                                  # incomplete (sweep still running)
        C = float(np.mean([r["cost_fpe"] for r in v["rows"]]))
        Fmax = max(s["F"] for s in sc.values())
        rms_all = float(np.sqrt(np.mean([(r["est"] - truth[r["model"]]["acc"]) ** 2 for r in v["rows"]]))) * 100
        table.append(dict(fam=fam, cfg=cfg, label=v["label"], C=C, Fmax=Fmax, rms_all_pp=rms_all, groups=sc,
                          n=len(v["rows"]), note=v.get("note", "")))
    json.dump(table, open(os.path.join(cf.HERE, "results", f"frontier_{panel}.json"), "w"), indent=1)
    hdr = "".join(f"{g:>17s}" for g in groups)
    print(f"{'family':13s} {'setting':38s} L {'C(FPE)':>8s} {'Fmax':>8s} | per group: bias / RMS pp / F\n{'':74s}{hdr}")
    for fam in sorted({t["fam"] for t in table}):
        rows = sorted([t for t in table if t["fam"] == fam], key=lambda t: t["Fmax"])[:6]
        for t in rows:
            print(f"{fam:13s} {t['cfg']:38s} {t['label']} {t['C']:8.0f} {t['Fmax']:8.1f} |" + "".join(
                f"{s['bias_pp']:+5.1f}/{s['rms_pp']:4.1f}/{s['F']:5.1f}".rjust(17) for s in (t['groups'][g] for g in groups)))
    print("\nPareto (C, Fmax), all labels:")
    for t in pareto(table, "C", "Fmax"):
        print(f"  {t['fam']:13s} {t['cfg']:38s} {t['label']} C={t['C']:8.0f} Fmax={t['Fmax']:8.2f}")
    print("Pareto (C, Fmax), deductive only:")
    for t in pareto([t for t in table if t["label"] == "D"], "C", "Fmax"):
        print(f"  {t['fam']:13s} {t['cfg']:38s} {t['label']} C={t['C']:8.0f} Fmax={t['Fmax']:8.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lb3")
