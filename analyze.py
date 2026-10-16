"""Score the pre-registered hypotheses. Writes results/summary.json and prints a report."""
import json
import os

import numpy as np

S = "study"
rng = np.random.default_rng(0)
items = {it["id"]: it for it in json.load(open(f"{S}/keys/items.json"))}
expl = {}
for f in sorted(os.listdir(f"{S}/explain_out")):
    for r in json.load(open(f"{S}/explain_out/{f}")):
        expl[r["id"]] = r
judge_key = {j["jid"]: j for j in json.load(open(f"{S}/keys/judge.json"))}
ratings = {}
for f in sorted(os.listdir(f"{S}/judge_out")):
    for r in json.load(open(f"{S}/judge_out/{f}")):
        ratings[r["jid"]] = r["ratings"]
assert set(ratings) == set(judge_key), set(judge_key) ^ set(ratings)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def boot(fn, units, n=2000):
    vals = [fn([units[i] for i in rng.integers(0, len(units), len(units))]) for _ in range(n)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def a_rate(ids):
    return float(np.mean([expl[i]["category"] == "A" for i in ids]))


def cat_dist(ids):
    return {c: float(np.mean([expl[i]["category"] == c for i in ids])) for c in "ABCD"}


# per-judge-item rows: (kind_of_snippet_owner, matched?, {bin: [ratings]})
rows = []
for jid, j in judge_key.items():
    r = ratings[jid]
    assert len(r) == len(j["snippets"]), jid
    by = {"top": [], "mid": [], "rand": []}
    for s, x in zip(j["snippets"], r):
        by[s["bin"]].append(x)
    rows.append({"kind": items[j["snips_of"]]["kind"], "matched": j["expl_of"] == j["snips_of"],
                 "cat": expl[j["expl_of"]]["category"], **by})


def pooled(rs, a, b):
    return auroc(sum((r[a] for r in rs), []), sum((r[b] for r in rs), []))


real = [i for i, it in items.items() if it["kind"] == "neuron"]
rand = [i for i, it in items.items() if it["kind"] == "random_dir"]
R = lambda kind, matched=True: [r for r in rows if r["kind"] == kind and r["matched"] == matched]
ctrl = [r for r in rows if not r["matched"]]

out = {"n_real": len(real), "n_random_dir": len(rand), "n_control": len(ctrl)}
out["H1_A_rate_real"] = a_rate(real); out["H1_ci"] = boot(a_rate, real)
out["cats_real"] = cat_dist(real); out["cats_random_dir"] = cat_dist(rand)
out["A_rate_random_dir"] = a_rate(rand); out["A_rate_random_dir_ci"] = boot(a_rate, rand)
out["H2_diff"] = out["H1_A_rate_real"] - out["A_rate_random_dir"]
diffs = [a_rate([real[i] for i in rng.integers(0, len(real), len(real))]) -
         a_rate([rand[i] for i in rng.integers(0, len(rand), len(rand))]) for _ in range(2000)]
out["H2_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
for name, rs in [("real", R("neuron")), ("random_dir", R("random_dir")), ("control", ctrl),
                 ("real_catA", [r for r in R("neuron") if r["cat"] == "A"])]:
    out[f"auroc_top_vs_rand_{name}"] = pooled(rs, "top", "rand")
    out[f"auroc_mid_vs_rand_{name}"] = pooled(rs, "mid", "rand")
    out[f"auroc_top_vs_mid_{name}"] = pooled(rs, "top", "mid")
    if rs:
        out[f"auroc_top_vs_rand_{name}_ci"] = boot(lambda x: pooled(x, "top", "rand"), rs)
        out[f"auroc_mid_vs_rand_{name}_ci"] = boot(lambda x: pooled(x, "mid", "rand"), rs)
out["mean_rating_by_bin_real"] = {b: float(np.mean(sum((r[b] for r in R("neuron")), []))) for b in
                                  ("top", "mid", "rand")}
out["mean_rating_by_bin_control"] = {b: float(np.mean(sum((r[b] for r in ctrl), []))) for b in
                                     ("top", "mid", "rand")}
real_rows = R("neuron")
out["H3_matched_minus_control_top"] = out["auroc_top_vs_rand_real"] - out["auroc_top_vs_rand_control"]
out["H3_matched_minus_control_mid"] = out["auroc_mid_vs_rand_real"] - out["auroc_mid_vs_rand_control"]
out["mass_above_20pct_real"] = float(np.median([items[i]["frac_mass_above_20"] for i in real]))
out["mass_above_20pct_random_dir"] = float(np.median([items[i]["frac_mass_above_20"] for i in rand]))

h1 = out["H1_A_rate_real"]
out["verdict_H1"] = "supports" if h1 >= .6 else ("against" if h1 < .4 else "inconclusive")
out["verdict_H2"] = "pass" if out["H2_diff"] >= .30 else "fail"
out["verdict_H3"] = "pass" if (out["auroc_top_vs_rand_real"] >= .85 and out["auroc_mid_vs_rand_real"] >= .70
                               and out["H3_matched_minus_control_top"] >= .15) else "fail"
v = (out["verdict_H1"], out["verdict_H2"], out["verdict_H3"])
out["overall"] = ("supported at small scale" if v == ("supports", "pass", "pass") else
                  "refuted at small scale" if "fail" in v[1:] else "mixed")
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
