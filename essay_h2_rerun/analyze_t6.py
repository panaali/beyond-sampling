"""T6: score the pre-registered hypotheses (README). Writes essay_h2_rerun/summary.json.

usage: python analyze_t6.py [keys_dir]   (default data/t6/keys; essay_h2_rerun/keys after unblinding)
"""
import json
import os
import sys

import numpy as np

S = "essay_h2_rerun"
K = sys.argv[1] if len(sys.argv) > 1 else "data/t6/keys"
MARGIN, NB = 0.30, 2000
rng = np.random.default_rng(0)
items = {it["id"]: it for it in json.load(open(f"{K}/items.json"))}
packets = json.load(open(f"{K}/explain_packets.json"))
judge_key = json.load(open(f"{K}/judge.json"))
calls = [p["packet"] for p in packets]
cat = {}  # (item, round) -> category
call_of = {}
for p in packets:
    out = {x["id"]: x for x in json.load(open(f"{S}/explain_out/{p['packet']}.json"))}
    assert set(out) == set(p["ids"]), p["packet"]
    for i in p["ids"]:
        cat[(i, p["round"])] = out[i]["category"]
        call_of[(i, p["round"])] = calls.index(p["packet"])
ratings = {}
for f in sorted(os.listdir(f"{S}/judge_out")):
    for r in json.load(open(f"{S}/judge_out/{f}")):
        ratings[r["jid"]] = r["ratings"]
assert set(ratings) == {j["jid"] for j in judge_key}, len(ratings)

arms = {k: [i for i, it in items.items() if it["kind"] == k] for k in ("neuron", "dense", "gauss")}
# rating table per arm: rows (item index within arm, call index, isA)
tab = {k: np.array([(n, call_of[(i, r)], cat[(i, r)] == "A") for n, i in enumerate(v) for r in (1, 2)])
       for k, v in arms.items()}


def a_rate(k, wi=None, wc=None):
    t = tab[k]
    w = np.ones(len(t)) if wi is None else wi[k][t[:, 0]] * wc[t[:, 1]]
    return float((w * t[:, 2]).sum() / w.sum())


boots = []
for _ in range(NB):
    wi = {k: np.bincount(rng.integers(0, len(v), len(v)), minlength=len(v)) for k, v in arms.items()}
    wc = np.bincount(rng.integers(0, len(calls), len(calls)), minlength=len(calls))
    boots.append({k: a_rate(k, wi, wc) for k in arms})
B = {k: np.array([b[k] for b in boots]) for k in arms}
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
out = {"n": {k: len(v) for k, v in arms.items()}, "n_explainer_calls": len(calls)}
for k in arms:
    out[f"A_rate_{k}"] = a_rate(k); out[f"A_rate_{k}_ci"] = ci(B[k])
    out[f"cats_{k}"] = {c: float(np.mean([cat[(i, r)] == c for i in arms[k] for r in (1, 2)])) for c in "ABCD"}
    a1 = np.array([cat[(i, 1)] == "A" for i in arms[k]]); a2 = np.array([cat[(i, 2)] == "A" for i in arms[k]])
    po, pe = (a1 == a2).mean(), a1.mean() * a2.mean() + (1 - a1.mean()) * (1 - a2.mean())
    out[f"agree_A_{k}"] = float(po); out[f"kappa_A_{k}"] = float((po - pe) / (1 - pe)) if pe < 1 else None
    out[f"exact_cat_agree_{k}"] = float(np.mean([cat[(i, 1)] == cat[(i, 2)] for i in arms[k]]))
for name, a, b in [("H2r", "neuron", "dense"), ("H2g", "neuron", "gauss"), ("R1test", "gauss", "dense")]:
    out[f"{name}_delta"] = out[f"A_rate_{a}"] - out[f"A_rate_{b}"]
    out[f"{name}_ci"] = ci(B[a] - B[b])
    lo, hi = out[f"{name}_ci"]
    if name != "R1test":
        out[f"{name}_verdict"] = "supports" if lo > MARGIN else "refutes" if hi < MARGIN else "inconclusive"
out["H2r0_neurons_cleaner"] = out["H2r_ci"][0] > 0
# H2r-c (amendment 1): neurons that the MX5 construction does not project out in their layer
pil = json.load(open("data/t6/pilot.json"))["stats"]
ratio = np.array(pil["absmax"]) / np.array(pil["sd"])
nidx = np.load("data/t6/neuron_idx.npy")
excl = [set(np.argsort(-ratio[L])[:ratio.shape[1] * 5 // 100].tolist()) for L in range(ratio.shape[0])]
kept = np.array([nidx[items[i]["layer"], items[i]["unit"]] not in excl[items[i]["layer"]] for i in arms["neuron"]])
t = tab["neuron"]
sel = kept[t[:, 0]]
out["H2rc_n_neurons_kept"] = int(kept.sum())
out["H2rc_A_rate_kept_neurons"] = float(t[sel, 2].mean())
out["H2rc_delta"] = out["H2rc_A_rate_kept_neurons"] - out["A_rate_dense"]
bc = []
for b in range(NB):
    ki = rng.choice(np.nonzero(kept)[0], kept.sum()); di = rng.integers(0, len(arms["dense"]), len(arms["dense"]))
    an = np.mean([t[t[:, 0] == i, 2].mean() for i in ki]); ad_ = np.mean([tab["dense"][tab["dense"][:, 0] == i, 2].mean() for i in di])
    bc.append(an - ad_)
out["H2rc_ci_items_only"] = ci(np.array(bc))
ad = out["A_rate_dense"]
out["baseline_adjusted_clean_share"] = (out["A_rate_neuron"] - ad) / (1 - ad)
out["baseline_adjusted_clean_share_ci"] = ci((B["neuron"] - B["dense"]) / (1 - B["dense"]))
h1 = out["A_rate_neuron"]
out["H1r_verdict"] = "supports" if h1 >= .6 else ("against" if h1 < .4 else "inconclusive")

# LLM-free descriptive comparisons
for k in arms:
    out[f"top_token_share_{k}"] = float(np.median([items[i]["top_token_share"] for i in arms[k]]))
    out[f"mass_above_20_{k}"] = float(np.median([items[i]["frac_mass_above_20"] for i in arms[k]]))
    out[f"heldtop_ngram_dropped_{k}"] = float(np.mean([items[i]["n_dropped_top"] > 0 for i in arms[k]]))

# H3: judges
rows = []
for j in judge_key:
    r = ratings[j["jid"]]
    assert len(r) == len(j["snippets"]) and all(x in (0, 1, 2, 3) for x in r), j["jid"]
    e_item, e_round = j["expl_of"]
    rows.append({"kind": items[j["snips_of"]]["kind"], "matched": e_item == j["snips_of"], "item": j["snips_of"],
                 "cat": cat[(e_item, e_round)],
                 "snips": [(s["bin"], s["lexical"], x) for s, x in zip(j["snippets"], r)]})


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def pooled(rs, a, b, lexfree):
    get = lambda bn: [x for r in rs for (bb, lex, x) in r["snips"] if bb == bn and not (lexfree and lex)]
    return auroc(get(a), get(b))


def boot_rows(rs, fn):
    v = [fn([rs[i] for i in rng.integers(0, len(rs), len(rs))]) for _ in range(NB)]
    return ci(np.array(v)[~np.isnan(v)])


mism = [r for r in rows if not r["matched"]]
for lexfree in (False, True):
    tag = "lexfree" if lexfree else "all"
    for k in arms:
        rs = [r for r in rows if r["kind"] == k and r["matched"]]
        for a in ("top", "mid"):
            out[f"auroc_{a}_vs_rand_{k}_{tag}"] = pooled(rs, a, "rand", lexfree)
            out[f"auroc_{a}_vs_rand_{k}_{tag}_ci"] = boot_rows(rs, lambda x, a=a: pooled(x, a, "rand", lexfree))
    out[f"auroc_top_vs_rand_mismatched_{tag}"] = pooled(mism, "top", "rand", lexfree)
    out[f"H3_matched_minus_mismatched_{tag}"] = (out[f"auroc_top_vs_rand_neuron_{tag}"] -
                                                 out[f"auroc_top_vs_rand_mismatched_{tag}"])
    out[f"H3r_{tag}_verdict"] = "pass" if (out[f"auroc_top_vs_rand_neuron_{tag}"] >= .85 and
                                           out[f"auroc_mid_vs_rand_neuron_{tag}"] >= .70 and
                                           out[f"H3_matched_minus_mismatched_{tag}"] >= .15) else "fail"
nr = [r for r in rows if r["kind"] == "neuron" and r["matched"]]
out["lexical_share_neuron_top"] = float(np.mean([lex for r in nr for (b, lex, _) in r["snips"] if b == "top"]))
out["n_lexfree_snips_neuron"] = {b: int(sum(1 for r in nr for (bb, lex, _) in r["snips"] if bb == b and not lex))
                                 for b in ("top", "mid", "rand")}

# exploratory (pre-registered as such): dense controls with top-1 share <= 0.15 only; shares vs A-score
if os.path.exists(f"{S}/density_check.json"):
    sh = {x["id"]: x["top1"] for x in json.load(open(f"{S}/density_check.json"))["items"]}
    score = lambda i: np.mean([cat[(i, r)] == "A" for r in (1, 2)])
    clean = [i for i in arms["dense"] if sh[i] <= 0.15]
    an = np.array([score(i) for i in arms["neuron"]]); ac = np.array([score(i) for i in clean])
    bx = [an[rng.integers(0, len(an), len(an))].mean() - ac[rng.integers(0, len(ac), len(ac))].mean() for _ in range(NB)]
    out["explor_dense_le015_n"] = len(clean)
    out["explor_dense_le015_A_rate"] = float(ac.mean())
    out["explor_H2r_le015_delta"] = float(an.mean() - ac.mean())
    out["explor_H2r_le015_ci_items_only"] = ci(np.array(bx))
    for k in ("dense", "gauss"):
        x = np.array([sh[i] for i in arms[k]]); y = np.array([score(i) for i in arms[k]])
        out[f"explor_corr_top1share_Ascore_{k}"] = float(np.corrcoef(x, y)[0, 1])
        q = np.quantile(x, [1 / 3, 2 / 3])
        out[f"explor_A_by_top1_tercile_{k}"] = [float(y[x <= q[0]].mean()), float(y[(x > q[0]) & (x <= q[1])].mean()),
                                                float(y[x > q[1]].mean())]

v1, v2, v3 = out["H1r_verdict"], out["H2r_verdict"], out["H3r_lexfree_verdict"]
dense_valid = json.load(open(f"{S}/density_check.json"))["valid"] if os.path.exists(f"{S}/density_check.json") else None
out["dense_controls_valid"] = dense_valid
if dense_valid is False:
    v2 = out["H2r_verdict"] = "inconclusive (control invalid)"
out["overall"] = ("supported at small scale" if (v1, v2, v3) == ("supports", "supports", "pass") else
                  "refuted at small scale" if v2 == "refutes" or v3 == "fail" else
                  "inconclusive" if v2.startswith("inconclusive") else "mixed")
json.dump(out, open(f"{S}/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
