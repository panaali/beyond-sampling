"""T383 scoring (own code): A-rates, two-way bootstrap CIs, pre-registered verdict, H1, H3, balance, descriptives.

usage: python score.py [keys_dir]   (default data/t383/keys) -> essay_h2_check/summary.json
"""
import json
import os
import sys

import numpy as np

S = "essay_h2_check"
K = sys.argv[1] if len(sys.argv) > 1 else "data/t383/keys"
B, MARGIN, CALIPER = 2000, 0.30, 0.05
ARMS = ("plain", "matched", "control")

items = json.load(open(f"{K}/items.json"))
by_id = {it["id"]: it for it in items}
packets = json.load(open(f"{K}/explain_packets.json"))
sel = json.load(open(f"{K}/select.json"))
choice = json.load(open(f"{S}/choice.json"))
call_of = {(i, p["round"]): n for n, p in enumerate(packets) for i in p["ids"]}
cat = {}
for p in packets:
    for x in json.load(open(f"{S}/explain_out/{p['packet']}.json")):
        cat[(x["id"], p["round"])] = x["category"]
assert set(cat) == set(call_of), "missing explanations"
ids = {a: [it["id"] for it in items if it["arm"] == a] for a in ARMS}
Aex = {a: np.array([[cat[(i, r)] == "A" for r in (1, 2)] for i in ids[a]], float) for a in ARMS}
CALL = {a: np.array([[call_of[(i, r)] for r in (1, 2)] for i in ids[a]]) for a in ARMS}
NC = len(packets)
rng = np.random.default_rng(3834)


def boot(subsets=None):
    """Two-way bootstrap: (B, arm) A-rates. subsets: optional {arm: index array} restricting items."""
    out = {a: np.empty(B) for a in ARMS}
    for b in range(B):
        wc = rng.multinomial(NC, np.ones(NC) / NC)
        for a in ARMS:
            base = np.arange(len(ids[a])) if subsets is None else subsets[a]
            idx = base[rng.integers(0, len(base), len(base))]
            w = wc[CALL[a][idx]]
            ws = w.sum(1)
            ok = ws > 0
            out[a][b] = ((w * Aex[a][idx]).sum(1)[ok] / ws[ok]).mean()
    return out


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def rule(lo, hi):
    return "supports" if lo > MARGIN else "refutes" if hi < MARGIN else "inconclusive"


def paired(pairs):
    """Paired matched - control A-score difference; bootstrap over pairs and calls."""
    pos = {a: {i: n for n, i in enumerate(ids[a])} for a in ARMS}
    key = {(it["arm"], it["layer"], it["j"]): it["id"] for it in items}
    pm = np.array([pos["matched"][key[("matched", p["layer"], p["pool"])]] for p in pairs])
    pc = np.array([pos["control"][key[("control", p["layer"], p["control"])]] for p in pairs])
    est = float(Aex["matched"][pm].mean() - Aex["control"][pc].mean())
    d = np.empty(B)
    for b in range(B):
        wc = rng.multinomial(NC, np.ones(NC) / NC)
        k = rng.integers(0, len(pairs), len(pairs))
        sc = []
        for a, ix in (("matched", pm[k]), ("control", pc[k])):
            w = wc[CALL[a][ix]]
            ws = w.sum(1)
            sc.append(np.where(ws > 0, (w * Aex[a][ix]).sum(1) / np.maximum(ws, 1), np.nan))
        d[b] = np.nanmean(sc[0] - sc[1])
    return est, ci(d), pm, pc


def kappa(x):
    a, b = x[:, 0], x[:, 1]
    po = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return po, float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def h3():
    jk = json.load(open(f"{K}/judge.json"))
    rat = {}
    for f in sorted(os.listdir(f"{S}/judge_out")):
        for x in json.load(open(f"{S}/judge_out/{f}")):
            rat[x["jid"]] = x["ratings"]
    assert all(j["jid"] in rat and len(rat[j["jid"]]) == len(j["snippets"]) for j in jk), "missing judge ratings"
    out = {}

    def rows(sel_items, lexfree):
        r = []
        for j in sel_items:
            for s, v in zip(j["snippets"], rat[j["jid"]]):
                if lexfree and s["lexical"]:
                    continue
                r.append((s["bin"], v, j["jid"]))
        return r

    def aurocs(sel_items, lexfree):
        r = rows(sel_items, lexfree)
        g = {k: [v for b, v, _ in r if b == k] for k in ("top", "mid", "rand")}
        return auroc(g["top"], g["rand"]), auroc(g["mid"], g["rand"])

    def boot_ci(sel_items, lexfree, nb=500):
        vals = []
        for _ in range(nb):
            pick = [sel_items[i] for i in rng.integers(0, len(sel_items), len(sel_items))]
            vals.append(aurocs(pick, lexfree))
        v = np.array(vals)
        return [ci(v[:, 0]), ci(v[:, 1])]

    mism = [j for j in jk if j["expl_of"][0] != j["snips_of"]]
    for lexfree in (False, True):
        tag = "lexfree" if lexfree else "all"
        mm_top = aurocs(mism, lexfree)[0]
        out[f"mismatched_top_vs_rand_{tag}"] = mm_top
        for a in ARMS:
            m = [j for j in jk if j["expl_of"][0] == j["snips_of"] and by_id[j["snips_of"]]["arm"] == a]
            t, md = aurocs(m, lexfree)
            c = boot_ci(m, lexfree)
            out[f"{a}_{tag}"] = {"top_vs_rand": t, "top_ci": c[0], "mid_vs_rand": md, "mid_ci": c[1],
                                 "matched_minus_mismatched": t - mm_top}
    lex_top = [s["lexical"] for j in jk if j["expl_of"][0] == j["snips_of"] for s in j["snippets"] if s["bin"] == "top"]
    out["frac_lexical_top"] = float(np.mean(lex_top))
    for a in ("plain", "matched"):
        x = out[f"{a}_lexfree"]
        out[f"H3_{a}_pass"] = bool(x["top_vs_rand"] >= 0.85 and x["mid_vs_rand"] >= 0.70 and
                                   x["matched_minus_mismatched"] >= 0.15)
    return out


def logistic(x, y, iters=50):
    X = np.stack([np.ones_like(x), x], 1)
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        beta += np.linalg.solve(H, X.T @ (y - p))
    return beta


def main():
    est = {a: float(Aex[a].mean()) for a in ARMS}
    bt = boot()
    res = {"n": {a: len(ids[a]) for a in ARMS}, "n_explainer_calls": NC,
           "A_rate": {a: {"est": est[a], "ci": ci(bt[a])} for a in ARMS}}
    dc = bt["plain"] - bt["control"]
    dm = bt["matched"] - bt["control"]
    dpm = bt["plain"] - bt["matched"]
    res["H2c"] = {"delta": est["plain"] - est["control"], "ci": ci(dc), "decision": rule(*ci(dc))}
    bal = choice["balance"]
    pairs = sel["pairs"]
    within = [p for p in pairs if p["absdiff"] <= CALIPER]
    if bal["pass"]:
        res["H2m"] = {"delta": est["matched"] - est["control"], "ci": ci(dm), "decision": rule(*ci(dm)),
                      "basis": "full arms (balance check passed)"}
    elif len(within) >= 100:
        e, c, _, _ = paired(within)
        res["H2m"] = {"delta": e, "ci": c, "decision": rule(*c), "basis": f"{len(within)} within-caliper pairs"}
    else:
        res["H2m"] = {"delta": None, "ci": None, "decision": "untestable", "basis": f"{len(within)} pairs < 100"}
    e, c, _, _ = paired(pairs)
    res["paired_matched_minus_control_all_pairs"] = {"delta": e, "ci": c}
    if within:
        e, c, _, _ = paired(within)
        res["paired_within_caliper"] = {"n": len(within), "delta": e, "ci": c}
    res["plain_minus_matched"] = {"delta": est["plain"] - est["matched"], "ci": ci(dpm)}
    res["H2_0"] = {"plain_ci_excludes_0": ci(dc)[0] > 0, "matched_ci_excludes_0": ci(dm)[0] > 0}
    res["H1"] = {"A_plain": est["plain"], "decision": "supports" if est["plain"] >= 0.6 else
                 "inconclusive" if est["plain"] >= 0.4 else "against"}
    h2c, h2m = res["H2c"]["decision"], res["H2m"]["decision"]
    if h2c == "refutes":
        v = "REFUTED"
    elif h2c == "supports":
        v = {"supports": "SUPPORTED", "refutes": "TEXTURE-DEPENDENT"}.get(h2m, "SUPPORTED (plain); texture question open")
    else:
        v = "INCONCLUSIVE (plain); matched supports" if h2m == "supports" else "INCONCLUSIVE"
    res["verdict"] = v
    res["categories"] = {a: {c: float(np.mean([[cat[(i, r)] == c for r in (1, 2)] for i in ids[a]])) for c in "ABCD"}
                         for a in ARMS}
    res["agreement"] = {a: dict(zip(("raw", "kappa"), kappa(Aex[a]))) for a in ARMS}
    desc = {}
    for a in ARMS:
        its = [by_id[i] for i in ids[a]]
        desc[a] = {k: float(np.median([it[k] for it in its])) for k in ("texture", "flat20", "top_token_share")}
        desc[a]["mean_texture"] = float(np.mean([it["texture"] for it in its]))
        desc[a]["frac_held_top_replaced"] = float(np.mean([it["n_dropped_top"] > 0 for it in its]))
    res["descriptives"] = desc
    res["gate"] = choice["gate"]
    res["balance"] = bal
    tx = np.array([by_id[i]["texture"] for a in ("plain", "matched") for i in ids[a] for _ in (1, 2)])
    ya = np.concatenate([Aex[a].reshape(-1) for a in ("plain", "matched")])
    beta = logistic((tx - tx.mean()) / tx.std(), ya)
    res["exploratory_logistic_A_on_texture_neurons"] = {"slope_per_sd": float(beta[1]), "sd_texture": float(tx.std())}
    ct = np.array([by_id[i]["texture"] for i in ids["control"] for _ in (1, 2)])
    res["exploratory_corr_texture_A_control"] = float(np.corrcoef(ct, Aex["control"].reshape(-1))[0, 1])
    if os.path.isdir(f"{S}/judge_out"):
        res["H3"] = h3()
    json.dump(res, open(f"{S}/summary.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("A_rate", "H2c", "H2m", "plain_minus_matched", "H1", "verdict")}, indent=1))


main()
