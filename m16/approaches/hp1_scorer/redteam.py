"""T281 red-team follow-ups for T317c (official = c70e22cd as executed) and T317d (DRI 2026-10-18; review/candidate_redteam).
(a) est(M_16,10) vs m16/truth.txt; (b) subset F (p > .95, p < .90) on HP2/HP3; (c) HP3 1k-floor baselines (constant
0.920, v5q); (d) stacked cost latitude C x 1.09 x 1.04 (+ T317c's per-call model-free table recomputation, counted in
TABLE_FLOPS). Group aggregates only. usage: python redteam.py  -> results/redteam.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
FPE, NB = 5948.0, 10_000
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
# T317c recomputes inside every est call (model-free, uncharged): r1w.gauss_v4 per region on a 40,000-point grid:
# pdf 20 + ndtr 20 + ^6 3 + product/sum/divide 3 + linspace 2 = 48 per point, K=1 moments + recurrence 11 per point
# -> 59 x 40,000 x 2 regions = 4.72e6 FLOPs; plus leggauss(20) eigensolve (~8e4) and Phi/pdf at 2 x 20 nodes (~2e3).
TABLE_FLOPS = 59 * 40_000 * 2 + 8e4 + 2e3


def panel(p):
    os.environ["PANEL"] = p
    import importlib, score
    importlib.reload(score)
    n, m, M, t = score.load_panel()
    return score, n, np.array([t[x]["acc"] for x in n])


def raw(score, name, B=0):
    r = {x["k"]: x for x in map(json.loads, open(os.path.join(score.D, f"raw_{name}.jsonl"))) if x["B"] == B}
    return np.array([r[k]["est"] for k in sorted(r)]), np.array([r[k]["cost"] for k in sorted(r)])


def F(est, p, cost_fpe, mask=None, seed=281):
    mask = np.ones(len(p), bool) if mask is None else mask
    e2, pq = ((est - p) ** 2)[mask], (p * (1 - p))[mask]
    Ce = max(float(np.mean(cost_fpe)), 1000.0)
    idx = np.random.default_rng(seed).integers(0, len(e2), (NB, len(e2)))
    Fb = e2[idx].sum(1) / pq[idx].sum(1) * Ce
    d = est[mask] - p[mask]
    return dict(n=int(mask.sum()), C=Ce, F=float(e2.mean() * Ce / pq.mean()), lo=float(np.percentile(Fb, 2.5)),
                hi=float(np.percentile(Fb, 97.5)), rms_pp=float(100 * np.sqrt(e2.mean())), bias_pp=float(100 * d.mean()),
                n_star=float(pq.mean() / e2.mean()))


out = {}
ENT = {"T317c": "t317c2_cont", "T317d": "t317d_cont"}
# (a) target
s, n, p = panel("hp1")
k0 = n.index("orig")
pt = 0.95317945
for lab, nm in ENT.items():
    e, c = raw(s, nm)
    err = e[k0] - pt
    out.setdefault("a_target", {})[lab] = dict(est=float(e[k0]), truth=pt, err_pp=float(100 * err),
                                               n_star=float(pt * (1 - pt) / err ** 2), C=float(c[k0] / FPE))
# (b) subsets, (d) stacked
for P in ("hp2", "hp3"):
    s, n, p = panel(P)
    for lab, nm in ENT.items():
        e, c = raw(s, nm); C = c / FPE
        out.setdefault("b_subsets", {}).setdefault(P, {})[lab] = {
            "all": F(e, p, C), "p>.95": F(e, p, C, p > .95), "p<.90": F(e, p, C, p < .90),
            "slope_err_vs_p": float(np.polyfit(p, e - p, 1)[0])}
        Cs = C * 1.09 * 1.04 + (TABLE_FLOPS / FPE if lab == "T317c" else 0.0)
        out.setdefault("d_stacked", {}).setdefault(P, {})[lab] = F(e, p, Cs)
# (c) baselines at the 1k floor on HP3
s, n, p = panel("hp3")
out["c_baselines_hp3"] = {"constant_0.920": F(np.full(len(p), .920), p, np.zeros(len(p)))}
e, c = raw(s, "v5q", 1000); out["c_baselines_hp3"]["v5q_1k"] = F(e, p, c / FPE)
e, c = raw(s, "v5q", 4000); out["c_baselines_hp3"]["v5q_4k"] = F(e, p, c / FPE)
e, c = raw(s, "t317c2_cont"); out["c_baselines_hp3"]["T317c"] = F(e, p, c / FPE)
out["table_flops_T317c"] = TABLE_FLOPS
json.dump(out, open(os.path.join(HERE, "results", "redteam.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
