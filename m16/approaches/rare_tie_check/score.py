"""T403 step 6: score the sealed estimates against the truths (README pre-registration). Reads results/selection.json and
results/truth_t*_seed1.json -> results/score.json, results/score.txt."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def main():
    sel = json.load(open(os.path.join(R, "selection.json"))); tr = {}
    for f in glob.glob(os.path.join(R, "truth_t*_seed1.json")):
        d = json.load(open(f))
        for nm, r in d["models"].items():
            tr[(nm, d["t0"])] = r
    rows = []
    for c in sel["A"] + sel["B"]:
        r = tr[(c["model"], c["t0"])]; p = r["p"]; ratio = c["est"] / p if p > 0 else float("inf")
        Ceff = max(c["C"], 1000.0)
        rows.append(dict(c, truth=p, se=r["se"], hits=r["hits"], n=r["n"], tie_truth=r["hits_tie"] / max(r["hits"], 1),
                         ratio=ratio, within2=bool(0.5 <= ratio <= 2), F=(c["est"] - p) ** 2 * Ceff / (p * (1 - p)),
                         pC=p * c["C"]))
    lines = []; P = lambda s: (lines.append(s), print(s))

    def summ(name, rs):
        if not rs:
            P(f"{name}: n 0"); return {}
        r = np.array([x["ratio"] for x in rs]); w = int(sum(x["within2"] for x in rs))
        d = dict(n=len(rs), within2=w, frac=w / len(rs), median=float(np.median(r)), mean_log10=float(np.mean(np.log10(r))),
                 high=int((r > 1).sum()), over2=int((r > 2).sum()), under_half=int((r < .5).sum()), min=float(r.min()), max=float(r.max()))
        P(f"{name}: n {d['n']}  within 2x {w} ({d['frac']:.0%})  median {d['median']:.2f}  mean log10 {d['mean_log10']:+.3f}  "
          f">2x {d['over2']}  <.5x {d['under_half']}  range {d['min']:.2f}-{d['max']:.2f}")
        return d
    S = {}
    P("T403 score (est = frozen estimator_t366, sealed; truth = own v3-conditional MC)")
    P("stratum model t0 | est truth (se) hits | ratio | s_pred tie_truth | C(FPE) pC F")
    for x in rows:
        P(f"{x['stratum']} {x['model']} {x['t0']:g} | {x['est']:.3e} {x['truth']:.3e} ({x['se'] / x['truth']:.1%}) {x['hits']} | "
          f"{x['ratio']:.2f}{'' if x['within2'] else ' MISS'} | {x['s']:.2f} {x['tie_truth']:.2f} | {x['C']:.3g} {x['pC']:.2g} {x['F']:.3g}")
    S["all"] = summ("ALL (pre-registered line: >= 80% within 2x)", rows)
    S["A"] = summ("predicted share 50-80%", [x for x in rows if x["stratum"].startswith("A")])
    S["B"] = summ("predicted share >= 80%", [x for x in rows if x["stratum"].startswith("B")])
    for lo, hi, nm in ((0, .5, "truth tie < 50%"), (.5, .8, "truth tie 50-80%"), (.8, 1.01, "truth tie >= 80%")):
        S[nm] = summ(nm, [x for x in rows if lo <= x["tie_truth"] < hi])
    S["verdict"] = "GO" if S["all"]["frac"] >= 0.8 else "NO GO"
    e2 = np.array([(x["est"] - x["truth"]) ** 2 for x in rows]); pq = np.array([x["truth"] * (1 - x["truth"]) for x in rows])
    se2 = np.array([x["se"] ** 2 for x in rows]); C = np.array([max(x["C"], 1000.0) for x in rows]); F = np.array([x["F"] for x in rows])
    S["F_panel"] = float(e2.mean() * C.mean() / pq.mean()); S["F_panel_noise_corr"] = float(max(e2.mean() - se2.mean(), 0) * C.mean() / pq.mean())
    S["F_point_median"] = float(np.median(F)); S["F_point_le1"] = int((F <= 1).sum()); S["F_point_le10"] = int((F <= 10).sum())
    S["F_rel_panel"] = float(np.mean([(x["ratio"] - 1) ** 2 * x["pC"] for x in rows]))
    P(f"F (Directive 8 amendment 2, C_eff = max(C, 1k), plain MC at equal FPE): panel {S['F_panel']:.3g} "
      f"(truth-noise corrected {S['F_panel_noise_corr']:.3g}); per point median {S['F_point_median']:.3g}, "
      f"<= 1 (strict) {S['F_point_le1']}/{len(rows)}, <= 10 (lenient) {S['F_point_le10']}/{len(rows)}")
    P(f"strict bar (F <= 1): {'PASS' if S['F_panel'] <= 1 else 'FAIL'}; lenient bar (F <= 10): {'PASS' if S['F_panel'] <= 10 else 'FAIL'}")
    P(f"VERDICT by the pre-registered line: {S['verdict']}")
    json.dump(dict(summary=S, rows=rows), open(os.path.join(R, "score.json"), "w"), indent=1)
    open(os.path.join(R, "score.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
