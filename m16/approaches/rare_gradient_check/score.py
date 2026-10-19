"""T385 scoring (pre-registered go lines in README). usage: python score.py -> results/score.txt, results/score.json"""
import json, os
import numpy as np
import core, dirs

D_ = os.path.join(core.ROOT, "data", "t385")
T372_ORIG = {"n6": 112, "n12": -44, "n7": -43, "n8": 38, "n11": -30}
G2_EPS = {"coupling n6<->n7": .03, "n2->n6": .003, "n2->n7": .003, "n2->n6 & n2->n7 (joint)": .03, "n6<-n7 only": .01, "n7<-n6 only": .01}
GATED = ["coupling n6<->n7", "n2->n6", "n2->n7"]


def est_dirs(nm, r, D):
    f = os.path.join(D_, f"est_{nm}_r{r}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f); zv, G, n = z["z"], z["G"], int(z["n"]); Pj = G @ D.T; P = zv.sum() / n; E = Pj.sum(0) / n / P
    res = Pj - E[None] * zv[:, None]; se = np.sqrt((res ** 2).sum(0) / n / n) / P
    top = np.abs(Pj).max(0) / np.maximum(np.abs(Pj).sum(0), 1e-300)          # share of the largest single line
    return dict(P=P, P_se=np.sqrt(max((zv ** 2).sum() / n - P ** 2, 0) / n), E=E, se=se, top=top, n=n, hits=len(zv))


def passes(e, t):
    return bool(np.sign(e) == np.sign(t) and 0.5 <= e / t <= 2)


def main():
    out, L = {}, []
    t372 = json.load(open(os.path.join(D_, "t372_cmc_orig.json"))); g372 = np.array(t372["grad"]) / t372["P"]
    g372se = np.array(t372["grad_se"]) / t372["P"]
    g1 = []
    for nm in dirs.MODELS:
        tf = f"results/truth_{nm}.json"
        if not os.path.exists(tf):
            continue
        T = json.load(open(tf)); names, D = dirs.all_dirs(nm); e1 = est_dirs(nm, 1, D); e2 = est_dirs(nm, 2, D)
        L.append(f"\n## {nm} E1({core.CASES[nm]})  truth P {T['P']:.4e} +- {T['P_se']:.1e} (N {T['n']:.0e})" +
                 (f"; est r1 P {e1['P']:.4e} +- {e1['P_se']:.1e} ({e1['n']:.0e} lines, {e1['hits']} hit)" if e1 else "") +
                 (f"; r2 P {e2['P']:.4e} +- {e2['P_se']:.1e}" if e2 else ""))
        L.append("| item | truth (eps) | z | est r1 | r1/truth | pass | est r2 | T372 | T372/truth |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        rows = []
        for a, row in enumerate(T["rows"]):
            t, ts = np.float64(row["truth"]), np.float64(row["truth_se"]); res = bool(np.isfinite(t) and abs(t) > 3 * ts)
            r = dict(model=nm, name=row["name"], truth=t, truth_se=ts, eps=T["ladder"][row["pick"]], resolved=res,
                     ladder=[(lv["eps"], lv["dlog"], lv["se"]) for lv in row["levels"]])
            if e1 is not None:
                r.update(est=float(e1["E"][a]), est_se=float(e1["se"][a]), top_share=float(e1["top"][a]), pass_=passes(e1["E"][a], t) if res else None)
            if e2 is not None:
                r.update(est2=float(e2["E"][a]), est2_se=float(e2["se"][a]))
            if nm == "orig":
                v = float(D[a] @ g372); vs = float(np.sqrt(((D[a] * g372se) ** 2).sum()))
                if row["name"].startswith("gain"):
                    i = int(row["name"][6:]); v, vs = t372["neuron_elast"][i], t372["neuron_elast_se"][i]
                r.update(t372=v, t372_se=vs, t372_pass=passes(v, t) if res else None)
            rows.append(r)
            if not row["name"].startswith("interv"):
                g1.append(r)
            f = lambda k: (f"{r[k]:.1f} ± {r[k + '_se']:.1f}" if k in r else "-")
            L.append(f"| {r['name']} | {t:.1f} ± {ts:.1f} ({r['eps']}) | {t / ts:.1f} | {f('est')} | "
                     f"{(r['est'] / t if 'est' in r else float('nan')):.2f} | {'✓' if r.get('pass_') else ('✗' if r.get('pass_') is False else 'unres.')} | "
                     f"{f('est2')} | {f('t372')} | {(r['t372'] / t if 't372' in r else float('nan')):.2f} |")
        out[nm] = rows
    sc = [r for r in g1 if r["resolved"] and r.get("pass_") is not None]
    if sc:
        k = sum(r["pass_"] for r in sc)
        verdict = "inconclusive (<19 resolved)" if len(sc) < 19 else ("GO" if k >= .8 * len(sc) else "NO-GO")
        L.insert(0, f"G1 (own estimator, gains + random, 2 models): {k}/{len(sc)} resolved pass ({len(g1)} items) -> {verdict}")
        out["G1"] = dict(k=k, n=len(sc), verdict=verdict)
        for nm in dirs.MODELS:
            s = [r for r in sc if r["model"] == nm]
            if s:
                L.insert(1, f"   {nm}: {sum(r['pass_'] for r in s)}/{len(s)}")
    if "orig" in out:
        by = {r["name"]: r for r in out["orig"]}
        ch = []
        for n_, v in T372_ORIG.items():
            r = by[f"gain {n_}"]; ch.append((n_, v, r["truth"], r["truth_se"], r["resolved"], r["resolved"] and passes(v, r["truth"]),
                                            (not r["resolved"]) or np.sign(v) == np.sign(r["truth"])))
        k = sum(c[5] for c in ch); sg = all(c[6] for c in ch)
        v = "REPRODUCED" if k >= 4 and sg else "NOT REPRODUCED"
        L.insert(0, f"G1-T372: {k}/5 named elasticities pass, signs {'all agree' if sg else 'DISAGREE'} -> {v}: " +
                 ", ".join(f"{c[0]} {c[1]:+d} vs {c[2]:+.1f}±{c[3]:.1f}" for c in ch))
        a16 = [r for r in out["orig"] if r["name"].startswith("gain") and r["resolved"]]
        L.insert(1, f"   all T372 orig gain elasticities: {sum(r['t372_pass'] for r in a16)}/{len(a16)} resolved pass")
        out["G1_T372"] = dict(k=k, signs=bool(sg), verdict=v)
        # G2 at the pre-registered eps
        T = json.load(open("results/truth_orig.json")); names = [r["name"] for r in T["rows"]]; g2 = []
        L.append("\n## G2 interventions (orig), observed half log-ratio at the pre-registered eps vs eps E_set")
        L.append("| intervention | eps | observed | pred T372 | ratio | pass | pred own r1 | ratio | P+/P | exp(+eps E) | P-/P | exp(-eps E) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k_, e in G2_EPS.items():
            row = T["rows"][names.index("interv " + k_)]; lv = row["levels"][T["ladder"].index(e)]
            obs, se = lv["dlog"] * e, lv["se"] * e; r = by["interv " + k_]
            pT, pO = e * r["t372"], (e * r["est"] if "est" in r else float("nan")); res = abs(obs) > 3 * se
            ok = passes(pT, obs) if res else None
            if k_ in GATED:
                g2.append(ok)
            L.append(f"| {k_}{' (gated)' if k_ in GATED else ''} | {e} | {obs:+.3f} ± {se:.3f} | {pT:+.3f} | {pT / obs:.2f} | "
                     f"{'✓' if ok else ('✗' if ok is False else 'unres.')} | {pO:+.3f} | {pO / obs:.2f} | "
                     f"{np.exp(lv['logratio_plus']):.3f} | {np.exp(pT):.3f} | {np.exp(lv['logratio_minus']):.3f} | {np.exp(-pT):.3f} |")
            out.setdefault("G2_rows", []).append(dict(name=k_, eps=e, obs=obs, obs_se=se, pred_t372=pT, pred_own=pO, pass_=ok,
                                                      ratio_plus=np.exp(lv["logratio_plus"]), ratio_minus=np.exp(lv["logratio_minus"])))
        v2 = "inconclusive" if None in g2 else ("GO" if all(g2) else "NO-GO")
        L.insert(2, f"G2 (T372-predicted intervention response, coupling / n2->n6 / n2->n7): {g2} -> {v2}")
        out["G2"] = v2
        # balance test at eps .03, per unit eps
        lv = lambda k_: T["rows"][names.index("interv " + k_)]["levels"][0]
        j, a6, a7 = (lv(k_)["dlog"] for k_ in ("n2->n6 & n2->n7 (joint)", "n2->n6", "n2->n7"))
        bal = abs(j) <= .25 * min(abs(a6), abs(a7))
        L.insert(3, f"Balance test (eps .03, per unit eps): joint {j:+.1f} vs n2->n6 {a6:+.1f}, n2->n7 {a7:+.1f} -> {'pass' if bal else 'fail'} (not gating)")
        out["balance"] = dict(joint=j, n2n6=a6, n2n7=a7, pass_=bool(bal))
    open("results/score.txt", "w").write("\n".join(L) + "\n"); json.dump(out, open("results/score.json", "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
