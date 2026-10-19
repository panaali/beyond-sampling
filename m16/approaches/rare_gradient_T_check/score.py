"""T405 scoring (pre-registered in README): T372's go line for the frozen T5/gl20 gradient (and K3 as a reference),
cost vs the K3 walk, F against both bars. usage: python score.py -> results/score.txt, results/score.json"""
import json
import os

import numpy as np

import make_cases

HERE = os.path.dirname(os.path.abspath(__file__))


def judge(pred, fd, se):
    res = abs(fd) > 3 * se; r = pred / fd
    return bool(res), bool(res and np.sign(pred) == np.sign(fd) and .5 <= r <= 2), float(r)


def main():
    L = []; rows = []; tot = {"T5": [0, 0], "K3": [0, 0]}; nres = 0; ndir = 0; Fsum = {"plain": [0.0, 0.0], "cond": [0.0, 0.0]}
    for nm, t0 in make_cases.CASES:
        C = json.load(open(make_cases.path(nm, t0))); T = json.load(open(os.path.join(HERE, "results", f"truth_{nm}_{t0}.json")))
        P = T["P"]; pg = T["p_gap"]; n = T["n"]; t5, k3 = C["t5"], C["k3"]
        cT5 = t5["C_walk"] + t5["C_grad"]; cK3 = k3["C_walk"]
        L.append(f"\n{nm} E1({t0}): truth P {P:.4e} +- {T['P_se']:.1e} (N {n:.0e}); T5/P {t5['est'] / P:.3f}, K3/P {k3['est'] / P:.3f}; "
                 f"cost T5 {t5['C_walk']:.3g} + {t5['C_grad']:.3g} FPE = {cT5 / cK3:.2f}x the K3 walk ({cK3:.3g}); "
                 f"F_P(T5) {(t5['est'] - P) ** 2 * cT5 / (P * (1 - P)):.2e}")
        L.append("  dir | truth dlogP (lvl, z) | T5 dlogP  ratio pass | K3 dlogP  ratio pass")
        cs = {"T5": [0, 0], "K3": [0, 0]}; e2 = 0.0; Vp = 0.0; Vc = 0.0
        for i in range(6):
            lv = T["level"][i]; fd = T["FD_sel"][i]; se = T["FD_sel_se"][i]; dl = T["delta"][i][lv]
            md = (T["hits_plus"][i][lv] - T["hits_minus"][i][lv]) / n; Ed2 = T["flips"][i][lv] / n
            Neq = cT5 / 2; Vp += (pg * Ed2 - (pg * md) ** 2) / (4 * dl * dl * Neq); Vc += pg * pg * (Ed2 - md * md) / (4 * dl * dl * Neq)
            e2 += (t5["pred"][i] - fd) ** 2
            out = {}
            for k, E in (("T5", t5), ("K3", k3)):
                res, ok, r = judge(E["pred"][i], fd, se); out[k] = (ok, r)
                cs[k][0] += ok; cs[k][1] += res
            nres += res; ndir += 1
            rows.append(dict(model=nm, t0=t0, dir=i, FD=fd, se=se, level=lv, resolved=res, P=P, T5=t5["pred"][i], K3=k3["pred"][i],
                             T5_pass=out["T5"][0], K3_pass=out["K3"][0]))
            L.append(f"  {'top' if i == 5 else i:>3} | {fd / P:9.2f} ({lv}, {fd / se:6.1f}) | {t5['pred'][i] / P:8.2f} {out['T5'][1]:6.2f} "
                     f"{'Y' if out['T5'][0] else ('-' if not res else 'n')} | {k3['pred'][i] / P:8.2f} {out['K3'][1]:6.2f} "
                     f"{'Y' if out['K3'][0] else ('-' if not res else 'n')}")
        for k in cs:
            tot[k][0] += cs[k][0]; tot[k][1] += cs[k][1]
        Fsum["plain"][0] += e2; Fsum["plain"][1] += Vp; Fsum["cond"][0] += e2; Fsum["cond"][1] += Vc
        L.append(f"  pass T5 {cs['T5'][0]}/{cs['T5'][1]} resolved, K3 {cs['K3'][0]}/{cs['K3'][1]}; "
                 f"F_g(T5) vs plain CRN-FD sampling at equal FPE {e2 / Vp:.3g}, vs conditional-law sampling {e2 / Vc:.3g}")
        rows[-1]["F_g_plain_case"] = e2 / Vp; rows[-1]["F_g_cond_case"] = e2 / Vc
    fr = tot["T5"][0] / max(tot["T5"][1], 1)
    verdict = "INCONCLUSIVE (< 24 resolved)" if nres < 24 else ("GO" if fr >= .8 else "NO-GO")
    head = [f"T405 pooled over {ndir} directions: resolved {nres}; T5/gl20 passes {tot['T5'][0]}/{tot['T5'][1]} ({fr:.0%}) -> {verdict}; "
            f"K3 reference {tot['K3'][0]}/{tot['K3'][1]}",
            f"F_g pooled (sum e^2 / sum V): plain sampling {Fsum['plain'][0] / Fsum['plain'][1]:.3g}, conditional-law sampling "
            f"{Fsum['cond'][0] / Fsum['cond'][1]:.3g} (strict <= 1, lenient <= 10)"]
    open(os.path.join(HERE, "results", "score.txt"), "w").write("\n".join(head + L) + "\n")
    json.dump(dict(verdict=verdict, resolved=nres, T5=tot["T5"], K3=tot["K3"], rows=rows,
                   F_g_plain=Fsum["plain"][0] / Fsum["plain"][1], F_g_cond=Fsum["cond"][0] / Fsum["cond"][1]),
              open(os.path.join(HERE, "results", "score.json"), "w"), indent=1)
    print("\n".join(head + L))


if __name__ == "__main__":
    main()
