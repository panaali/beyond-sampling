"""T372 scoring (pre-registered go line, README): per direction the K3 prediction g . u, the truth FD (amendment A1
level rule; literal run for s526 also scored), resolved if |FD| > 3 s.e., pass if same sign and .5 <= pred / FD <= 2.
Also the secondary (unbiased line-CMC) projections. -> results/score.json, results/score.txt"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda f: json.load(open(os.path.join(HERE, "results", f)))
NAMES = ("orig", "dev3_s526", "dev3_s528")


def judge(pred, fd, se):
    res = abs(fd) > 3 * se; r = pred / fd
    return bool(res), bool(res and np.sign(pred) == np.sign(fd) and .5 <= r <= 2), float(r)


def main():
    out = {"rows": []}; lines = []
    pool = {"K3": [0, 0], "CMC": [0, 0], "K3_conv": [0, 0], "CMC_conv": [0, 0]}
    for nm in NAMES:
        if not os.path.exists(os.path.join(HERE, "results", f"truth_fd2_{nm}.json")):
            continue
        D, T = R(f"dirs_{nm}.json"), R(f"truth_fd2_{nm}.json"); C = R(f"cmc_{nm}.json") if os.path.exists(os.path.join(HERE, "results", f"cmc_{nm}.json")) else None
        lines.append(f"{nm}{' (DEV, not scored)' if nm == 'dev3_s526' else ''} E1({D['t0']}): truth P {T['P']:.4e} +- {T['P_se']:.1e} (N {T['n']:.0e}); K3 {D['est']:.4e} ({D['est'] / T['P']:.2f}x)"
                     + (f"; line-CMC {C['P']:.4e} +- {C['P_se']:.1e} ({C['n_lines']:.0e} lines, {C['C_fpe']:.3g} FPE)" if C else ""))
        lines.append("  dir   delta     FD/P (lvl)        K3/P  ratio  ok | CMC/P            ratio ok | FD/P conv (lvl 2-4 mean)")
        FD = np.array(T["FD"]); SE = np.array(T["FD_se"])
        for i in range(len(D["dirs"])):
            lvl = T["level"][i]; fd, se = T["FD_sel"][i], T["FD_sel_se"][i]; pred = D["pred"][i]
            w = 1 / SE[i, 2:] ** 2; conv = float((FD[i, 2:] * w).sum() / w.sum()); conv_se = float(1 / np.sqrt(w.sum()))
            res, ok, r = judge(pred, fd, se); resc, okc, rc = judge(pred, conv, conv_se)
            row = dict(model=nm, dir=i, top=i == 5, delta=D["delta"][i], level=lvl, FD=fd, FD_se=se, pred=pred, resolved=res, ok=ok,
                       ratio=r, FD_conv=conv, FD_conv_se=conv_se, resolved_conv=resc, ok_conv=okc, ratio_conv=rc)
            dev = nm == "dev3_s526"; row["dev"] = dev                 # amendment A2: s526 is development, not in the go count
            if not dev:
                pool["K3"][0] += res; pool["K3"][1] += ok; pool["K3_conv"][0] += resc; pool["K3_conv"][1] += okc
            s = f"  {'top' if i == 5 else i:>3} {D['delta'][i]:8.4f} {fd / T['P']:8.2f}+-{se / T['P']:5.2f} ({lvl}) {pred / T['P']:8.2f} {r:6.2f} {'Y' if ok else ('n' if res else '-')}"
            if C:
                pc, pcs = C["proj"][i], C["proj_se"][i]; resC, okC, rC = judge(pc, fd, se); _, okCc, rCc = judge(pc, conv, conv_se)
                row.update(cmc=pc, cmc_se=pcs, ok_cmc=okC, ratio_cmc=rC, ok_cmc_conv=okCc, ratio_cmc_conv=rCc)
                if not dev:
                    pool["CMC"][0] += resC; pool["CMC"][1] += okC; pool["CMC_conv"][0] += resc; pool["CMC_conv"][1] += okCc
                s += f" | {pc / T['P']:8.2f}+-{pcs / T['P']:6.2f} {rC:6.2f} {'Y' if okC else ('n' if resC else '-')}"
            s += f" | {conv / T['P']:8.2f}+-{conv_se / T['P']:5.2f}"
            lines.append(s); out["rows"].append(row)
    for k, (nres, nok) in pool.items():
        go = nres >= 8 and nok >= .8 * nres
        lines.append(f"{k} (s528 + orig): {nok}/{nres} resolved directions pass -> {'GO' if go else ('INCONCLUSIVE' if nres < 8 else 'NO-GO')}")
        out[k] = dict(resolved=nres, ok=nok, verdict="GO" if go else ("INCONCLUSIVE" if nres < 8 else "NO-GO"))
    out["primary"] = out["K3"]["verdict"]
    lit = R("truth_fd_dev3_s526.json"); D = R("dirs_dev3_s526.json")
    lit_ok = [judge(D["pred"][i], lit["FD"][i], lit["FD_se"][i]) for i in range(6)]
    out["literal_s526"] = dict(n=lit["n"], resolved=sum(a for a, _, _ in lit_ok), ok=sum(b for _, b, _ in lit_ok), ratios=[c for _, _, c in lit_ok])
    lines.append("literal pre-registered delta (s526 only, N %.0e): %d/%d pass, ratios %s" % (lit["n"], out["literal_s526"]["ok"], out["literal_s526"]["resolved"],
                                                                                      np.round(out["literal_s526"]["ratios"], 2)))
    lines.append("(K3 = primary; CMC = secondary unbiased line estimator; *_conv = vs the inverse-variance mean of ladder levels 2-4, not pre-registered)")
    json.dump(out, open(os.path.join(HERE, "results", "score.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "score.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
