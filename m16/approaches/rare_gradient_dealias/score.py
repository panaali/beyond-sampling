"""T386 scoring of the frozen config (results/frozen.json) on dev3_s528 + orig with T372's go line (A2 form); s526
reported as dev. Also T372's K3 and line-CMC for reference, cos with the line-CMC gradient, the converged-FD variant.
-> results/score.txt, results/score.json"""
import json
import os

import numpy as np

import common

NAMES = ("orig", "dev3_s528", "dev3_s526"); KWALK = {"orig": 574e3, "dev3_s526": 325201.0, "dev3_s528": 388e3}


def main(F, label):
    lines = [f"{label}: {F['v2']} / {F['g12']}"]
    pool = {"D": [0, 0], "D_conv": [0, 0], "K3": [0, 0]}; rows = []
    for nm in NAMES:
        D, T, C = common.R(f"dirs_{nm}.json"), common.R(f"truth_fd2_{nm}.json"), common.R(f"cmc_{nm}.json")
        r = common.R(f"cfg_{nm}_{F['v2']}_{F['g12']}.json"); g, gc, gk = np.array(r["grad"]), np.array(C["grad"]), np.array(D["grad"])
        cs = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)); dev = nm == "dev3_s526"
        lines.append(f"{nm}{' (DEV)' if dev else ''} E1({D['t0']}): truth P {T['P']:.4e}; dealiased est {r['est']:.4e} ({r['est'] / T['P']:.2f}x), "
                     f"K3 {D['est'] / T['P']:.2f}x; cos(g, g_CMC) dealiased {cs(g, gc):.3f}, K3 {cs(gk, gc):.3f}; "
                     f"cost walk {r['C_walk']:.3g} FPE ({r['C_walk'] / KWALK[nm]:.1f}x K3 walk) + endpoints {r['C_grad']:.3g} FPE")
        lines.append("  dir   FD/P truth      dealiased/P ratio ok | K3/P  ratio ok | CMC/P | conv FD/P  ratio ok")
        FD = np.array(T["FD"]); SE = np.array(T["FD_se"])
        for i in range(6):
            fd, se = T["FD_sel"][i], T["FD_sel_se"][i]; w = 1 / SE[i, 2:] ** 2
            conv = float((FD[i, 2:] * w).sum() / w.sum()); conv_se = float(1 / np.sqrt(w.sum()))
            res, ok, rr = common.judge(r["pred"][i], fd, se); _, okk, rk = common.judge(D["pred"][i], fd, se)
            resc, okc, rc = common.judge(r["pred"][i], conv, conv_se)
            if not dev:
                pool["D"][0] += res; pool["D"][1] += ok; pool["K3"][0] += res; pool["K3"][1] += okk
                pool["D_conv"][0] += resc; pool["D_conv"][1] += okc
            rows.append(dict(model=nm, dir=i, dev=dev, FD=fd, FD_se=se, pred=r["pred"][i], ratio=rr, resolved=res, ok=ok,
                             K3_pred=D["pred"][i], K3_ok=okk, CMC=C["proj"][i], FD_conv=conv, ok_conv=okc))
            lines.append(f"  {'top' if i == 5 else i:>3} {fd / T['P']:8.2f}+-{se / T['P']:5.2f} {r['pred'][i] / T['P']:9.2f} {rr:6.2f} {'Y' if ok else ('n' if res else '-')}"
                         f" | {D['pred'][i] / T['P']:7.2f} {rk:5.2f} {'Y' if okk else 'n'} | {C['proj'][i] / T['P']:7.2f} | {conv / T['P']:7.2f} {rc:5.2f} {'Y' if okc else 'n'}")
    p = pool["D"]; go = p[0] >= 8 and p[1] >= 0.8 * p[0]
    lines.append(f"GO LINE (s528 + orig): dealiased {p[1]}/{p[0]} resolved pass -> {'GO' if go else 'NO-GO'}  "
                 f"(K3 {pool['K3'][1]}/{pool['K3'][0]}; dealiased vs converged FD {pool['D_conv'][1]}/{pool['D_conv'][0]})")
    return "\n".join(lines), dict(config=F, label=label, pool=pool, go=go, rows=rows)


if __name__ == "__main__":
    F = common.R("frozen.json"); res = [main(F, "GATING (frozen by A1)")]
    res += [main(S, f"secondary, not gating ({S['why']})") for S in F["secondary_not_gating"]]
    txt = "\n\n".join(r[0] for r in res); print(txt); open(os.path.join(common.HERE, "results", "score.txt"), "w").write(txt + "\n")
    json.dump([r[1] for r in res], open(os.path.join(common.HERE, "results", "score_t386.json"), "w"), indent=1)
