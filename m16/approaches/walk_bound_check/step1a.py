"""Step 1a: tail of the weight-derived switch-point sensitivity rho = |P c| / |c.d| on the frozen T10 lines."""
import json, sys
import numpy as np
import plwalk as P


def hill(x, frac):
    x = np.sort(x[np.isfinite(x) & (x > 0)])[::-1]
    k = max(int(frac * len(x)), 5)
    return float(1 / np.mean(np.log(x[:k] / x[k])))


out = {}
for name in (sys.argv[1:] or P.MODELS):
    m = P.load(name)
    E, KE, G, mE256, mG256 = [], [], [], [], []
    for j in range(len(P.STRATA)):
        U = P.C.sobol(P.T, 2048, P.RULE_SEED + 101 * j)
        o = P.run(m, j, U, want_gates=True)
        for i in range(2048):
            rE, kE, rG = P.sensitivity(m, o["w"][i], o["d"][i], o["roots"][i], o["gates"][i])
            E.append(rE); KE.append(kE * rE); G.append(rG)
            if i == 255:
                mE256.append(np.concatenate(E[-256:]).max()); mG256.append(np.concatenate(G[-256:]).max())
    E, KE, G = (np.concatenate(v) for v in (E, KE, G))
    r = {}
    for lab, x in (("E", E), ("G", G)):
        r[lab] = {"n": int(len(x)), "median": float(np.median(x)), "p99": float(np.quantile(x, .99)),
                  "max": float(x.max()), **{f"hill{int(f*100)}": hill(x, f) for f in (.01, .02, .05)}}
    r["E"]["max_k_rho"] = float(KE.max()); r["E"]["p99_k_rho"] = float(np.quantile(KE, .99))
    r["E"]["max256"] = float(max(mE256)); r["G"]["max256"] = float(max(mG256))
    out[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(out, open("step1a.json", "w"), indent=1)
