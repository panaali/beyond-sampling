"""T333 B3 (DIAGNOSTIC): where the line-set product's overstatement lives in (g12, g23); corner-only fix and its cost."""
import json, os, sys
import numpy as np
sys.argv = sys.argv[:1]
import budget as B  # noqa: E402
import addendum as A  # noqa: E402
import addB2p as Q  # noqa: E402
from budget import Point, dev2, T, TMAX  # noqa: E402
DEL = (.025, .05, .1, .2, .5)
tr = json.load(open(os.path.join(B.ROOT, "data/oracle_floor/dev2_truths.json")))["truths"]; out = {}; models = dev2()
for idx in range(0, 80, 8):
    name, model = models[idx]; rng = np.random.default_rng(324000 + idx); x = rng.standard_normal((B.N, T)); P = Point(x)
    cut23 = P.g23 >= TMAX; cut12 = (P.g12 >= TMAX) | P.b
    comb = lambda s23, s12, X: np.where(cut23 & cut12, 1.0, np.where(cut23, s12, np.where(cut12, s23, X)))
    s23, s12, J1 = A.lines(model, P, A.anchor_from(P, A.PT1, False)); t23, t12, J2 = A.lines(model, P, A.anchor_from(P, A.PT2, True))
    P1, J1c = comb(s23, s12, s23 * s12), comb(s23, s12, J1); P2, J2c = comb(t23, t12, t23 * t12), comb(t23, t12, J2)
    r = dict(one_minus_p=1 - tr[name]["acc"])
    for tag, Pp, Jc in (("1", P1, J1c), ("2", P2, J2c)):
        D = Pp - Jc; r[f"over{tag}"] = float(100 * (D > 0).mean()); r[f"under{tag}"] = float(100 * (D < 0).mean()); r[f"net{tag}"] = float(100 * D.mean())
        for d in DEL:
            cor = (P.g12 < d) & (P.g23 < d); band = np.minimum(P.g12, P.g23) < d
            r[f"corner{tag}_{d}"] = dict(over=float(100 * ((D > 0) & cor).mean()), net=float(100 * (D * cor).mean()),
                                        band_over=float(100 * ((D > 0) & band).mean()), band_net=float(100 * (D * band).mean()), mass=float(cor.mean()))
    out[name] = r
    print(idx, name, "net1 %.3f over1 %.3f | corner net share:" % (r["net1"], r["over1"]), {d: round(r[f"corner1_{d}"]["net"] / r["net1"], 2) for d in DEL}, flush=True)
# cost of the corner joint sets (nb anchor of the one-point design; census inside [0, d]^2, 50 x 50)
cost = {}
for d in DEL:
    tot = []
    for idx in range(0, 80, 8):
        name, model = models[idx]
        Q.TMAX = d; c = Q.census(model, A.PT1["nb"], 50); Q.TMAX = TMAX
        tot.append(float(np.maximum(c - 1, 0).sum() * .54))
    cost[d] = dict(extra_fpe_mean=float(np.mean(tot)), extra_fpe_max=float(np.max(tot)))
    print("delta", d, cost[d], flush=True)
names = list(out); omp = np.array([out[n]["one_minus_p"] for n in names]) * 100
S = {"cost": cost}
for tag in ("1", "2"):
    net = np.array([out[n][f"net{tag}"] for n in names]); over = np.array([out[n][f"over{tag}"] for n in names])
    S[tag] = dict(net_mean=float(net.mean()), net_sd=float(net.std(ddof=1)), over_mean=float(over.mean()),
                  slope_net_on_err=float(np.polyfit(omp, net, 1)[0]), corr_net_err=float(np.corrcoef(omp, net)[0, 1]),
                  slope_over_on_err=float(np.polyfit(omp, over, 1)[0]), corr_over_err=float(np.corrcoef(omp, over)[0, 1]),
                  corner={d: dict(net_share=float(np.mean([out[n][f"corner{tag}_{d}"]["net"] for n in names]) / net.mean()),
                                  over_share=float(np.mean([out[n][f"corner{tag}_{d}"]["over"] for n in names]) / over.mean()),
                                  band_net_share=float(np.mean([out[n][f"corner{tag}_{d}"]["band_net"] for n in names]) / net.mean()),
                                  bias_removed_pp=float(np.mean([out[n][f"corner{tag}_{d}"]["net"] for n in names])),
                                  input_mass=float(np.mean([out[n][f"corner{tag}_{d}"]["mass"] for n in names]))) for d in DEL})
S["go"] = [d for d in DEL if S["1"]["corner"][d]["bias_removed_pp"] >= .25 and cost[d]["extra_fpe_mean"] <= 600]
json.dump(dict(per_model=out, summary=S), open(os.path.join(B.HERE, "corner.json"), "w"), indent=1)
print(json.dumps(S, indent=1))
