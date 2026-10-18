"""T351: the 4 arms on all 80 DEV2 + checks; selection by min MSE x C. -> arms_dev2.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../t347"))
import cont_f, cont_h  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "../../oracle_floor"))
from panel import dev2, ROOT  # noqa: E402
import tilt  # noqa: E402
tr = json.load(open(os.path.join(ROOT, "data/oracle_floor/dev2_truths.json")))["truths"]
ARMS = {"A0_T317f": (0.0, False), "A1_spread": (-1.0, False), "A2_tilt": (0.0, True), "A3_spread_tilt": (-1.0, True)}
rows = {}; chk = dict(a0_bitwise=True, d8_vs_d40=[])
for name, m in dev2():
    r = {"truth": tr[name]["acc"]}
    ef, ff = cont_f.est(m)
    for a, (sp, tl) in ARMS.items():
        e, f, info = cont_h.est_h(m, sp, tl, parts=True); r[a] = dict(est=e, C=f / 5948, D=info.get("D", 0.0))
    chk["a0_bitwise"] &= (r["A0_T317f"]["est"] == ef) and (abs(r["A0_T317f"]["C"] * 5948 - ff) == 0)
    chk["d8_vs_d40"].append(100 * r["A2_tilt"]["D"] - tilt.D_pred(m)[0])
    rows[name] = r
    print(name, {a: (round(100 * (r[a]["est"] - r["truth"]), 3), round(r[a]["C"])) for a in ARMS}, flush=True)
names = list(rows); p = np.array([rows[n]["truth"] for n in names]); pq = float((p * (1 - p)).mean()); S = {}
for a in ARMS:
    e = 100 * (np.array([rows[n][a]["est"] for n in names]) - p); C = float(np.mean([rows[n][a]["C"] for n in names]))
    mse = float((e ** 2).mean())
    S[a] = dict(bias=float(e.mean()), sd=float(e.std(ddof=1)), rms=float(np.sqrt(mse)), C=C, mse_x_C=mse * C,
                F=float((mse / 1e4) * max(C, 1000) / pq), corr=float(np.corrcoef(e / 100 + p, p)[0, 1]))
S["selected"] = min(ARMS, key=lambda a: S[a]["mse_x_C"])
d = np.array(chk["d8_vs_d40"]); chk["d8_vs_d40"] = dict(mean=float(d.mean()), max_abs=float(np.abs(d).max()))
S["checks"] = chk
json.dump(dict(per_model=rows, summary=S), open(os.path.join(HERE, "arms_dev2.json"), "w"), indent=1)
for k, v in S.items():
    print(k, v)
