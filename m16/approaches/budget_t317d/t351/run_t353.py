"""T353 gate + go on all 80 DEV2: cont_h2.est vs A3 (arms_dev2.json estimates and cont_h.est_h fresh); MSE x C."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cont_h, cont_h2  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "../../oracle_floor"))
from panel import dev2, ROOT  # noqa: E402
tr = json.load(open(os.path.join(ROOT, "data/oracle_floor/dev2_truths.json")))["truths"]
arms = json.load(open(os.path.join(HERE, "arms_dev2.json")))["per_model"]; rows = {}
for n, m in dev2():
    a, fa = cont_h.est_h(m, -1.0, True); b, fb = cont_h2.est(m)
    rows[n] = dict(truth=tr[n]["acc"], A3=a, A3_saved=arms[n]["A3_spread_tilt"]["est"], T353=b, C_A3=fa / 5948, C=fb / 5948, A0=arms[n]["A0_T317f"]["est"], C_A0=arms[n]["A0_T317f"]["C"])
    print(n, "diff %.3g  C %.0f (A3 %.0f)" % (b - a, fb / 5948, fa / 5948), flush=True)
N = list(rows); p = np.array([rows[n]["truth"] for n in N]); pq = float((p * (1 - p)).mean())
def st(k, ck):
    e = 100 * (np.array([rows[n][k] for n in N]) - p); C = float(np.mean([rows[n][ck] for n in N])); mse = float((e ** 2).mean())
    return dict(bias=float(e.mean()), sd=float(e.std(ddof=1)), rms=float(np.sqrt(mse)), C=C, mse_x_C=mse * C, F=float(mse / 1e4 * max(C, 1000) / pq))
d = np.array([abs(rows[n]["T353"] - rows[n]["A3"]) for n in N])
S = dict(T353=st("T353", "C"), A3=st("A3", "C_A3"), T317f=st("A0", "C_A0"), max_abs_diff=float(d.max()), n_bitwise=int((d == 0).sum()),
         saved_vs_fresh_A3_max=float(max(abs(rows[n]["A3"] - rows[n]["A3_saved"]) for n in N)))
S["gate"] = bool(d.max() <= 1e-9); S["go_line"] = 0.9 * 2276.27; S["go"] = bool(S["gate"] and S["T353"]["mse_x_C"] <= S["go_line"])
json.dump(dict(per_model=rows, summary=S), open(os.path.join(HERE, "t353_dev2.json"), "w"), indent=1)
for k, v in S.items():
    print(k, v)
