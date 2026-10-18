"""T347 test on all 80 DEV2: D_pred (tilt.py, frozen) vs measured D_m = V1 - J1 (budget.py results, V1 convention)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import tilt  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "../../oracle_floor"))
from panel import dev2  # noqa: E402
res = json.load(open(os.path.join(HERE, "../results.json"))); out = {}
for name, m in dev2():
    d, info = tilt.D_pred(m); Dm = 100 * (res[name]["V1"]["mean"] - res[name]["off_product"]["mean"])
    out[name] = dict(D_pred=d, D_meas=Dm, **info); print(name, "pred %+.3f meas %+.3f" % (d, Dm), {k: round(v, 3) for k, v in info.items()}, flush=True)
P = np.array([v["D_pred"] for v in out.values()]); M = np.array([v["D_meas"] for v in out.values()])
S = dict(mean_abs_err=float(np.abs(P - M).mean()), corr=float(np.corrcoef(P, M)[0, 1]), pred_mean=float(P.mean()), meas_mean=float(M.mean()),
         pred_sd=float(P.std(ddof=1)), meas_sd=float(M.std(ddof=1)), const_oracle_mae=float(np.abs(M - M.mean()).mean()), go=bool(np.abs(P - M).mean() <= .1))
print(S); json.dump(dict(per_model=out, summary=S), open(os.path.join(HERE, "dev2.json"), "w"), indent=1)
