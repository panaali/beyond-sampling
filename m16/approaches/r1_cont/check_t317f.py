"""T354 item (1): is T317f (the sealed HP4 submission) affected by the simultaneous-flip bug?
(a) model-free: does any input coordinate of T317f's lines cross 0 inside the window [0, tmax = .5]?
(b) per model: T317f's walk (cont_f's tables, its walk arithmetic = cont_e.walk_cont, without the declared C1 stop)
    vs a dense float64 forward along every line (2,000 midpoints on [0, .5]); and the fixed walk's count of
    simultaneous flips. 10 DEV2 models (every 8th) and 10 DEV3 models (s500..s509). -> results_check_t317f.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
import cont_f as F  # noqa: E402
import walkfix as WF  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
import r0  # noqa: E402

out = {"zero_crossings": {}, "models": {}}
# (a) the lines are model-free and identical in T317e / T317f
for tb, tf in zip(E.TABLES, F.TABLES):
    for kind in ("23", "12"):
        if f"line{kind}" not in tb:
            continue
        x, dx = tf[f"line{kind}"]; assert np.array_equal(x, tb[f"line{kind}"][0]) and np.array_equal(dx, tb[f"line{kind}"][1])
        with np.errstate(divide="ignore", invalid="ignore"):
            tz = -x / dx
        hit = (dx != 0) & (tz > 0) & (tz <= E.TMAX)
        out["zero_crossings"][f"{tb['name']}-{kind}"] = dict(rows_with_crossing=int(hit.any(1).sum()), min_root=float(np.nanmin(np.where(dx != 0, np.where(tz > 0, tz, np.inf), np.inf))))
print("zero crossings in [0, .5]:", out["zero_crossings"], flush=True)

z = np.load(os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz"))
models = [(nm, m) for i, (nm, m) in enumerate(r0.dev2()) if i % 8 == 0] + \
         [(str(z["names"][i]), (z["Wi"][i].astype(np.float64), z["Wh"][i].astype(np.float64), z["Wo"][i].astype(np.float64))) for i in range(10)]
ts = (np.arange(2000) + 0.5) * (E.TMAX / 2000); dt = E.TMAX / 2000
for nm, m in models:
    m = tuple(np.asarray(w, np.float64) for w in m); Wi, Wh, Wo = m; k = E.TRI[:, 1]; rec = {}
    for tb in E.TABLES:
        for kind in ("23", "12"):
            if f"line{kind}" not in tb:
                continue
            x, dx = tb[f"line{kind}"]
            p, _ = E.walk_cont(m, x, dx, k, E.TMAX)
            _, _, _, _, _, nsim = WF.walk3_fixed(E, m, x, dx, k, k, E.TMAX)
            wl = np.zeros(len(x))
            for r, lo, hi in p:
                wl[r] += np.maximum(hi - lo, 0)
            dl = np.zeros(len(x))
            for i0 in range(0, len(x), 60):
                X = x[i0:i0 + 60, None, :] + ts[None, :, None] * dx[i0:i0 + 60, None, :]; X = X.reshape(-1, 10); h = np.zeros((len(X), 16))
                for s in range(10):
                    h = np.maximum(h @ Wh.T + np.outer(X[:, s], Wi), 0)
                am = (h @ Wo.T).argmax(1).reshape(-1, len(ts)); dl[i0:i0 + 60] = (am == k[i0:i0 + 60, None]).sum(1) * dt
            rec[f"{tb['name']}-{kind}"] = dict(max_abs_diff=float(np.abs(wl - dl).max()), mean_abs_diff=float(np.abs(wl - dl).mean()), simultaneous_flips=nsim)
    out["models"][nm] = rec
    print(nm, {l: (round(v["max_abs_diff"], 5), v["simultaneous_flips"]) for l, v in rec.items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "results_check_t317f.json"), "w"), indent=1)
mx = max(v["max_abs_diff"] for r in out["models"].values() for v in r.values()); ns = sum(v["simultaneous_flips"] for r in out["models"].values() for v in r.values())
print("SUMMARY: max |walk - dense| correct length over all lines and models = %.2e (grid dt %.1e); simultaneous flips = %d" % (mx, dt, ns))
