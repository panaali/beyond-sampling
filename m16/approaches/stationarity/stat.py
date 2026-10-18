"""T310 (pre-registered in README): readout-gradient stationarity and the calibration identity on DEV2 (diagnostic sampling).
usage (repo root): python m16/approaches/stationarity/stat.py DEV2_WEIGHTS.npz DEV2_TRUTHS.json -> stat.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
T, B, NB, WD = 10, 1024, 4096, 0.001
z = np.load(sys.argv[1], allow_pickle=False); TR = json.load(open(sys.argv[2]))["truths"]; out = {}
for i, name in enumerate(map(str, z["names"])):
    Wi, Wh, Wo = z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float)
    rng = np.random.default_rng(310); Gb = np.zeros((NB, 10, 16)); Db = np.zeros(NB); Lb = np.zeros(NB); Ab = np.zeros(NB)
    for c in range(0, NB, 64):
        X = rng.standard_normal((64 * B, T)); y = np.argsort(X, 1)[:, -2]; h = np.zeros((len(X), 16))
        for t in range(T):
            h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
        L = h @ Wo.T; L = L - L.max(1, keepdims=True); p = np.exp(L); p /= p.sum(1, keepdims=True)
        E = p.copy(); E[np.arange(len(y)), y] -= 1
        Gb[c:c + 64] = np.einsum("bnk,bnj->bkj", E.reshape(64, B, 10), h.reshape(64, B, 16)) / B
        Ly = L[np.arange(len(y)), y]
        Db[c:c + 64] = ((p * L).sum(1) - Ly).reshape(64, B).mean(1); Lb[c:c + 64] = (-np.log(p[np.arange(len(y)), y])).reshape(64, B).mean(1)
        Ab[c:c + 64] = (L.argmax(1) == y).reshape(64, B).mean(1)
    G = Gb.mean(0); s = np.linalg.norm(Gb, axis=(1, 2)).mean(); se = np.sqrt(((Gb - G) ** 2).sum((1, 2)).mean() / NB)
    rms = np.sqrt((Gb ** 2).mean(0)); pred = WD * np.linalg.norm(Wo * rms)
    out[name] = dict(G_norm=float(np.linalg.norm(G)), G_se=float(se), single_batch=float(s), rho=float(np.linalg.norm(G) / s),
                     adamw_pred=float(pred), D=float(Db.mean()), D_se=float(Db.std() / np.sqrt(NB)), D_check=float((G * Wo).sum()),
                     loss=float(Lb.mean()), acc_mc=float(Ab.mean()), acc=TR[name]["acc"], kernel=name.split("_")[0])
    print(name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "stat.json"), "w"), indent=1)
a = np.array([v["acc"] for v in out.values()]); summ = {}
for key in ("D", "loss", "rho", "G_norm"):
    x = np.array([v[key] for v in out.values()]); summ[f"corr_{key}_acc"] = float(np.corrcoef(x, a)[0, 1])
    for kn in ("fused", "torch"):
        sel = [v["kernel"] == kn for v in out.values()]; summ[f"corr_{key}_acc_{kn}"] = float(np.corrcoef(x[sel], a[sel])[0, 1])
summ["median_rho"] = float(np.median([v["rho"] for v in out.values()])); summ["median_G_over_se"] = float(np.median([v["G_norm"] / v["G_se"] for v in out.values()]))
summ["median_G_over_adamw_pred"] = float(np.median([v["G_norm"] / v["adamw_pred"] for v in out.values()]))
summ["stop"] = bool(summ["median_rho"] > 0.10 or abs(summ["corr_D_acc"]) < 0.5)
print(summ); json.dump(summ, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
