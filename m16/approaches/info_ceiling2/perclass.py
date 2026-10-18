"""T314: per-class cheap quantities (<= ~2.5k FPE) for the 80 DEV2 models + 5 LB3 seeds. Diagnostic."""
import json
import sys
import numpy as np
sys.path.insert(0, "/home/user/t218/m16/approaches/r0_integral")
import r0  # noqa: E402
import tb  # noqa: E402
TRI = r0.TRIPLES; n = len(TRI); rr = np.arange(n)
V2 = tb.MF["q"](0.5); V1 = V2 + tb.MF["g12"]; V3M = V2 - tb.MF["g23"]
Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
models = [(str(nm), (Z["Wi"][i].astype(float), Z["Wh"][i].astype(float), Z["Wo"][i].astype(float))) for i, nm in enumerate(Z["names"])]
truth = {k: v["acc"] for k, v in json.load(open("/home/user/dl/dev2/truths.json"))["truths"].items()}
sys.path.insert(0, "/home/user/lb3/m16")
import leaderboard as LB  # noqa: E402
lbm = LB.models(); lbt = json.load(open("/home/user/lb3/" + LB.truths_path(16, 10)))
for s in range(5):
    models.append((f"seed{s}", lbm[f"seed{s}"])); truth[f"seed{s}"] = lbt[f"seed{s}"]["acc"]
pr = {}
for path in ("/home/user/dl/t286/sweep_raw_dev2.jsonl", "/home/user/dl/t286/sweep_raw.jsonl"):
    for line in open(path):
        d = json.loads(line)
        if d["fam"] == "probe" and d["cfg"] == {"version": 1, "iters": 8, "stride": 1}:
            pr[d["model"]] = d["est"]
names, P, S = [], [], []
for name, model in models:
    acc_tb, flops, parts = tb.est(model, n23=1, newton=1, nblind=0, use12=False, return_parts=True)
    m_tie, dm_tie = tb._margin(model, TRI, np.full(n, V1), np.full(n, V2), np.full(n, V2 - 1e-6),
                               np.zeros(n), -np.ones(n), -tb.dmu(np.full(n, V2 - 1e-6)))
    x = r0.r0_inputs(TRI, np.full(n, V1), np.full(n, V2), np.full(n, V3M))
    L = r0.logits(model, x); L2 = L.copy(); L2[rr, TRI[:, 1]] = -np.inf
    m_med = L[rr, TRI[:, 1]] - L2.max(1)
    names.append(name)
    P.append(np.stack([m_tie, dm_tie, parts["t23"][0], parts["M"], m_med]))
    S.append([acc_tb, truth[name], pr.get(name, np.nan), flops / 5948 + n])
    print(name, round(acc_tb, 4), round(truth[name], 4), flush=True)
np.savez("results/perclass.npz", names=np.array(names), P=np.array(P), S=np.array(S),
         S_cols=np.array(["tb", "truth", "probe1", "fpe"]))
