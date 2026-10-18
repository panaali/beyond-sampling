"""T287 on DEV2 (80 open retrains; T192 branch 611796bd gpu_cost_law/hp2/dev2/weights.npz, read from git, not copied into the
repo): E1 estimates for D0..D3 (deterministic) + provisional 2e6-sample MC truths (rng 2870+i) until the panel's open truths.json
lands. F at C_eff = max(C, 1k) over all 80 models. usage (repo root): python m16/approaches/frozen_cheap/dev2.py WEIGHTS.npz"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
src = open(os.path.join(HERE, "cheap.py")).read().split("models = lb.models(); out = {}")[0]
sys.argv = [sys.argv[0], "1"] + sys.argv[1:]; g = {"__file__": os.path.join(HERE, "cheap.py")}; exec(src, g)
np = g["np"]; CL = g["CL"]; verdicts = g["verdicts"]; run = g["run"]
z = np.load(sys.argv[2], allow_pickle=False); out = {}
for i, name in enumerate(z["names"]):
    m = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float)); r = {}
    for d, c in CL.items():
        W = np.array([x[0] for x in c]); Xw = np.array([x[1] for x in c]); top = [x[2] for x in c]
        r[f"E1_{d}"] = dict(est=float((W * verdicts(m, Xw, top)).sum()), cost_fpe=len(c))
    rng = np.random.default_rng(2870 + i); ok = 0
    for _ in range(4):
        X = rng.standard_normal((500_000, 10)); ok += int((run(m, X).argmax(1) == np.argsort(X, 1)[:, -2]).sum())
    p = ok / 2e6; r["truth_mc"] = dict(acc=p, se=float(np.sqrt(p * (1 - p) / 2e6)))
    out[str(name)] = r; print(name, round(p, 4), {k: round(100 * (v["est"] - p), 2) for k, v in r.items() if k.startswith("E1")}, flush=True)
json.dump(out, open(os.path.join(HERE, "dev2.json"), "w"), indent=1)
for d in ("E1_D0", "E1_D1", "E1_D2", "E1_D3"):
    e2 = np.mean([(o[d]["est"] - o["truth_mc"]["acc"]) ** 2 - o["truth_mc"]["se"] ** 2 for o in out.values()])
    v = np.mean([o["truth_mc"]["acc"] * (1 - o["truth_mc"]["acc"]) for o in out.values()]); C = max(out[next(iter(out))][d]["cost_fpe"], 1000)
    print(d, "DEV2 rms %.2f pp  F %.3g" % (100 * np.sqrt(max(e2, 0)), e2 * C / v))
