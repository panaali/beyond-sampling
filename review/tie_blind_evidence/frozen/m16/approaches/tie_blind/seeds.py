"""T223 control: MSE over 16 independent rule scrambles (rule seeds = node-set choices, not model inputs) for sci_tao
and tie_blind on 6 models at 16k and 64k FPE. ratio = mean_seeds (est - truth)^2 / (p(1-p)/cost), cost charged as on
the leaderboard (sci_tao x2.23 at scoring; tie_blind charges 2.23 inside). -> seeds.json"""
import json, os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path[:0] = [os.path.join(ROOT, "m16"), HERE]
import numpy as np, leaderboard as lb
def load(p):
    s = importlib.util.spec_from_file_location(os.path.basename(os.path.dirname(p)), p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ST, TB = load("m16/approaches/sci_tao/estimator.py"), load("m16/approaches/tie_blind/estimator.py")
tr = json.load(open("results/leaderboard_truths.json")); ms = lb.models(); out = {}
for name in ("orig", "seed0", "s0.0025_0", "s0.005_0", "s0.01_0", "s0.02_0"):
    p = tr[name]["acc"]; v = p * (1 - p)
    for b in (16000, 64000):
        r = {}
        for meth, mod, fac in (("sci_tao", ST, 2.23), ("tie_blind", TB, 1.0)):
            e2 = []; cs = []
            for s in range(16):
                mod.RULE_SEED = 5000 + 10 * s
                if meth == "tie_blind": mod.BLIND_SEED = 7000 + s
                e, c = mod.est(ms[name], b); e2.append((e - p) ** 2 - tr[name]["se"] ** 2); cs.append(fac * c)
            r[meth] = dict(ratio=float(np.mean(e2) / (v / np.mean(cs))), cost=float(np.mean(cs)), rmse_pp=100 * float(np.sqrt(max(np.mean(e2), 0))))
        out[f"{name}@{b}"] = r; print(name, b, {k: round(x["ratio"], 2) for k, x in r.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "seeds.json"), "w"), indent=1)
