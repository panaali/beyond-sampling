"""T391 development (DEV2 models 40-79, NOT used to fit the table): frozen T10 recipe vs isg, one deterministic run per
model, MSE ratio vs sampling at executed cost, cluster bootstrap. Also wall time per call."""
import importlib.util, json, sys, time
import numpy as np
D = np.load("../../../data/dev2/weights.npz"); tr = json.load(open("../../../data/dev2/truths.json"))["truths"]
names = [str(n) for n in D["names"]]
def load(path, nm):
    spec = importlib.util.spec_from_file_location(nm, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
ests = {"frozen": load("estimator.py", "e0"), "isg": load("isg/estimator.py", "e1")}
for b in (16000, 64000):
    for k, E in ests.items():
        rows = []
        for i in range(40, 80):
            m = (D["Wi"][i], D["Wh"][i], D["Wo"][i]); t0 = time.time(); e, c = E.est(m, b); wt = time.time() - t0
            rows.append((e, c, tr[names[i]]["acc"], tr[names[i]]["se"], wt))
        R = np.array(rows); se2 = (R[:, 0] - R[:, 2]) ** 2 - R[:, 3] ** 2; v = R[:, 2] * (1 - R[:, 2]) / R[:, 1]
        rng = np.random.default_rng(0); bs = [se2[ii].mean() / v[ii].mean() for ii in (rng.integers(0, 40, 40) for _ in range(2000))]
        print(f"{k:6s} {b}: MSE ratio {se2.mean()/v.mean():.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]  cost {R[:,1].mean():.0f}  wall {R[:,4].mean():.2f}s/call", flush=True)
