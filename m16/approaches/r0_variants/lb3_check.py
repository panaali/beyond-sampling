"""Out-of-panel check (post hoc): the same paired closure biases on the 30 LB3 test models (32M-sample truths),
500k inputs per model, seed 298500 + index. Diagnostic only."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/lb3/m16")
import leaderboard as L
import importlib.util
spec = importlib.util.spec_from_file_location("r", "run.py")
src = open("run.py").read().split("REG = (")[0].split("z = np.load")[0]
exec(src)                                          # imports + predict/closures definitions only
exec("def predict" + open("run.py").read().split("def predict")[1].split("REG = (")[0])
EXPLORE = False
T = 10
ms = L.models()
tr = json.load(open("/home/user/lb3/" + L.truths_path(16, 10)))
REG = ("blind", "band23", "band12", "tail")
res = {}
for k, (name, model) in enumerate(ms.items()):
    rng = np.random.default_rng(298500 + k)
    sums = {}
    for c in range(0, 500_000, 100_000):
        x = rng.standard_normal((100_000, 10))
        for EXPLORE in (False, True):
            cl, tgt, reg = closures(x)
            ok = (predict(model, x) == tgt).astype(float)
            for cn, pts in cl.items():
                d = np.mean([predict(model, p) == tgt for p in pts], 0) - ok
                sums[cn] = sums.get(cn, 0.0) + d.sum()
    res[name] = {cn: 100 * s / 500_000 for cn, s in sums.items()}
    res[name]["truth_pp"] = 100 * tr[name]["acc"]
    print(name, " ".join(f"{cn}:{v:+.3f}" for cn, v in res[name].items() if cn != "truth_pp"), flush=True)
json.dump(res, open("results/lb3_check.json", "w"), indent=1)
names = list(res); acc = np.array([res[n]["truth_pp"] for n in names])
for cn in [c for c in res[names[0]] if c != "truth_pp"]:
    b = np.array([res[n][cn] for n in names])
    print(f"{cn}: RMS {np.sqrt(np.mean(b**2)):.3f} mean {b.mean():+.3f} slope {np.polyfit(acc, b, 1)[0]:+.4f} max {np.abs(b).max():.2f}")
