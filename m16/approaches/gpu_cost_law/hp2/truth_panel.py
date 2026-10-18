"""T284: panel truths (as hp1/truth_hp1.py). float64 GPU MC, n samples per model, independent CUDA Philox stream per
model (seed BASE + model index; dev2 2840000, hp2 2841000, hp3 2842000), no common random numbers.
usage: python truth_panel.py dev2|hp2 [n] -> dev2/truths.json (open) | sealed/truths.json"""
import hashlib, json, math, os, sys, time
import numpy as np, torch as th
HERE = os.path.dirname(os.path.abspath(__file__)); which = sys.argv[1]
n = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10 ** 8
d = {"dev2": os.path.join(HERE, "dev2"), "hp2": HERE, "hp3": os.path.join(HERE, "..", "hp3")}[which]
base = {"dev2": 2840000, "hp2": 2841000, "hp3": 2842000}[which]
z = np.load(os.path.join(d, "weights.npz")); names = list(z["names"]); out = {}; t0 = time.time()
for k, nm in enumerate(names):
    Wi, Wh, Wo = (th.as_tensor(z[a][k], dtype=th.float64, device="cuda") for a in ("Wi", "Wh", "Wo"))
    g = th.Generator(device="cuda"); g.manual_seed(base + k); ok = 0; done = 0
    while done < n:
        m = min(1 << 24, n - done); x = th.randn(m, 10, generator=g, device="cuda", dtype=th.float64)
        h = th.zeros(m, 16, dtype=th.float64, device="cuda")
        for t in range(10):
            h = th.relu(h @ Wh.T + x[:, t:t + 1] * Wi[None])
        ok += int(((h @ Wo.T).argmax(1) == x.topk(2, 1).indices[:, 1]).sum()); done += m
    p = ok / n; out[str(nm)] = dict(acc=p, se=math.sqrt(p * (1 - p) / n), n=n, correct=ok, mc_seed=base + k)
f = os.path.join(d, "truths.json") if which == "dev2" else os.path.join(d, "sealed", "truths.json")
os.makedirs(os.path.dirname(f), exist_ok=True)
json.dump(dict(panel=which.upper(), n_per_model=n, sec=time.time() - t0, truths=out), open(f, "w"), indent=1)
a = np.array([v["acc"] for v in out.values()])
print("done", len(out), "models in %.0fs; truths sha256" % (time.time() - t0), hashlib.sha256(open(f, "rb").read()).hexdigest(),
      "min %.5f max %.5f mean %.5f sd %.5f" % (a.min(), a.max(), a.mean(), a.std(ddof=1)))
