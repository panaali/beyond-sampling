"""T276: HP1 truths. float64 GPU MC, n samples per model, independent CUDA Philox stream per model
(seed 2760000 + model index), no common random numbers. -> sealed/truths.json. usage: python truth_hp1.py [n]"""
import json, math, os, sys, time
import numpy as np, torch as th
HERE = os.path.dirname(os.path.abspath(__file__))
n = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10 ** 8
z = np.load(os.path.join(HERE, "weights.npz")); names = list(z["names"]); out = {}; t0 = time.time()
for k, nm in enumerate(names):
    Wi, Wh, Wo = (th.as_tensor(z[a][k], dtype=th.float64, device="cuda") for a in ("Wi", "Wh", "Wo"))
    g = th.Generator(device="cuda"); g.manual_seed(2760000 + k); ok = 0; done = 0
    while done < n:
        m = min(1 << 24, n - done); x = th.randn(m, 10, generator=g, device="cuda", dtype=th.float64)
        h = th.zeros(m, 16, dtype=th.float64, device="cuda")
        for t in range(10):
            h = th.relu(h @ Wh.T + x[:, t:t + 1] * Wi[None])
        ok += int(((h @ Wo.T).argmax(1) == x.topk(2, 1).indices[:, 1]).sum()); done += m
    p = ok / n; out[nm] = dict(acc=p, se=math.sqrt(p * (1 - p) / n), n=n, correct=ok, mc_seed=2760000 + k)
os.makedirs(os.path.join(HERE, "sealed"), exist_ok=True)
json.dump(dict(panel="HP1", n_per_model=n, sec=time.time() - t0, truths=out), open(os.path.join(HERE, "sealed", "truths.json"), "w"), indent=1)
print("done", len(out), "models in %.0fs" % (time.time() - t0))
