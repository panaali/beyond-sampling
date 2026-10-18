"""T234 interim: evaluate the FINISHED grid3 models in the synced state.pt (data/capacity_ceiling/grid3/state.pt) on CPU,
float64, N samples per model (common random inputs across models: paired comparisons; se per model ~ sqrt(p(1-p)/N)).
Only models with n_steps <= state step. Labelled INTERIM: the in-kernel 1e8 evaluation at grid3's end is the record.
usage: python interim_eval.py [N] -> results/grid3_interim.json"""
import json, os, sys
import numpy as np, torch as th
HERE = os.path.dirname(os.path.abspath(__file__)); N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10 ** 6
st = th.load(os.path.join(HERE, "..", "..", "..", "data", "capacity_ceiling", "grid3", "state.pt"), weights_only=False)
spec = st["meta"]["spec"]; idx = [k for k, sp in enumerate(spec) if sp["n_steps"] <= st["step"]]
Wi = st["Wi"][idx].double().numpy(); Wh = st["Wh"][idx].double().numpy(); Wo = st["Wo"][idx][:, :10].double().numpy()
rng = np.random.default_rng(20261018); ok = np.zeros(len(idx)); done = 0
while done < N:
    m = min(2000, N - done); x = rng.standard_normal((m, 10)); y = np.argsort(x, 1)[:, -2]
    h = np.zeros((len(idx), m, 16))
    for t in range(10):
        h = np.maximum(np.einsum("kij,kmj->kmi", Wh, h) + x[None, :, t:t + 1] * Wi[:, None, :], 0)
    ok += (np.einsum("kij,kmj->kmi", Wo, h).argmax(2) == y[None]).sum(1); done += m
acc = ok / N; out = {"step": int(st["step"]), "N": N, "models": {spec[k]["name"]: float(a) for k, a in zip(idx, acc)}}
groups = {}
for k, a in zip(idx, acc):
    groups.setdefault(spec[k]["name"].rsplit("_s", 1)[0], []).append(a)
rng2 = np.random.default_rng(0); out["groups"] = {}
for g, v in sorted(groups.items()):
    v = np.array(v); bm = [rng2.choice(v, len(v)).mean() for _ in range(2000)]
    out["groups"][g] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)), max=float(v.max()),
                            top5_mean=float(np.sort(v)[-5:].mean()), mean_ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))])
    print(g, out["groups"][g], flush=True)
json.dump(out, open(os.path.join(HERE, "results", "grid3_interim.json"), "w"), indent=1)
