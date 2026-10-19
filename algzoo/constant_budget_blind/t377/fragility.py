"""T377: model-intrinsic readout fragility, measured without any investigator (frozen before any T377 run).

For each model, on 200,000 fresh N(0,1) sequences (seed 37701):
  S1 = mean over neurons i of [model acc - acc after replacing h_10[i] by h_10[i] + e_i], e_i ~ N(0, 0.1 var(h_i))
       (noise at the level of an R^2 = 0.9 formula), averaged over 3 noise draws.
  F  = gap recovered, (acc - 0.1)/(model acc - 0.1), when ALL neurons get such noise at once, averaged over 3 draws.
  F80 = the same at R^2 = 0.8 (noise var 0.2 var(h_i)).
Usage: python fragility.py  -> t377/fragility.json
"""
import json
from pathlib import Path

import numpy as np
import torch
from alg_zoo import example_2nd_argmax, zoo_2nd_argmax

torch.set_num_threads(1)
MODELS = {
    "M8_s4(default)": lambda: zoo_2nd_argmax(8, 10),
    "M16_s2(default)": lambda: zoo_2nd_argmax(16, 10),
    "M16_s0": lambda: zoo_2nd_argmax(16, 10, seed=0),
    "M16_s1": lambda: zoo_2nd_argmax(16, 10, seed=1),
    "M16_example": example_2nd_argmax,
    "M32_s0(default)": lambda: zoo_2nd_argmax(32, 10),
    "M32_s1": lambda: zoo_2nd_argmax(32, 10, seed=1),
    "M32_s2": lambda: zoo_2nd_argmax(32, 10, seed=2),
}

rng = np.random.default_rng(37701)
x = rng.standard_normal((200_000, 10)).astype(np.float32)
target = np.argsort(x, -1)[:, -2]
out = {}
for name, load in MODELS.items():
    m = load().eval()
    with torch.no_grad():
        _, h = m.rnn(torch.tensor(x).unsqueeze(-1))
    h = h[0].numpy().astype(np.float64)
    W = m.linear.weight.detach().numpy().astype(np.float64)
    acc = lambda hh: float(((hh @ W.T).argmax(-1) == target).mean())
    a0 = acc(h)
    sd = h.std(0)
    res = {"H": h.shape[1], "model_acc": a0}
    for tag, frac in (("90", 0.1), ("80", 0.2)):
        single, full = [], []
        for d in range(3):
            noise = rng.standard_normal(h.shape) * sd * np.sqrt(frac)
            full.append((acc(h + noise) - 0.1) / (a0 - 0.1))
            if tag == "90":
                drops = []
                for i in range(h.shape[1]):
                    hh = h.copy(); hh[:, i] += noise[:, i]
                    drops.append(a0 - acc(hh))
                single.append(drops)
        res[f"F{tag}"] = float(np.mean(full))
        if single:
            per = np.mean(single, 0)
            res["S1"] = float(per.mean())
            res["S1_per_neuron"] = [float(v) for v in per]
    out[name] = res
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "S1_per_neuron"},
          flush=True)
json.dump(out, open(Path(__file__).parent / "fragility.json", "w"), indent=1)
