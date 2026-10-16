"""Score neuron labels for AlgZoo M_16,10 (pre-registered addendum E4b).

algzoo/labels.py must define LABELS = {i: (category, text)} and f(i, x) -> np.ndarray, where x is an
(N, 10) float array of inputs, returning the predicted h_10[:, i]. It must not use the model's weights.
"""
import importlib.util
import json
import re
import sys

import numpy as np
import torch
from alg_zoo import example_2nd_argmax

path = sys.argv[1] if len(sys.argv) > 1 else "algzoo/labels.py"
src = open(path).read()
banned = re.findall(r"alg_zoo|weight_hh|weight_ih|state_dict|\.pth|torch", src)
assert not banned, f"labels file touches the model: {banned}"
spec = importlib.util.spec_from_file_location("labels", path)
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)

model = example_2nd_argmax().eval()
rng = np.random.default_rng(20261016)  # held-out seed, never given to the investigator
x = rng.standard_normal((200_000, 10)).astype(np.float32)
with torch.no_grad():
    xt = torch.tensor(x)
    _, h = model.rnn(xt.unsqueeze(-1))
    h = h[0].numpy()  # (N, 16) final hidden state
    logits = model(xt).numpy()
W_oh = model.linear.weight.detach().numpy()
target = np.argsort(x, -1)[:, -2]
acc = lambda lg: float((lg.argmax(-1) == target).mean())

F = np.stack([np.asarray(L.f(i, x.copy()), dtype=np.float64) for i in range(16)], 1)
r2 = 1 - ((F - h) ** 2).sum(0) / ((h - h.mean(0)) ** 2).sum(0)
one_at_a_time = []
for i in range(16):
    hh = h.copy(); hh[:, i] = F[:, i]
    one_at_a_time.append(acc(hh @ W_oh.T))

model_acc, repl_acc = acc(logits), acc(F @ W_oh.T)
cats = {i: L.LABELS[i][0] for i in range(16)}
legible = [i for i in range(16) if cats[i] in "AB" and r2[i] >= 0.8]
out = {
    "model_acc": model_acc, "chance": 0.1, "replacement_acc": repl_acc,
    "gap_recovered": (repl_acc - 0.1) / (model_acc - 0.1),
    "n_legible": len(legible), "categories": cats,
    "r2": [float(v) for v in r2], "acc_replacing_one_neuron": one_at_a_time,
    "verdict_legible": len(legible) >= 12,
    "verdict_mechanism": repl_acc >= 0.1 + 0.9 * (model_acc - 0.1),
}
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
