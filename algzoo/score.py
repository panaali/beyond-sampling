"""Score neuron labels for an AlgZoo 2nd-argmax RNN (pre-registered addendum E4b/E4c).

The labels file must define LABELS = {i: (category, text)} and f(i, x) -> np.ndarray, where x is an
(N, seq_len) float array of inputs, returning the predicted final hidden state [:, i]. It must not use the model's weights.
"""
import argparse
import importlib.util
import json
import re
import sys

import numpy as np
import torch
from alg_zoo import example_2nd_argmax, zoo_2nd_argmax

ap = argparse.ArgumentParser()
ap.add_argument("labels", nargs="?", default="algzoo/labels.py")
ap.add_argument("out", nargs="?")
ap.add_argument("--zoo", help="hidden,seq_len of a zoo 2nd-argmax model (default: the 432-param example)")
args = ap.parse_args()
path = args.labels
src = open(path).read()
banned = re.findall(r"alg_zoo|weight_hh|weight_ih|state_dict|\.pth|torch", src)
assert not banned, f"labels file touches the model: {banned}"
spec = importlib.util.spec_from_file_location("labels", path)
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)

model = (zoo_2nd_argmax(*map(int, args.zoo.split(","))) if args.zoo else example_2nd_argmax()).eval()
H, T = model.rnn.hidden_size, model.linear.out_features
rng = np.random.default_rng(20261016)  # held-out seed, never given to the investigator
x = rng.standard_normal((200_000, T)).astype(np.float32)
with torch.no_grad():
    xt = torch.tensor(x)
    _, h = model.rnn(xt.unsqueeze(-1))
    h = h[0].numpy()  # (N, H) final hidden state
    logits = model(xt).numpy()
W_oh = model.linear.weight.detach().numpy()
target = np.argsort(x, -1)[:, -2]
acc = lambda lg: float((lg.argmax(-1) == target).mean())

F = np.stack([np.asarray(L.f(i, x.copy()), dtype=np.float64) for i in range(H)], 1)
sst = ((h - h.mean(0)) ** 2).sum(0)
sse = ((F - h) ** 2).sum(0)
# dead neurons (no variance) count as explained only if predicted exactly
r2 = np.where(sst > 1e-9, 1 - sse / np.maximum(sst, 1e-9), (sse < 1e-6).astype(float))
one_at_a_time = []
for i in range(H):
    hh = h.copy(); hh[:, i] = F[:, i]
    one_at_a_time.append(acc(hh @ W_oh.T))

model_acc, repl_acc, chance = acc(logits), acc(F @ W_oh.T), 1 / T
cats = {i: L.LABELS[i][0] for i in range(H)}
legible = [i for i in range(H) if cats[i] in "AB" and r2[i] >= 0.8]
out = {
    "model_acc": model_acc, "chance": chance, "replacement_acc": repl_acc,
    "gap_recovered": (repl_acc - chance) / (model_acc - chance),
    "n_legible": len(legible), "categories": cats,
    "r2": [float(v) for v in r2], "acc_replacing_one_neuron": one_at_a_time,
    "n_neurons": H, "verdict_legible": len(legible) >= 0.75 * H,
    "verdict_mechanism": repl_acc >= chance + 0.9 * (model_acc - chance),
}
print(json.dumps(out, indent=1))
if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
