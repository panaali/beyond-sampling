"""Record MLP neuron activations (input to down_proj) of Qwen3-0.6B over pile-10k windows.

Stores, per layer, 32 random neurons plus 4 random unit directions in the 3072-d MLP hidden space.
Resumable: progress is kept in data/acts/progress.json.
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
CTX, PER_DOC, N_WIN, BS = 64, 4, 18750, 16
N_NEURONS, N_DIRS = 32, 4
OUT = "data/acts"

torch.set_num_threads(os.cpu_count())
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()
n_layers, d_mlp = model.config.num_hidden_layers, model.config.intermediate_size

rng = np.random.default_rng(0)
neuron_idx = np.stack([rng.choice(d_mlp, N_NEURONS, replace=False) for _ in range(n_layers)])
dirs = rng.standard_normal((n_layers, d_mlp, N_DIRS)).astype(np.float32)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

if not os.path.exists(f"{OUT}/tokens.npy"):
    texts = pq.read_table(sys.argv[1] if len(sys.argv) > 1 else
                          "data/pile10k/data/train-00000-of-00001-4746b8785c874cc7.parquet")["text"].to_pylist()
    wins, doc_of = [], []
    for d, t in enumerate(texts):
        ids = tok(t[:20000])["input_ids"]
        for k in range(min(PER_DOC, len(ids) // CTX)):
            wins.append(ids[k * CTX:(k + 1) * CTX]); doc_of.append(d)
        if len(wins) >= N_WIN:
            break
    np.save(f"{OUT}/tokens.npy", np.array(wins[:N_WIN], dtype=np.int32))
    np.save(f"{OUT}/doc_of.npy", np.array(doc_of[:N_WIN], dtype=np.int32))
    np.save(f"{OUT}/neuron_idx.npy", neuron_idx)
    np.save(f"{OUT}/dirs.npy", dirs)
tokens = np.load(f"{OUT}/tokens.npy")
assert (np.load(f"{OUT}/neuron_idx.npy") == neuron_idx).all()

shape = (len(tokens), CTX, n_layers, N_NEURONS + N_DIRS)
mode = "r+" if os.path.exists(f"{OUT}/acts.f16") else "w+"
acts = np.memmap(f"{OUT}/acts.f16", dtype=np.float16, mode=mode, shape=shape)
json.dump({"shape": shape}, open(f"{OUT}/meta.json", "w"))
prog = f"{OUT}/progress.json"
start = json.load(open(prog))["done"] if os.path.exists(prog) else 0

buf = {}
idx_t = torch.tensor(neuron_idx)
dirs_t = torch.tensor(dirs)


def hook(layer):
    def f(mod, inp):
        h = inp[0]
        buf[layer] = torch.cat([h[..., idx_t[layer]], h @ dirs_t[layer]], -1)
    return f


for i, layer in enumerate(model.layers):
    layer.mlp.down_proj.register_forward_pre_hook(hook(i))

for b in range(start, len(tokens), BS):
    with torch.inference_mode():
        model(torch.tensor(tokens[b:b + BS], dtype=torch.long))
    acts[b:b + BS] = torch.stack([buf[i] for i in range(n_layers)], 2).numpy().astype(np.float16)
    if (b // BS) % 25 == 0 or b + BS >= len(tokens):
        acts.flush()
        json.dump({"done": min(b + BS, len(tokens))}, open(prog, "w"))
        print(f"{min(b + BS, len(tokens))}/{len(tokens)}", flush=True)
