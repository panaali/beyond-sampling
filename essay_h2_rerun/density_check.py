"""T6 post-hoc validity check of the controls (pre-registered): per-neuron contribution shares on the 12
strongest shown examples of every dense and Gaussian control item, as in ../review/R1/q3.py.
Writes essay_h2_rerun/density_check.json.  usage: python density_check.py [keys_dir]
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModel

A, S = "data/t6", "essay_h2_rerun"
K = sys.argv[1] if len(sys.argv) > 1 else "data/t6/keys"
torch.set_num_threads(int(os.environ.get("T6_THREADS", "4")))
tokens = np.load(f"{A}/tokens.npy")
D = np.concatenate([np.load(f"{A}/gdirs.npy"), np.load(f"{A}/ddirs.npy")], 2)
items = [it for it in json.load(open(f"{K}/items.json")) if it["kind"] != "neuron"]
model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="sdpa").eval()
buf = {}
for i, layer in enumerate(model.layers):
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, inp, i=i: buf.__setitem__(i, inp[0].detach()))
res = []
for n, it in enumerate(items):
    L, s = it["layer"], it["sign"]
    d = D[L][:, it["unit"] - 32]
    ex = it["show_top"][:12]
    with torch.inference_mode():
        model(torch.tensor(tokens[[e[0] for e in ex]], dtype=torch.long))
    h = buf[L].numpy()[np.arange(len(ex)), [e[1] for e in ex]]
    c = s * h * d[None]
    proj = c.sum(1)
    res.append({"id": it["id"], "kind": it["kind"], "layer": L,
                "top1": float(np.median(c.max(1) / proj)), "top5": float(np.median(np.sort(c, 1)[:, -5:].sum(1) / proj)),
                "proj_rel_err": float(np.max(np.abs(proj - np.array([e[2] for e in ex])) / it["max"]))})
    if n % 25 == 0:
        print(n, res[-1], flush=True)
out = {"items": res}
for k in ("dense", "gauss"):
    r = [x for x in res if x["kind"] == k]
    out[f"median_top1_{k}"] = float(np.median([x["top1"] for x in r]))
    out[f"median_top5_{k}"] = float(np.median([x["top5"] for x in r]))
    out[f"frac_top1_le_0.15_{k}"] = float(np.mean([x["top1"] <= 0.15 for x in r]))
out["valid"] = out["median_top1_dense"] <= 0.15
json.dump(out, open(f"{S}/density_check.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "items"})
