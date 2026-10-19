"""T6: record Qwen3-0.6B MLP activations (input to down_proj) on the original E1 corpus windows.

Same model, corpus, windows and seed-0 neuron/direction draw as ../collect.py (the original H2), plus new
"dense" control directions chosen by a mechanical density rule on a pilot (see README, pre-registration).

usage:
  python collect_t6.py pilot   -> data/t6/pilot.json (per-neuron scale stats, candidate density table)
  python collect_t6.py main    -> data/t6/L{layer:02d}.f16, shape (U, N_WIN*CTX), U = 32 neurons + 4 Gaussian
                                  dirs + 12 dense dirs; resumable via data/t6/progress.json
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
CTX, PER_DOC, N_WIN, BS, SKIP = 64, 4, 18750, 16, 4
N_NEURONS, N_DIRS, N_DENSE = 32, 4, 12
N_PILOT, N_CAND, TOPK = 512, 8, 40
OUT = "data/t6"
PARQUET = "data/pile10k/data/train-00000-of-00001-4746b8785c874cc7.parquet"

torch.set_num_threads(int(os.environ.get("T6_THREADS", "4")))
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()
n_layers, d_mlp = model.config.num_hidden_layers, model.config.intermediate_size

# identical to ../collect.py: same rng stream, same order of draws
rng = np.random.default_rng(0)
neuron_idx = np.stack([rng.choice(d_mlp, N_NEURONS, replace=False) for _ in range(n_layers)])
gdirs = rng.standard_normal((n_layers, d_mlp, N_DIRS)).astype(np.float32)
gdirs /= np.linalg.norm(gdirs, axis=1, keepdims=True)

if not os.path.exists(f"{OUT}/tokens.npy"):
    texts = pq.read_table(PARQUET)["text"].to_pylist()
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
    np.save(f"{OUT}/gdirs.npy", gdirs)
tokens = np.load(f"{OUT}/tokens.npy")
buf = {}


def hook(layer):
    def f(mod, inp):
        buf[layer] = inp[0].detach()
    return f


for i, layer in enumerate(model.layers):
    layer.mlp.down_proj.register_forward_pre_hook(hook(i))


def forward(w):
    with torch.inference_mode():
        model(torch.tensor(tokens[w], dtype=torch.long))


def candidates(stats, seed, n=N_CAND):
    """Candidate control constructions, n directions per layer each, shape (L, d_mlp, n)."""
    r = np.random.default_rng(seed)
    g = r.standard_normal((n_layers, d_mlp, n)).astype(np.float32)
    sd, mx = np.array(stats["sd"])[..., None], np.array(stats["absmax"])[..., None]
    out = {"G": g, "S": g / sd, "M": g / mx}
    px = g.copy()  # top 1% of neurons by absolute max projected out
    sx = g / sd    # sd-balanced, top 5% by max/sd projected out
    for L in range(n_layers):
        px[L, np.argsort(-mx[L, :, 0])[:d_mlp // 100]] = 0
        sx[L, np.argsort(-(mx[L, :, 0] / sd[L, :, 0]))[:d_mlp // 20]] = 0
    out["PX"], out["SX"] = px, sx
    # amendment 1 (README): max-normalised with the most kurtotic neurons (top q by max/sd) projected out
    for q in (5, 10, 20):
        mxq = g / mx
        for L in range(n_layers):
            mxq[L, np.argsort(-(mx[L, :, 0] / sd[L, :, 0]))[:d_mlp * q // 100]] = 0
        out[f"MX{q}"] = mxq
    return {k: v / np.linalg.norm(v, axis=1, keepdims=True) for k, v in out.items()}


def pilot():
    if os.path.exists(f"{OUT}/pilot.json"):  # amendment 1: reuse the stats, add only new candidates
        old = json.load(open(f"{OUT}/pilot.json"))
        return pilot_pass2(old["stats"], old["n_tokens"], old["table"])
    s1 = np.zeros((n_layers, d_mlp)); s2 = np.zeros((n_layers, d_mlp)); mx = np.zeros((n_layers, d_mlp)); n = 0
    for b in range(0, N_PILOT, BS):
        forward(slice(b, b + BS))
        for L in range(n_layers):
            h = buf[L][:, SKIP:].reshape(-1, d_mlp).double().numpy()
            s1[L] += h.sum(0); s2[L] += (h ** 2).sum(0); mx[L] = np.maximum(mx[L], np.abs(h).max(0))
        n += BS * (CTX - SKIP)
    sd = np.sqrt(s2 / n - (s1 / n) ** 2)
    stats = {"sd": sd.tolist(), "absmax": mx.tolist()}
    pilot_pass2(stats, n, {})


def pilot_pass2(stats, n, table):
    cands = {k: v for k, v in candidates(stats, 10).items() if k not in table}
    # second pilot pass: top-TOPK positions per (candidate, layer, direction, pole) and their neuron shares
    best = {k: [[[[] for _ in range(2)] for _ in range(N_CAND)] for _ in range(n_layers)] for k in cands}
    for b in range(0, N_PILOT, BS):
        forward(slice(b, b + BS))
        for L in range(n_layers):
            h = buf[L][:, SKIP:].reshape(-1, d_mlp).numpy()
            for k, D in cands.items():
                proj = h @ D[L]
                for j in range(N_CAND):
                    for si, s in enumerate((1, -1)):
                        v = s * proj[:, j]
                        top = np.argsort(-v)[:TOPK]
                        c = s * h[top] * D[L][:, j][None]
                        sh1 = c.max(1) / v[top]
                        sh5 = np.sort(c, 1)[:, -5:].sum(1) / v[top]
                        lst = best[k][L][j][si] + list(zip(v[top].tolist(), sh1.tolist(), sh5.tolist()))
                        best[k][L][j][si] = sorted(lst, reverse=True)[:TOPK]
    for k in cands:
        e = np.array([x for L in best[k] for j in L for p in j for x in p])
        per_unit = np.array([[np.median([x[1] for x in p]), np.median([x[2] for x in p])]
                             for L in best[k] for j in L for p in j])
        table[k] = {"median_top1_share": float(np.median(per_unit[:, 0])),
                    "median_top5_share": float(np.median(per_unit[:, 1])),
                    "pooled_median_top1": float(np.median(e[:, 1]))}
        print(k, table[k], flush=True)
    json.dump({"stats": stats, "table": table, "n_tokens": n}, open(f"{OUT}/pilot.json", "w"))


def main():
    pil = json.load(open(f"{OUT}/pilot.json"))
    choice = json.load(open("essay_h2_rerun/control_choice.json"))["construction"]
    ddirs = candidates(pil["stats"], 20, N_DENSE)[choice].astype(np.float32)  # fresh seed, not the pilot dirs
    np.save(f"{OUT}/ddirs.npy", ddirs)
    D = torch.tensor(np.concatenate([gdirs, ddirs], 2))
    idx = torch.tensor(neuron_idx)
    U, T = N_NEURONS + N_DIRS + N_DENSE, len(tokens) * CTX
    for L in range(n_layers):  # files of shape (U, T) float16, written by seek (memmaps exceed ulimit -v)
        fn = f"{OUT}/L{L:02d}.f16"
        if not os.path.exists(fn) or os.path.getsize(fn) != U * T * 2:
            with open(fn, "wb") as f:
                f.truncate(U * T * 2)
    json.dump({"U": U, "T": T, "CTX": CTX, "construction": choice}, open(f"{OUT}/meta.json", "w"))
    prog = f"{OUT}/progress.json"
    start = json.load(open(prog))["done"] if os.path.exists(prog) else 0
    chunk, c0 = [], start

    def flush(end):
        a = np.concatenate(chunk, 1)  # (L, rows, U)
        for L in range(n_layers):
            with open(f"{OUT}/L{L:02d}.f16", "r+b") as f:
                for u in range(U):
                    f.seek((u * T + c0 * CTX) * 2)
                    f.write(np.ascontiguousarray(a[L, :, u]).tobytes())
        json.dump({"done": end}, open(prog, "w"))
        print(f"{end}/{len(tokens)}", flush=True)

    for b in range(start, len(tokens), BS):
        forward(slice(b, b + BS))
        chunk.append(np.stack([torch.cat([buf[L].reshape(-1, d_mlp)[:, idx[L]], buf[L].reshape(-1, d_mlp) @ D[L]], 1)
                               .numpy().astype(np.float16) for L in range(n_layers)]))
        end = min(b + BS, len(tokens))
        if len(chunk) == 50 or end == len(tokens):
            flush(end)
            chunk, c0 = [], end


{"pilot": pilot, "main": main}[sys.argv[1]]()
