"""T192 wind-down (DRI): high-precision MC truth for the target M_16,10 (alg_zoo.example_2nd_argmax, x ~ N(0, I_10)).

Blocks of 1e9 samples; block b uses its own torch CUDA Philox generator seeded SEED0 + b (counter-based, independent
streams). Inputs and forward pass in float64 (a CPU test showed float32 flips ~1 in 2e6 samples, a ~5e-7 bias larger
than the target SE ~2e-7), counts in int64; every block count is appended to the log (resumable). Correct = argmax
of the logits equals the index of the 2nd largest input. Block 0 is also recomputed in float32 on the same inputs. usage: python mc_target.py n_blocks [device] -> results/mc_target/blocks.jsonl"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SEED0 = 20261018000
BLOCK = 10 ** 9
CHUNK = 1 << 24


def model(dtype, dev):
    from alg_zoo import example_2nd_argmax
    m = example_2nd_argmax().eval()
    Wi = m.rnn.weight_ih_l0.detach()[:, 0].to(dtype=dtype, device=dev)
    Wh = m.rnn.weight_hh_l0.detach().to(dtype=dtype, device=dev)
    Wo = m.linear.weight.detach().to(dtype=dtype, device=dev)
    return Wi, Wh, Wo


@torch.no_grad()
def block(b, W64, W32=None, dev="cuda", n=BLOCK):
    g = torch.Generator(device=dev); g.manual_seed(SEED0 + b)
    ok32 = 0; ok64 = 0; done = 0
    while done < n:
        m = min(CHUNK, n - done)
        x = torch.randn(m, 10, generator=g, device=dev, dtype=torch.float64)
        lab = x.topk(2, 1).indices[:, 1]
        for W, dt in ((W64, torch.float64),) + (((W32, torch.float32),) if W32 else ()):
            Wi, Wh, Wo = W; xx = x.to(dt); h = torch.zeros(m, 16, dtype=dt, device=dev)
            for t in range(10):
                h = torch.relu(h @ Wh.T + xx[:, t:t + 1] * Wi[None])
            c = int(((h @ Wo.T).argmax(1) == lab).sum())
            if dt == torch.float64:
                ok64 += c
            else:
                ok32 += c
        done += m
    return ok64, (ok32 if W32 else None)


if __name__ == "__main__":
    nb = int(sys.argv[1]); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    n = BLOCK if len(sys.argv) <= 3 else int(float(sys.argv[3]))           # block size override for tests
    out = os.path.join(HERE, "results", "mc_target"); os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "blocks.jsonl" if n == BLOCK else f"blocks_n{n}.jsonl")
    have = {json.loads(l)["block"] for l in open(log)} if os.path.exists(log) else set()
    W32 = model(torch.float32, dev); W64 = model(torch.float64, dev)
    for b in range(nb):
        if b in have:
            continue
        t0 = time.time(); c64, c32 = block(b, W64, W32 if b == 0 else None, dev, n)
        r = dict(block=b, seed=SEED0 + b, n=n, correct=c64, correct_float32=c32, sec=time.time() - t0)
        with open(log, "a") as f:
            f.write(json.dumps(r) + "\n")
    rows = [json.loads(l) for l in open(log)]
    N = sum(r["n"] for r in rows); K = sum(r["correct"] for r in rows); p = K / N
    acc = [r["correct"] / r["n"] for r in rows]
    s = dict(blocks=len(rows), n=N, correct=K, acc=p, se=math.sqrt(p * (1 - p) / N),
             between_block_sd_of_mean=float(np.std(acc, ddof=1) / math.sqrt(len(acc))) if len(acc) > 1 else None,
             float32_minus_float64_block0=(rows[0]["correct_float32"] - rows[0]["correct"]) if rows else None)
    json.dump(s, open(os.path.join(out, "summary.json" if n == BLOCK else f"summary_n{n}.json"), "w"), indent=1)
    print(json.dumps(s))
