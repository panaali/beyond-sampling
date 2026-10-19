"""T394: train a bias-free 10-128-10 ReLU MLP on 2nd-argmax with T345's stated recipe (written from its README).

usage: python train.py SEED [LOG2_NTRAIN=28]   -> models/s<SEED>.json
"""
import json
import math
import sys
import time

import torch as th

T, H, BATCH = 10, 128, 1024


def main(seed, log2n=28):
    th.manual_seed(seed)
    th.set_num_threads(1)
    net = th.nn.Sequential(th.nn.Linear(T, H, bias=False), th.nn.ReLU(), th.nn.Linear(H, T, bias=False))
    opt = th.optim.AdamW(net.parameters(), lr=0.04 / math.sqrt(H), weight_decay=0.001)
    steps, warm, floor = 2 ** log2n // BATCH, 2 ** 14, 0.01

    def mult(s):  # linear warm-up, then exponential decay reaching `floor` at half the run, held there
        return min(s / warm, 1.0) * max(floor ** (2 * s / steps), floor)
    sched = th.optim.lr_scheduler.LambdaLR(opt, mult)
    t0, hist, acc_sum = time.time(), [], 0.0
    for s in range(steps):
        x = th.randn(BATCH, T)
        y = x.argsort(-1)[:, -2]
        out = net(x)
        loss = th.nn.functional.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc_sum += (out.argmax(-1) == y).float().mean().item()
        if (s + 1) % 16384 == 0:
            hist.append([s + 1, acc_sum / 16384, time.time() - t0]); acc_sum = 0.0
            print(*hist[-1], flush=True)
    W1, W2 = (m.weight.detach().double().tolist() for m in (net[0], net[2]))
    json.dump({"seed": seed, "n_train": 2 ** log2n, "W1": W1, "W2": W2, "log": hist}, open(f"models/s{seed}.json", "w"))


if __name__ == "__main__":
    main(int(sys.argv[1]), *(int(a) for a in sys.argv[2:3]))
