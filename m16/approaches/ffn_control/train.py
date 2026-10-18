"""Train bias-free ReLU MLPs on alg_zoo's 2nd_argmax task (x ~ N(0,1)^10, label = x.argsort(-1)[..., -2]).

Recipe copied from alg_zoo.training.train (AdamW, weight decay .001, lr .04/sqrt(width), 2**14 warmup steps,
exponential decay to .01x, batch 1024, cross-entropy on the 10 logits, bias=False). Only n_train is smaller by
default (2**28 instead of 2**30) to fit a shared CPU.

usage: python train.py ARCH SEED [LOG2_NTRAIN]     ARCH in {L1h22, L1h43, L2h13, L2h21}
"""
import functools
import json
import sys
import time

import torch as th

ARCHS = {"L1h22": [22], "L1h43": [43], "L2h13": [13, 13], "L2h21": [21, 21]}
T = 10


def make(widths):
    layers, d = [], T
    for w in widths:
        layers += [th.nn.Linear(d, w, bias=False), th.nn.ReLU()]
        d = w
    layers.append(th.nn.Linear(d, T, bias=False))
    return th.nn.Sequential(*layers)


def lr_mult(step, *, n_steps, warmup_steps, decay_min_mult):          # = alg_zoo.training.lr_mult_schedule
    return min(step / warmup_steps, 1) * max(decay_min_mult ** (2 * step / n_steps), decay_min_mult)


def main(arch, seed, log2n=28, batch=1024):
    th.manual_seed(seed)
    th.set_num_threads(1)
    widths = ARCHS[arch]
    model = make(widths)
    n_params = sum(p.numel() for p in model.parameters())
    opt = th.optim.AdamW(model.parameters(), lr=0.04 / widths[0] ** 0.5, weight_decay=0.001)
    n_steps = 2 ** log2n // batch
    sched = th.optim.lr_scheduler.LambdaLR(opt, functools.partial(
        lr_mult, n_steps=n_steps, warmup_steps=2 ** 14, decay_min_mult=0.01))
    t0, run_loss, run_acc, log = time.time(), 0.0, 0.0, []
    for step in range(n_steps):
        x = th.randn(batch, T)
        y = x.argsort(-1)[..., -2]
        logits = model(x)
        loss = th.nn.functional.cross_entropy(logits, y)
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        run_loss += loss.item(); run_acc += (logits.argmax(-1) == y).float().mean().item()
        if (step + 1) % 8192 == 0:
            log.append((step + 1, run_loss / 8192, run_acc / 8192, time.time() - t0))
            print(*log[-1], flush=True); run_loss = run_acc = 0.0
    W = [m.weight.detach().double().tolist() for m in model if isinstance(m, th.nn.Linear)]
    json.dump({"arch": arch, "widths": widths, "seed": seed, "n_params": n_params, "n_train": 2 ** log2n,
               "weights": W, "log": log}, open(f"models/{arch}_s{seed}.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *(int(a) for a in sys.argv[3:4]))
