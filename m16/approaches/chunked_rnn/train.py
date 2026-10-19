"""Train a chunked-input bias-free ReLU RNN on alg_zoo's 2nd_argmax task (T392).

x ~ N(0, I_10), label = x.argsort(-1)[..., -2] (the position of the 2nd-largest input, as alg_zoo/tasks.py).
The RNN reads k consecutive inputs per step for S = 10 / k steps: h_0 = 0, h_t = relu(W_ih x[(t-1)k : tk] + W_hh h_{t-1}),
logits = W_out h_S (Linear(H, 10), bias-free). k = 1 is alg_zoo's DistRNN; k = 10 is a 1-hidden-layer MLP.

Recipe = alg_zoo.training.train: AdamW (weight decay .001), lr .04 / sqrt(H), 2^14 warm-up steps, exponential decay to
.01x, batch 1024, cross-entropy, n_train = 2^30 fresh samples, torch.manual_seed(seed).

usage: python train.py K H SEED [LOG2_NTRAIN]   -> models/k{K}h{H}_s{SEED}.json
"""
import functools
import json
import os
import sys
import time

import torch as th

T = 10
HERE = os.path.dirname(os.path.abspath(__file__))


class ChunkRNN(th.nn.Module):
    def __init__(self, k, H):
        super().__init__()
        self.k = k
        self.rnn = th.nn.RNN(input_size=k, hidden_size=H, nonlinearity="relu", bias=False, batch_first=True)
        self.linear = th.nn.Linear(H, T, bias=False)

    def forward(self, x):
        _, h = self.rnn(x.view(len(x), T // self.k, self.k))
        return self.linear(h.squeeze(0))


def lr_mult(step, *, n_steps, warmup_steps, decay_min_mult):          # = alg_zoo.training.lr_mult_schedule
    return min(step / warmup_steps, 1) * max(decay_min_mult ** (2 * step / n_steps), decay_min_mult)


def main(k, H, seed, log2n=30, batch=1024):
    th.manual_seed(seed)
    th.set_num_threads(1)
    model = ChunkRNN(k, H)
    opt = th.optim.AdamW(model.parameters(), lr=0.04 / H ** 0.5, weight_decay=0.001)
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
    W = {"Wih": model.rnn.weight_ih_l0.detach().double().tolist(), "Whh": model.rnn.weight_hh_l0.detach().double().tolist(),
         "Wout": model.linear.weight.detach().double().tolist()}
    tag = f"k{k}h{H}_s{seed}{'' if log2n == 30 else f'_n{log2n}'}"
    json.dump({"k": k, "H": H, "seed": seed, "n_train": 2 ** log2n, "weights": W, "log": log},
              open(f"{HERE}/models/{tag}.json", "w"))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
