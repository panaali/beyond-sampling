"""Zoo recipe (alg_zoo.training.train, reproduced) for any T432 candidate.

usage: python train.py TAG SEED LR_BASE LOG2_NTRAIN [OUTDIR [NAME]]
Writes OUTDIR/<TAG>_s<SEED>.pt (state_dict) and OUTDIR/<TAG>_s<SEED>.json (val accuracy on 2^20, wall, config).
"""
import functools
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch as th

th.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import archs  # noqa: E402


def lr_mult_schedule(step, *, n_steps, warmup_steps, decay_min_mult):
    warmup = min(step / warmup_steps, 1)
    decay = max(decay_min_mult ** (2 * step / n_steps), decay_min_mult)
    return warmup * decay


def train(tag, seed, lr_base=0.04, n_train=2 ** 30, batch=1024, wd=1e-3, warmup=2 ** 14, n_val=2 ** 20, log=None):
    th.manual_seed(seed)
    model = archs.build(tag)
    opt = th.optim.AdamW(model.parameters(), lr=lr_base / archs.WIDTH[tag] ** 0.5, weight_decay=wd)
    n_steps = n_train // batch
    sched = th.optim.lr_scheduler.LambdaLR(opt, functools.partial(
        lr_mult_schedule, n_steps=n_steps, warmup_steps=warmup, decay_min_mult=0.01))
    t0 = time.time(); acc_run = 0.0
    for step in range(n_steps):
        x = th.randn(batch, archs.T)
        y = x.argsort(-1)[..., -2]
        logits = model(x)
        loss = th.nn.functional.cross_entropy(logits, y)
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        if log is not None and (step + 1) % 8192 == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(json.dumps({"step": step + 1, "loss": loss.item(), "acc": acc, "s": time.time() - t0}), file=log, flush=True)
    wall = time.time() - t0
    with th.no_grad():
        ok = 0
        for _ in range(n_val // batch):
            x = th.randn(batch, archs.T)
            ok += (model(x).argmax(-1) == x.argsort(-1)[..., -2]).sum().item()
    return model, ok / n_val, wall


if __name__ == "__main__":
    tag, seed, lr_base, lg = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(HERE, "weights")
    os.makedirs(out, exist_ok=True)
    name = sys.argv[6] if len(sys.argv) > 6 else f"{tag}_s{seed}" + ("" if lg == 30 else f"_lr{lr_base:g}_n{lg}")
    with open(os.path.join(out, name + ".log"), "w") as log:
        model, val, wall = train(tag, seed, lr_base, 2 ** lg, log=log)
    th.save(model.state_dict(), os.path.join(out, name + ".pt"))
    json.dump({"tag": tag, "seed": seed, "lr_base": lr_base, "n_train": 2 ** lg, "val_acc_2^20": val, "wall": wall,
               "params": archs.nparams(model), "torch": th.__version__,
               "threads": th.get_num_threads()}, open(os.path.join(out, name + ".json"), "w"), indent=1)
    print(name, val, wall)
