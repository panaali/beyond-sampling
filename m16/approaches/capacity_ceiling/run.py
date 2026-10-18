"""T234 driver: train K seeds of the (H, T) 2nd-argmax DistRNN with zoo settings (or a variant), checkpoint, evaluate.

usage: python run.py TAG --H 16 --T 10 --seeds 0 1 2 3 4 --log2n 30 [--lr-scale 1] [--warmup 16384] [--floor 0.01]
                     [--f-blind 0] [--f-tie 0] [--gap 0.05] [--eval-n 1e8] [--device cuda]
-> data/capacity_ceiling/TAG/ (weights npz per seed, checkpoints at n_train / 2^i, i = 0..4) and
   results/TAG.json (config, wall time, training curve, final MC accuracy + error anatomy per seed)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trainer as TR  # noqa: E402
from evaluate import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("--H", type=int, default=16); ap.add_argument("--T", type=int, default=10)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); ap.add_argument("--log2n", type=float, default=30)
ap.add_argument("--lr-scale", type=float, default=1.0); ap.add_argument("--warmup", type=float, default=2 ** 14)
ap.add_argument("--floor", type=float, default=0.01); ap.add_argument("--f-blind", type=float, default=0.0)
ap.add_argument("--f-tie", type=float, default=0.0); ap.add_argument("--gap", type=float, default=0.05)
ap.add_argument("--eval-n", type=float, default=1e8); ap.add_argument("--device", default="cuda")
ap.add_argument("--data-seed", type=int, default=0)
a = ap.parse_args()

n_train = int(2 ** a.log2n); K = len(a.seeds)
out_dir = os.path.join(HERE, "..", "..", "..", "data", "capacity_ceiling", a.tag); os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
tr = TR.Trainer(a.H, a.T, a.seeds, n_train, warmup=a.warmup, device=a.device, f_blind=a.f_blind, f_tie=a.f_tie,
                gap=a.gap, data_seed=a.data_seed, lr_scale=[a.lr_scale] * K, floor=a.floor)
n_steps = n_train // 1024
ckpt = {n_steps // 2 ** i for i in range(5)}
curve = []; t0 = time.time(); lsum = th.zeros(K, device=a.device); asum = th.zeros(K, device=a.device); nlog = 0
import hashlib  # noqa: E402
BLOCK = 2 ** 18                                   # data reseed + state checkpoint interval (steps)
src_hash = hashlib.sha256(b"".join(open(os.path.join(HERE, f), "rb").read() for f in ("trainer.py", "run.py"))).hexdigest()
meta = dict(config=vars(a), seeds=a.seeds, src_hash=src_hash)
state_f = os.path.join(out_dir, "state.pt"); start = 1
if os.path.exists(state_f):                       # resume after a reclaimed runtime (only at block boundaries)
    st = th.load(state_f, weights_only=False)
    if st.get("meta") != meta:
        raise SystemExit(f"{a.tag}: checkpoint config/source/seeds differ from this run; refusing to resume")
    tr.load_state(st, a.data_seed); start = int(st["step"]) + 1; curve = st["curve"]
    print(a.tag, "resumed at step", start - 1, flush=True)
tr.reseed(a.data_seed, (start - 1) // BLOCK)
for s in range(start, n_steps + 1):
    tr.step(); lsum += tr.loss; asum += tr.acc; nlog += 1
    if s % 2 ** 14 == 0 or s == n_steps:
        curve.append(dict(step=s, sec=time.time() - t0, loss=(lsum / nlog).tolist(), acc=(asum / nlog).tolist()))
        lsum.zero_(); asum.zero_(); nlog = 0
        print(a.tag, s, "%.0fs" % (time.time() - t0), ["%.4f" % v for v in curve[-1]["acc"]], flush=True)
    if s % BLOCK == 0 or s == n_steps:
        th.save(tr.state() | {"curve": curve, "meta": meta}, state_f + ".tmp"); os.replace(state_f + ".tmp", state_f)
        if s % BLOCK == 0:
            tr.reseed(a.data_seed, s // BLOCK)
    if s in ckpt:
        W = tr.weights()
        for k, seed in enumerate(a.seeds):
            np.savez(os.path.join(out_dir, f"seed{seed}_n{s * 1024}.npz"), Wi=W[0][k], Wh=W[1][k], Wo=W[2][k])
wall = time.time() - t0
W = tr.weights(); models = [(W[0][k], W[1][k], W[2][k]) for k in range(K)]
ev = evaluate(models, int(a.eval_n), device=a.device)
res = dict(config=vars(a), src_hash=src_hash, n_train=n_train, n_steps=n_steps, wall_sec=wall, curve=curve,
           eval={str(seed): e for seed, e in zip(a.seeds, ev)})
json.dump(res, open(os.path.join(HERE, "results", f"{a.tag}.json"), "w"), indent=1)
os.makedirs(os.path.join(HERE, "weights"), exist_ok=True)              # final weights also kept in the repo folder
for k, seed in enumerate(a.seeds):
    np.savez(os.path.join(HERE, "weights", f"{a.tag}_seed{seed}.npz"), Wi=W[0][k], Wh=W[1][k], Wo=W[2][k])
for seed, e in zip(a.seeds, ev):
    print(a.tag, "seed", seed, "acc %.5f +- %.5f  blind %.4f  2-3 %.4f  1-2 %.4f  other %.4f" % (
        e["acc"], e["se"], e["err_blind"], e["err_23"], e["err_12"], e["err_other"]))
