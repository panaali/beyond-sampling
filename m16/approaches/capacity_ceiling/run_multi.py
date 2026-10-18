"""T234: many stacked models in one process (one batch size, per-model n_train / lr scale / warm-up), e.g. the
large-batch RECIPE VARIANTS (batch 8192 / 16384, lr scaled by sqrt(B/1024) or B/1024, warm-up = 1/64 of the run as in
the zoo, zoo schedule otherwise). Deterministic data (reseeded per 2^18-step block), resumable state with validated
metadata, final-state checkpoint, per-model weights, 1e8-sample paired evaluation.

usage: python run_multi.py TAG --batch 8192 --spec 'name:seed:log2n:lr_scale[:warmup_steps]' ... [--device cuda]
(warm-up default = 1/64 of the run; zoo settings = fixed 16384 steps)
-> results/TAG.json, weights/TAG_<name>.npz, data/capacity_ceiling/TAG/"""
import argparse
import hashlib
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
ap.add_argument("tag"); ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--H", type=int, default=16)
ap.add_argument("--T", type=int, default=10); ap.add_argument("--spec", nargs="+", required=True)
ap.add_argument("--floor", type=float, default=0.01); ap.add_argument("--f-blind", type=float, default=0.0)
ap.add_argument("--f-tie", type=float, default=0.0); ap.add_argument("--gap", type=float, default=0.05)
ap.add_argument("--eval-n", type=float, default=1e8); ap.add_argument("--device", default="cuda")
ap.add_argument("--data-seed", type=int, default=0)
a = ap.parse_args()

spec = []
for s in a.spec:
    f = s.split(":"); name, seed, log2n, lrs = f[:4]
    n_train = int(2 ** float(log2n)); n_steps = n_train // a.batch
    warm = int(f[4]) if len(f) > 4 else max(1, n_steps // 64)       # zoo: 2^14 of 2^20 steps = 1/64 of the run
    spec.append(dict(name=name, seed=int(seed), n_train=n_train, n_steps=n_steps, lr_scale=float(lrs), warmup=warm))
K = len(spec); n_max = max(s["n_steps"] for s in spec)
out_dir = os.path.join(HERE, "..", "..", "..", "data", "capacity_ceiling", a.tag); os.makedirs(out_dir, exist_ok=True)
for d in ("results", "weights"):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)
tr = TR.Trainer(a.H, a.T, [s["seed"] for s in spec], [s["n_train"] for s in spec], warmup=[s["warmup"] for s in spec],
                batch=a.batch, device=a.device, f_blind=a.f_blind, f_tie=a.f_tie, gap=a.gap, data_seed=a.data_seed,
                lr_scale=[s["lr_scale"] for s in spec], floor=a.floor)
BLOCK = 2 ** 18 if n_max >= 2 ** 18 else 2 ** 14
src_hash = hashlib.sha256(b"".join(open(os.path.join(HERE, f), "rb").read() for f in ("trainer.py", "run_multi.py"))).hexdigest()
meta = dict(config=vars(a), spec=spec, src_hash=src_hash)
state_f = os.path.join(out_dir, "state.pt"); start = 1; curve = []
if os.path.exists(state_f):
    st = th.load(state_f, weights_only=False)
    if st.get("meta") != meta:
        raise SystemExit(f"{a.tag}: checkpoint metadata differ; refusing to resume")
    tr.load_state(st, a.data_seed); start = int(st["step"]) + 1; curve = st["curve"]
    print(a.tag, "resumed at step", start - 1, flush=True)
tr.reseed(a.data_seed, (start - 1) // BLOCK)
t0 = time.time(); lsum = th.zeros(K, device=a.device); asum = th.zeros(K, device=a.device); nlog = 0
done_models = set()
for s in range(start, n_max + 1):
    tr.step(); lsum += tr.loss; asum += tr.acc; nlog += 1
    if s % 2 ** 12 == 0 or s == n_max:
        curve.append(dict(step=s, sec=time.time() - t0, loss=(lsum / nlog).tolist(), acc=(asum / nlog).tolist()))
        lsum.zero_(); asum.zero_(); nlog = 0
        if s % 2 ** 14 == 0 or s == n_max:
            print(a.tag, s, "%.0fs" % (time.time() - t0), "%.3f ms/step" % (1e3 * (time.time() - t0) / (s - start + 1)),
                  ["%.4f" % v for v in curve[-1]["acc"]], flush=True)
    if s % BLOCK == 0 or s == n_max:
        th.save(tr.state() | {"curve": curve, "meta": meta}, state_f + ".tmp"); os.replace(state_f + ".tmp", state_f)
        if s % BLOCK == 0:
            tr.reseed(a.data_seed, s // BLOCK)
    for k, sp in enumerate(spec):                                   # save each model when its own run ends
        if s == sp["n_steps"] and k not in done_models:
            W = tr.weights(); done_models.add(k)
            np.savez(os.path.join(HERE, "weights", f"{a.tag}_{sp['name']}.npz"), Wi=W[0][k], Wh=W[1][k], Wo=W[2][k])
wall = time.time() - t0
W = tr.weights()
ev = evaluate([(W[0][k], W[1][k], W[2][k]) for k in range(K)], int(a.eval_n), device=a.device)
res = dict(config=vars(a), spec=spec, src_hash=src_hash, batch=a.batch, wall_sec=wall, ms_per_step=1e3 * wall / max(1, n_max - start + 1),
           curve=curve, eval={sp["name"]: e for sp, e in zip(spec, ev)},
           note="large-batch runs are a RECIPE VARIANT (not zoo settings)" if a.batch != 1024 else "zoo batch")
json.dump(res, open(os.path.join(HERE, "results", f"{a.tag}.json"), "w"), indent=1)
for sp, e in zip(spec, ev):
    print(a.tag, sp["name"], "acc %.5f +- %.5f blind %.4f 2-3 %.4f 1-2 %.4f other %.4f" % (
        e["acc"], e["se"], e["err_blind"], e["err_23"], e["err_12"], e["err_other"]), flush=True)
