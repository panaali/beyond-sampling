"""T234: train stacked models with the fused Triton kernel in chunks of CHUNK steps per launch, checkpoint all model
states between chunks (resumable, metadata validated), then evaluate (evaluate.py, 1e8 paired samples).
usage: python fused_run.py TAG --spec 'name:seed:log2n:lr_scale[:warmup]' ... [--data-seed 0] [--eval-n 1e8]
-> results/TAG.json, weights/TAG_<name>.npz, data/capacity_ceiling/TAG/state.pt"""
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
import fused as F  # noqa: E402
from evaluate import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("--spec", nargs="+", required=True); ap.add_argument("--data-seed", type=int, default=0)
ap.add_argument("--eval-n", type=float, default=1e8); ap.add_argument("--chunk", type=int, default=4096)
a = ap.parse_args()
spec = []
for s in a.spec:
    f = s.split(":"); n_steps = int(2 ** float(f[2])) // 1024
    spec.append(dict(name=f[0], seed=int(f[1]), n_train=int(2 ** float(f[2])), n_steps=n_steps, lr_scale=float(f[3]),
                     warmup=int(f[4]) if len(f) > 4 else 2 ** 14))
M = len(spec); n_max = max(s["n_steps"] for s in spec)
out_dir = os.path.join(HERE, "..", "..", "..", "data", "capacity_ceiling", a.tag); os.makedirs(out_dir, exist_ok=True)
for d in ("results", "weights"):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)
ft = F.FusedTrainer([s["seed"] for s in spec], [s["n_train"] for s in spec], lr_scale=[s["lr_scale"] for s in spec],
                    warmup=[s["warmup"] for s in spec], data_seed=a.data_seed)
src = hashlib.sha256(b"".join(open(os.path.join(HERE, f), "rb").read() for f in ("fused.py", "fused_run.py"))).hexdigest()
meta = dict(config=vars(a), spec=spec, src_hash=src)
names = ["Wi", "Wh", "Wo"]
state_f = os.path.join(out_dir, "state.pt"); curve = []
if os.path.exists(state_f):
    st = th.load(state_f, weights_only=False)
    if st["meta"] != meta:
        raise SystemExit("checkpoint metadata differ; refusing to resume")
    for i, nm in enumerate(names):
        getattr(ft, nm).copy_(st[nm]); ft.m[i].copy_(st["m"][i]); ft.v[i].copy_(st["v"][i])
    ft.step_n = st["step"]; curve = st["curve"]; print(a.tag, "resumed at", ft.step_n, flush=True)
t0 = time.time(); s0 = ft.step_n
while ft.step_n < n_max:
    n = min(a.chunk, n_max - ft.step_n); ft.lossacc.zero_(); ft.step(n); th.cuda.synchronize()
    curve.append(dict(step=ft.step_n, sec=time.time() - t0, loss=(ft.lossacc[:, 0] / n).tolist(), acc=(ft.lossacc[:, 1] / n).tolist()))
    th.save(dict(meta=meta, step=ft.step_n, curve=curve, m=[x.cpu() for x in ft.m], v=[x.cpu() for x in ft.v],
                 **{nm: getattr(ft, nm).cpu() for nm in names}), state_f + ".tmp")
    os.replace(state_f + ".tmp", state_f)
    if len(curve) % 16 == 0 or ft.step_n == n_max:
        el = time.time() - t0
        print(a.tag, ft.step_n, "%.0fs" % el, "%.3f ms/step" % (1e3 * el / (ft.step_n - s0)), ["%.4f" % v for v in curve[-1]["acc"][:8]], flush=True)
wall = time.time() - t0
W = ft.weights()
for k, sp in enumerate(spec):
    np.savez(os.path.join(HERE, "weights", f"{a.tag}_{sp['name']}.npz"), Wi=W[0][k], Wh=W[1][k], Wo=W[2][k])
ev = evaluate([(W[0][k], W[1][k], W[2][k]) for k in range(M)], int(a.eval_n))
json.dump(dict(meta=meta, wall_sec=wall, ms_per_step=1e3 * wall / max(1, n_max - s0), curve=curve,
               eval={sp["name"]: e for sp, e in zip(spec, ev)}, note="fused Triton kernel (in-kernel Philox data)"),
          open(os.path.join(HERE, "results", f"{a.tag}.json"), "w"), indent=1)
for sp, e in zip(spec, ev):
    print(a.tag, sp["name"], "acc %.5f +- %.5f blind %.4f 2-3 %.4f 1-2 %.4f other %.4f" % (
        e["acc"], e["se"], e["err_blind"], e["err_23"], e["err_12"], e["err_other"]), flush=True)
