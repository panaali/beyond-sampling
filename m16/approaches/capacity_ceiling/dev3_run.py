"""T342 DEV3: fresh 2^30 zoo-config retrains (seeds 500-599) with fused_gid.py, data ids 5000 + (seed - 500), disjoint from
grid3, HP2 (0..1024), HP3-HP5 (2000..4099), data seed 41, batch 1024. OPEN development panel (truths published).
Checkpoint each chunk (resumable). -> data/capacity_ceiling/dev3/state.pt in make_panel.py's format (meta.spec, step,
Wi, Wh, Wo). usage: python dev3_run.py"""
import os, sys, time
import torch as th
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fused_gid as G  # noqa: E402
seeds = list(range(500, 600)); gid = [5000 + (s - 500) for s in seeds]; NS = 2 ** 30 // 1024
spec = [dict(name=f"zoo_n30_s{s}", seed=s, n_train=2 ** 30, n_steps=NS, lr_scale=1.0, warmup=2 ** 14, batch=1024,
             f_blind=0.0, f_tie=0.0, gap=0.05, data_id=g) for s, g in zip(seeds, gid)]
meta = dict(spec=spec, data_seed=41, kernel="fused_gid.py")
d = os.path.join(HERE, "..", "..", "..", "data", "capacity_ceiling", "dev3"); os.makedirs(d, exist_ok=True)
f = os.path.join(d, "state.pt"); ft = G.FusedTrainer(seeds, [2 ** 30] * 100, data_seed=41, gid=gid)
if os.path.exists(f):
    st = th.load(f, weights_only=False); assert st["meta"] == meta
    for i, nm in enumerate(("Wi", "Wh", "Wo")):
        getattr(ft, nm).copy_(st[nm]); ft.m[i].copy_(st["m"][i]); ft.v[i].copy_(st["v"][i])
    ft.step_n = st["step"]
t0 = time.time()
while ft.step_n < NS:
    ft.step(min(8192, NS - ft.step_n)); th.cuda.synchronize()
    th.save(dict(meta=meta, step=ft.step_n, Wi=ft.Wi.cpu(), Wh=ft.Wh.cpu(), Wo=ft.Wo.cpu(), m=[x.cpu() for x in ft.m],
                 v=[x.cpu() for x in ft.v]), f + ".tmp"); os.replace(f + ".tmp", f)
    if ft.step_n % (2 ** 17) == 0 or ft.step_n == NS:
        print("dev3", ft.step_n, "%.0fs" % (time.time() - t0), flush=True)
print("DEV3DONE", flush=True)
