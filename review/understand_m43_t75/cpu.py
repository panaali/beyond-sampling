"""T84 item 3b: CPU time of the frozen recipe (reviewed est75) vs plain sampling at equal ops and at equal time.
usage: OMP_NUM_THREADS=1 python cpu.py"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_m43"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import est75 as E75
from m43 import Ops
import recipe as R
from mine import zoo_w, noisy_w, fwd

d = json.load(open(os.path.join(HERE, "fresh_0.05_30-31-32-33.json")))
g = np.random.default_rng(3)
for zs in range(5):
    ms = noisy_w(zoo_w(zs), 20, 0.05, 90)
    t = time.process_time(); [E75.est75(m, Ops(), **E75.RECIPE) for m in ms]; te = (time.process_time() - t) / 20
    t = time.process_time(); [R.recipe(R.as_lists(m)) for m in ms]; tr = (time.process_time() - t) / 20
    z = d["zoo"][str(zs)]; ns = int(z["ops"] / 72)
    t = time.process_time()
    for m in ms:
        x = g.standard_normal((ns, 3)); _ = np.mean(fwd(m, x).argmax(1) == np.argsort(x, 1)[:, 1])
    ts = (time.process_time() - t) / 20
    # samples drawable in te seconds (measured throughput at 1e5 per batch)
    x = g.standard_normal((100_000, 3)); t = time.process_time()
    _ = np.mean(fwd(ms[0], x).argmax(1) == np.argsort(x, 1)[:, 1]); thr = 100_000 / (time.process_time() - t)
    n_eq = te * thr; mse_samp_time = z["pq"] / n_eq; mse_est = z["rms"] ** 2
    print(f"zoo {zs}: est75 {1000*te:.1f} ms, re-impl (pure Python) {1000*tr:.1f} ms; sampling at equal ops ({ns} samples) "
          f"{1000*ts:.2f} ms -> {te/ts:.0f}x; equal-CPU-time sampling n = {n_eq:.0f}, MSE ratio at equal time "
          f"{mse_est/mse_samp_time:.2f}", flush=True)
