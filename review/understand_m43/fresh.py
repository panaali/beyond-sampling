"""T74 items 1 and 4: fresh noise seeds and other noise levels. Truth = exact.py (validated independently by
truth_mc.py). Tiers exactly as in understand_m43/run_mech.py. Adds a bootstrap CI on the MSE ratio, the
empirical sampling MSE at equal ops (not the formula), and an eps sensitivity scan of tier B.
usage: OMP_NUM_THREADS=1 python fresh.py SIGMA SEED [SEED ...]   (writes fresh_<sigma>_<seeds>.json)"""
import json, os, sys, time
import numpy as np
from common_review import base_model, noisy, my_forward
import est_mech as EM
from m43 import Ops
from exact import exact_acc
from run_mech import TIERS

sigma = float(sys.argv[1]); seeds = [int(s) for s in sys.argv[2:]]
b = base_model(); models = []
for s in seeds:
    models += noisy(b, 256, sigma, s)
t0 = time.time()
truth = np.array([exact_acc(m) for m in models]); pq = float(np.mean(truth * (1 - truth)))
print(f"sigma {sigma} seeds {seeds}: n={len(models)} mean truth {truth.mean():.4f} min {truth.min():.4f} pq {pq:.4f} ({time.time()-t0:.0f}s)")
rng = np.random.default_rng(12345)
variants = dict(TIERS)
for eps in (0.02, 0.1, 0.2):
    variants[f"B eps={eps}"] = dict(TIERS["B default (A2 + resolved crossover)"], eps=eps)
out = {"sigma": sigma, "seeds": seeds, "mean_truth": float(truth.mean()), "pq": pq, "tiers": {}}
for name, kw in variants.items():
    kw = dict(kw); EM.COMPLETE = kw.pop("complete")
    est, ops = [], []; t1 = time.time()
    for m in models:
        o = Ops(); est.append(EM.est_mech(m, o, **kw)); ops.append(o.n)
    sec = (time.time() - t1) / len(models)
    est = np.array(est); ops = np.array(ops); e = est - truth; om = ops.mean()
    n_samp = om / 72; mse_s = pq / n_samp; ratio = np.mean(e ** 2) / mse_s
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(e), len(e)); tt = truth[idx]
        bs.append(np.mean(e[idx] ** 2) / (np.mean(tt * (1 - tt)) / (ops[idx].mean() / 72)))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    row = dict(rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), max_abs=float(np.abs(e).max()),
               ops_mean=float(om), ops_max=int(ops.max()), mse_ratio=float(ratio), ci95=[float(lo), float(hi)],
               rms_per_seed=[float(np.sqrt(np.mean(e[256 * i:256 * (i + 1)] ** 2))) for i in range(len(seeds))],
               sec_per_model=sec)
    out["tiers"][name] = row
    print(f"{name:36s} rms {row['rms']:.4f} bias {row['bias']:+.4f} max {row['max_abs']:.4f} ops {om:.0f} "
          f"ratio {ratio:.4f} [{lo:.4f},{hi:.4f}] per-seed {[round(x,4) for x in row['rms_per_seed']]} {sec*1e3:.1f} ms/model", flush=True)
# empirical plain sampling at tier B's mean op count (forward passes only; RNG uncharged as in ARC)
nB = int(round(out["tiers"]["B default (A2 + resolved crossover)"]["ops_mean"] / 72)); reps = 20
errs = []; t1 = time.time()
for m, tr in zip(models, truth):
    x = rng.standard_normal((reps, nB, 3)).reshape(-1, 3)
    ok = (my_forward(m, x).argmax(1) == np.argsort(x, 1)[:, 1]).reshape(reps, nB).mean(1)
    errs += list(ok - tr)
sec = (time.time() - t1) / (len(models) * reps)
errs = np.array(errs)
out["sampling_empirical"] = dict(n=nB, rms=float(np.sqrt(np.mean(errs ** 2))), formula_rms=float(np.sqrt(pq / nB)),
                                 sec_per_estimate=sec)
print(f"plain sampling, n={nB} samples ({nB*72} ops): empirical rms {np.sqrt(np.mean(errs**2)):.4f} vs formula "
      f"{np.sqrt(pq/nB):.4f}; {sec*1e3:.2f} ms per estimate (vectorised numpy)")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 f"fresh_{sigma}_{'-'.join(map(str, seeds))}.json"), "w"), indent=1)
print(f"done {time.time()-t0:.0f}s")
