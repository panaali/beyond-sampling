"""For one model (index into [original] + noisy_models(6,.0025) + noisy_models(6,.005)): truth (4M samples),
a pool of great circles from orthogonal frames (exact fraction, pieces, partial cost, CV features), and a
stratified-sampling pilot (per-stratum variance). Saves pool_<idx>.npz."""
import sys, json, time
import numpy as np
HERE = "/home/user/clean-neurons/m16/approaches/great_circle"
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, HERE)
from common import Wi, Wh, Wo
from arc_eval import forward, true_acc, noisy_models
from stratified import stratum, N_STRATA, draw
from gcirc import walk, frame_planes

MODELS = [(Wi, Wh, Wo)] + noisy_models(6, 0.0025) + noisy_models(6, 0.005)
idx = int(sys.argv[1]); nf = int(sys.argv[2]) if len(sys.argv) > 2 else 400
M = MODELS[idx]
t0 = time.time()
truth = 0.95317945 if idx == 0 else true_acc(M, n=4_000_000)
rng = np.random.default_rng(100 + idx)
U, V = frame_planes(nf, rng)
f, p, pc = walk(M, U, V)
G = 512; th = np.arange(G) / G * 2 * np.pi
feat = np.stack([np.bincount(stratum(np.cos(th)[:, None] * U[i] + np.sin(th)[:, None] * V[i]), minlength=N_STRATA) / G
                 for i in range(len(f))])
# stratified pilot: per-stratum Bernoulli variance
sv = np.zeros(N_STRATA); sm = np.zeros(N_STRATA)
for h in range(N_STRATA):
    x = draw(h, 20000, rng)
    y = forward(M, x).argmax(1) == np.argsort(x, 1)[:, -2]
    sm[h], sv[h] = y.mean(), y.var()
np.savez(f"{HERE}/pool_{idx}.npz", truth=truth, f=f, p=p, pc=pc, feat=feat, sv=sv, sm=sm)
print(idx, f"truth {truth:.5f} gc mean {f.mean():.5f} +- {f.std()/np.sqrt(len(f)):.5f} pieces {p.mean():.1f} "
      f"var {f.var():.3e} time {time.time()-t0:.0f}s", flush=True)
