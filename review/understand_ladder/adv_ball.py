"""T97: adversarial search INSIDE ARC's noise scale. Start from the best-of-5 seed + delta, delta ~ N(0, 0.05^2) per
weight; greedy hill-climb on delta (steps 0.01 N(0,1)), each candidate rescaled so RMS(delta) = 0.05 exactly (the typical
size of an ARC sigma = 0.05 copy). Objective: per-model MSE ratio of F2L2 (author's fibre.py, LOOK = 2) vs sampling.
The final worst model is also scored with fibre_fix (id-reuse fix) and LOOK = 1, 3.
usage: OMP_NUM_THREADS=1 python adv_ball.py H steps restarts seed  -> adv_ball_H{H}.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_ladder"))
import fibre  # noqa: E402
import fibre_fix  # noqa: E402
import rtruth as R  # noqa: E402
from alg_zoo.zoo import best_of_5_seed_2nd_argmax  # noqa: E402


def ev(m, mod=fibre, look=2):
    H = len(m[0]); ex = R.exact(m)[0]
    mod.LOOK = look; o = mod.Ops(); e = mod.est_fibre2(m, o, {})
    return dict(truth=ex, err=e - ex, ops=o.n, ratio=(e - ex) ** 2 / (ex * (1 - ex) / (o.n / (3 * H * (H + 2)))))


def rescale(d, s=0.05):
    r = np.sqrt(np.mean(np.concatenate([x.ravel() for x in d]) ** 2))
    return tuple(x * s / r for x in d)


H, steps, restarts, sd = (int(a) for a in sys.argv[1:5])
rng = np.random.default_rng(sd)
base = R.load(H, 3, best_of_5_seed_2nd_argmax(hidden_size=H, seq_len=3))[0]
t0 = time.time(); runs = []
for k in range(restarts):
    d = rescale(tuple(rng.standard_normal(w.shape) * 0.05 for w in base))
    r = ev(tuple(w + x for w, x in zip(base, d))); start = r["ratio"]
    for j in range(steps):
        d2 = rescale(tuple(x + 0.01 * rng.standard_normal(x.shape) for x in d))
        r2 = ev(tuple(w + x for w, x in zip(base, d2)))
        if r2["ratio"] > r["ratio"]:
            d, r = d2, r2
    m = tuple(w + x for w, x in zip(base, d))
    extra = {f"orig_L{L}": ev(m, fibre, L) for L in (1, 3)}
    extra.update({f"fix_L{L}": ev(m, fibre_fix, L) for L in (1, 2, 3)})
    runs.append(dict(start_ratio=start, final=r, extra=extra, model=[w.tolist() for w in m]))
    print(f"H={H} restart {k}: ratio {start:.2e} -> {r['ratio']:.3g} (err {r['err']:+.2e}, acc {r['truth']:.4f}); "
          + " ".join(f"{a} {b['ratio']:.2g}" for a, b in extra.items()) + f"  [{time.time() - t0:.0f}s]", flush=True)
json.dump(runs, open(os.path.join(HERE, f"adv_ball_H{H}.json"), "w"), indent=1)
