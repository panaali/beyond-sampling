"""T84 item 4d: does a margin-aware step-3 impact (recipe.py mmode='margin', this review's own variant; not the
frozen T75 recipe and not T85's RECIPE85) fix zoo seed 4? Fresh copies whose exact truths are already in
fresh_*_zoo4.json and fresh_0.05_30-31-32-33.json. Ops: recipe.py charge.
usage: OMP_NUM_THREADS=1 python seed4_margin.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import recipe as R
from mine import zoo_w, noisy_w

cases = [("fresh_0.05_30-31-32-33.json", 0.05, 30, 256), ("fresh_0.02_56_zoo4.json", 0.02, 56, 256),
         ("fresh_0.015_54_zoo4.json", 0.015, 54, 256)]
for f, sig, ns, n in cases:
    tr = np.array(json.load(open(os.path.join(HERE, f)))["zoo"]["4"]["truths"][:n])
    ms = [R.as_lists(m) for m in noisy_w(zoo_w(4), n, sig, ns)]
    for name, kw in (("T75 frozen (re-impl)", dict(eps=0.1)), ("margin-aware step 3, eps 0.1", dict(eps=(0.1, 0.1), mmode="margin"))):
        e, ops = [], []
        for m, t in zip(ms, tr):
            ch = R.Charge(); e.append(R.recipe(m, ch, **kw) - t); ops.append(ch.n)
        e = np.array(e); om = np.mean(ops)
        print(f"sigma {sig} seed {ns}: {name:30s} rms {100*np.sqrt(np.mean(e**2)):.2f}% max {100*np.abs(e).max():.1f}% "
              f"ops {om:.0f} ratio {np.mean(e**2)/(np.mean(tr*(1-tr))/(om/72)):.4f}", flush=True)
