"""Pilot (estimator only, no truths; seeds 385950+): per-rank s.e. of P and of the projections on the
pre-registered directions, per CPU second."""
import sys, time
import numpy as np
import core, lines, dirs
nm = sys.argv[1]; n = int(float(sys.argv[2])); model = core.load(nm); law = core.CondLaw(core.CASES[nm])
names, D = dirs.all_dirs(nm); show = [0, 1, 2, 6, 7, 8, 11, 12]
for r in [int(a) for a in sys.argv[3].split(",")]:
    pr = np.zeros(10); pr[r - 1] = 1; rng = np.random.default_rng(385950 + r); t = time.time(); zs, gs = [], []; m = 0
    while m < n:
        o = lines.run_chunk(model, law, rng, 200_000, pr); zs.append(o["zval"]); gs.append(o["grad"]); m += o["n"]
    sec = time.time() - t; z = np.concatenate(zs); G = np.concatenate(gs) @ D.T
    P = z.sum() / m; E = G.sum(0) / m / P
    resid = G - E[None] * z[:, None]; se = np.sqrt((resid ** 2).sum(0) / m) / np.sqrt(m) / P   # zero rows add nothing
    print(nm, "rank", r, "lines", m, "hits", len(z), "P %.3e" % P, "%.0fs" % sec)
    print("   ", " ".join(f"{names[i]}: {E[i]:.1f}+-{se[i]:.1f} (se*sqrt(sec) {se[i]*np.sqrt(sec):.0f})" for i in show), flush=True)
