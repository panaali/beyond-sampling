"""ANAT-80 + own truths (DIAGNOSTIC SAMPLING, not an estimator). Per model n i.i.d. N(0,I_10) inputs
(seed 292000 + model index); tallies of error by region (blind / 2~3 band / 1~2 band / soft tail), by ordered
top-3 triple, by x9 case, by the rank of the wrong answer, and the tail at min gap >= .05/.1/.2.
usage: python anat.py [n=2e7] [first] [last]  -> results/anat/<name>.npz + .json"""
import json
import os
import sys
import time

import numpy as np

from panel import HERE, dev2, target, forward

n = int(float(sys.argv[1])) if len(sys.argv) > 1 else 20_000_000
models = [("target", target())] + dev2()
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(models)
OUT = os.path.join(HERE, "results/anat"); os.makedirs(OUT, exist_ok=True)
CH = 500_000
GE = np.array([0, .01, .02, .05, .1, .2, .5, np.inf])   # min-gap bins (non-blind)
for k in range(lo, hi):
    name, m = models[k]
    if os.path.exists(os.path.join(OUT, name + ".json")):
        continue
    rng = np.random.default_rng(292000 + k); t0 = time.time()
    # region r: 0 blind, 1 band23, 2 band12, 3 tail ; x9 case c: 0 max, 1 2nd, 2 below
    tot = np.zeros((4, 3, 1000)); er = np.zeros((4, 3, 1000))
    gtot = np.zeros((len(GE) - 1, 1000)); ger = np.zeros((len(GE) - 1, 1000))     # min-gap bin x triple (non-blind)
    g23tot = np.zeros(len(GE) - 1); g23er = np.zeros(len(GE) - 1)
    rank = np.zeros((4, 10))                                                     # rank of the chosen input, errors
    for _ in range(n // CH):
        x = rng.standard_normal((CH, 10))
        o = np.argsort(-x, 1); s = np.take_along_axis(x, o[:, :3], 1)
        pred = forward(m, x).argmax(1)
        err = pred != o[:, 1]
        g12, g23 = s[:, 0] - s[:, 1], s[:, 1] - s[:, 2]
        blind = s[:, 1] <= 0
        r = np.where(blind, 0, np.where(np.minimum(g12, g23) >= .05, 3, np.where(g23 <= g12, 1, 2)))
        c = np.where(o[:, 0] == 9, 0, np.where(o[:, 1] == 9, 1, 2))
        tri = o[:, 0] * 100 + o[:, 1] * 10 + o[:, 2]
        idx = (r * 3 + c) * 1000 + tri
        tot += np.bincount(idx, minlength=12000).reshape(4, 3, 1000)
        er += np.bincount(idx[err], minlength=12000).reshape(4, 3, 1000)
        nb = ~blind
        gb = np.clip(np.searchsorted(GE, np.minimum(g12, g23), side="right") - 1, 0, len(GE) - 2)
        gi = gb * 1000 + tri
        gtot += np.bincount(gi[nb], minlength=7000).reshape(7, 1000)
        ger += np.bincount(gi[nb & err], minlength=7000).reshape(7, 1000)
        b23 = np.clip(np.searchsorted(GE, g23, side="right") - 1, 0, len(GE) - 2)
        g23tot += np.bincount(b23[nb], minlength=7); g23er += np.bincount(b23[nb & err], minlength=7)
        rk = np.argsort(o, 1)[np.arange(CH), pred]                                  # rank of predicted position
        rank += np.bincount(r[err] * 10 + rk[err], minlength=40).reshape(4, 10)
    np.savez_compressed(os.path.join(OUT, name + ".npz"), tot=tot, er=er, gtot=gtot, ger=ger, rank=rank)
    acc = 1 - er.sum() / n
    E = er.sum((1, 2)) / n
    j = dict(name=name, n=n, seed=292000 + k, acc=acc, se=float(np.sqrt(acc * (1 - acc) / n)),
             err_pp={nm: 100 * float(E[i]) for i, nm in enumerate(("blind", "band23", "band12", "tail"))},
             tail_pp={str(GE[i]): 100 * float(ger[i:].sum() / n) for i in (3, 4, 5)},
             err_by_g23_bin=(g23er / np.maximum(g23tot, 1)).tolist(),
             wrong_rank_pp=(100 * rank / n).tolist(), sec=time.time() - t0)
    json.dump(j, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(k, name, "acc %.5f  blind %.3f b23 %.3f b12 %.3f tail %.3f pp  %.0fs" % (
        acc, *[j["err_pp"][q] for q in ("blind", "band23", "band12", "tail")], j["sec"]), flush=True)
