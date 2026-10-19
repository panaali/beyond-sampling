"""Pilot (estimator only; no truth comparison, seeds 385900+): hit fraction, cells, time and P per rank line."""
import sys, time
import numpy as np
import core, lines
nm = sys.argv[1]; n = int(float(sys.argv[2])); model = core.load(nm); law = core.CondLaw(core.CASES[nm])
for r in (1, 2, 3, 4):
    pr = np.zeros(10); pr[r - 1] = 1; rng = np.random.default_rng(385900 + r); t = time.time()
    o = lines.run_chunk(model, law, rng, n, pr)
    P = o["zval"].sum() / n; se = np.sqrt((o["zval"] ** 2).sum() / n - P ** 2) / np.sqrt(n)
    print(nm, "rank", r, "hits %d (%.2e)" % (len(o["zval"]), len(o["zval"]) / n), "ends", o["n_end"], "cells/line %.2f" % (o["cells"] / n),
          "P %.4e +- %.1e" % (P, se), "max|m| at roots %.1e" % o["max_mres"], "%.1fs" % (time.time() - t), flush=True)
