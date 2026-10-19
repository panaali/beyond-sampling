"""Gate: my walk's per-line wrong masses vs the frozen walk (radial, hom.tcdf) on the step-1a lines."""
import sys, time
import numpy as np
import plwalk as P
import walk as W, hom
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
for name in (sys.argv[2:] or P.MODELS):
    m = P.load(name); worst = 0.0; bad = 0; tot = 0; t0 = time.time()
    for j in range(len(P.STRATA)):
        U = P.C.sobol(P.T, n, P.RULE_SEED + 101 * j)
        o = P.run(m, j, U)
        w, d, tw, kn, kp, wt, lab = P.lines(j, U)
        kind = P.STRATA[j][0]
        a = np.where(kind == "23", kp, kn); b = np.where(kind == "23", kn, kp)
        F, c, _ = W.walk(m, w, a, b, tw, kind, cdf=hom.tcdf(P.T))
        e = np.abs(F - o["mass"]); worst = max(worst, e.max()); bad += int((e > 1e-10).sum()); tot += n
    print(f"{name}: lines {tot} worst |dmass| {worst:.2e} fails(>1e-10) {bad} ({time.time()-t0:.0f}s)", flush=True)
