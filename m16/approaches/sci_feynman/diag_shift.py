"""Expected error of the rule class (random Cranley-Patterson shifts; diagnosis) on leaderboard models.
run from repo root: python m16/approaches/sci_feynman/diag_shift.py budget reps name..."""
import sys, json
import numpy as np
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/sci_feynman")
import leaderboard as L
import feyn as F
b = int(sys.argv[1]); reps = int(sys.argv[2]); names = sys.argv[3:]
truth = json.load(open(L.TRUTHS)); ms = L.models()
for nm in names:
    E, C = [], []
    for r in range(reps):
        e, c, _ = F.estimate(ms[nm], b, "sobol", shift_seed=100 + r); E.append(e); C.append(c)
    E = np.array(E); t = truth[nm]["acc"]; c = np.mean(C); s2 = t * (1 - t) / c
    print(f"{nm:10s} b={b} cost {c:.0f} bias {100*(E.mean()-t):+.3f}+-{100*E.std()/np.sqrt(reps):.3f}pp sd {100*E.std():.3f}pp "
          f"samp sd {100*np.sqrt(s2):.3f}pp  MSE/sampling {np.mean((E-t)**2)/s2:.2f} (var-only {E.var()/s2:.2f})", flush=True)
