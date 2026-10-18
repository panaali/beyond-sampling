"""Exploratory (post hoc, not pre-registered): the same train-LOO selection restricted to features without rank_exact
(F9, 25.8k FPE), i.e. cost <= ~1.8k (tb) + 0.7k (median margins) + 5k (7 low tangents) + 3.9k (probe-v1) FPE."""
import itertools
import numpy as np
exec(open("fit.py").read().split("subs = [")[0])
allowed = [i for i in range(len(FN)) if FN[i] != "F9_rx"]
subs = [c for k in range(1, 5) for c in itertools.combinations(allowed, k)]
best = min((loo(list(c)), c) for c in subs)
report(list(best[1]), "CHEAP-ONLY selected")
no_tangent = [i for i in allowed if FN[i] not in ("F7_logJm", "F8_lowflip")]
subs = [c for k in range(1, 5) for c in itertools.combinations(no_tangent, k)]
best = min((loo(list(c)), c) for c in subs)
report(list(best[1]), "CHEAP, no low tangents (<= ~6.4k FPE)")
