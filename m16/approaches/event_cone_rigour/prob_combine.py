"""Combine prob_bound.py shards: one-sided Clopper-Pearson upper bound on P(E(t0)) at confidence 1 - alpha.
usage: python prob_combine.py t0 alpha -> prints the bound (PROBABILISTIC, sampling-class)"""
import glob
import sys

import numpy as np
from scipy.stats import beta

t0 = float(sys.argv[1]); alpha = float(sys.argv[2])
fs = sorted(glob.glob(f"data/prob_{t0}_*.npz"))
N = sum(int(np.load(f)["draws"]) for f in fs); h = sum(int(np.load(f)["hits"]) for f in fs)
kept = sum(int(np.load(f)["kept"]) for f in fs)
ub = beta.ppf(1 - alpha, h + 1, N - h)
print(f"t0 {t0}: shards {len(fs)} draws {N:.3g} gap-kept {kept} hits {h}; point {h / N:.3g}; "
      f"Clopper-Pearson upper (conf 1-{alpha:g}) {ub:.3g}  [probabilistic]")
