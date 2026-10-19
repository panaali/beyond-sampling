"""T413 design note calculation (CPU, no scoring): best-case MSE ratio of a hybrid estimator that knows the top-N
k = 10 error patterns' exact error masses (cost c FPE per pattern) and samples the rest, vs plain sampling at budget B.
Hybrid: p_hat = sum_S m + mean_n 1[error and pattern not in S], n = B - c N forward passes (each sample's pattern is
known from its own forward pass). Var = q (1 - q) / n with q = (1 - s(N)) p, s(N) = error-mass share of the top N
patterns (T395 census10.json, plug-in; optimistic at large N since the census's unseen mass is 5.7%).
MSE ratio R = q (1 - q) B / (p (1 - p) (B - c N)), minimised over N. usage: python design.py census10.json"""
import json
import sys

import numpy as np

r = json.load(open(sys.argv[1]))["results"]
p = 1 - 0.95311                                                          # M_16,10 error rate (truth.txt)
out = {}
for k in (7, 10):
    v = r[f"err_k{k}"]; n = v["n"]; cs = sorted(((int(c), m) for c, m in v["coc"].items()), reverse=True)
    counts = np.repeat([c for c, _ in cs], [m for _, m in cs]); share = np.concatenate([[0], np.cumsum(counts) / n])
    print(f"k = {k}: top-N error-mass share", {f"{N:.0e}": round(float(share[min(int(N), len(share) - 1)]), 3) for N in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)})
    for B in (1e3, 4e3, 1.6e4, 6.4e4, 2.56e5):
        row = {}
        for c in (1, 10, 100, 1000):
            Ns = np.unique(np.r_[0, np.logspace(0, np.log10(B / c), 400).astype(int)]); Ns = Ns[(c * Ns < B) & (Ns < len(share))]
            q = (1 - share[Ns]) * p; R = q * (1 - q) * B / (p * (1 - p) * (B - c * Ns)); i = int(R.argmin())
            row[c] = (round(float(R[i]), 3), int(Ns[i]))
        out[f"k{k}_B{int(B)}"] = row
        print(f"  B = {int(B):6d}: best (MSE ratio, N) for c = 1/10/100/1000 FPE per pattern:", row)
json.dump(out, open(sys.argv[1].replace("census10.json", "../t413/design.json"), "w"), indent=1)
