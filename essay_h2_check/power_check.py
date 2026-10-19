"""T383 power check (own code). Simulates the T383 design and the pre-registered decision rule.

Design: N items per arm, 2 explainers per item, each round packs all 3 arms into packets of 21.
Item latent A-probability p ~ Beta(k*m, k*(1-m)); each explainer call adds a logit shift ~ N(0, tau).
CI: two-way bootstrap (items within arm x explainer calls), as in score.py. Writes power.json.
usage: python power_check.py
"""
import json

import numpy as np

N, PACK, ARMS, B, SIMS = 308, 21, 3, 400, 300
CTRL = 0.20


def simulate(rng, deltas, kappa, tau):
    m = np.array([CTRL + deltas[0], CTRL + deltas[1], CTRL])  # plain, matched, controls
    n = ARMS * N
    arm = np.repeat(np.arange(ARMS), N)
    p = rng.beta(kappa * m[arm], kappa * (1 - m[arm]))
    lp = np.log(p / (1 - p))
    calls = []  # call id of each item's explainer in rounds 1, 2
    a = np.zeros((n, 2))
    for r in range(2):
        perm = rng.permutation(n)
        cid = np.empty(n, int)
        cid[perm] = np.arange(n) // PACK + r * (n // PACK)
        u = rng.normal(0, tau, n // PACK * 2)
        a[:, r] = rng.random(n) < 1 / (1 + np.exp(-(lp + u[cid])))
        calls.append(cid)
    return arm, a, np.stack(calls, 1)


def boot_delta(rng, arm, a, calls, contrast):
    n_calls = calls.max() + 1
    out = np.empty(B)
    for b in range(B):
        wc = rng.multinomial(n_calls, np.ones(n_calls) / n_calls)
        w = wc[calls]  # (n, 2) weights of each explanation
        rates = []
        for k in contrast:
            idx = np.flatnonzero(arm == k)
            idx = idx[rng.integers(0, len(idx), len(idx))]
            ws = w[idx].sum(1)
            ok = ws > 0
            rates.append(((w[idx] * a[idx]).sum(1)[ok] / ws[ok]).mean())
        out[b] = rates[0] - rates[1]
    return np.percentile(out, [2.5, 97.5])


def main():
    rng = np.random.default_rng(383)
    res = []
    for kappa, tau in ((2, 0.4), (2, 0.8), (10, 0.8)):
        for d in (0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.44, 0.50):
            sup = ref = exc0 = 0
            se = []
            for s in range(SIMS):
                arm, a, calls = simulate(rng, (d, d), kappa, tau)
                lo, hi = boot_delta(rng, arm, a, calls, (0, 2))
                sup += lo > 0.30; ref += hi < 0.30; exc0 += lo > 0
                se.append((hi - lo) / 3.92)
            r = {"kappa": kappa, "tau": tau, "true_delta": d, "P_support": sup / SIMS, "P_refute": ref / SIMS,
                 "P_inconclusive": 1 - (sup + ref) / SIMS, "P_ci_excludes_0": exc0 / SIMS, "mean_se": float(np.mean(se))}
            print(r, flush=True)
            res.append(r)
    json.dump({"N_per_arm": N, "sims": SIMS, "boot": B, "control_rate": CTRL, "rows": res},
              open("essay_h2_check/power.json", "w"), indent=1)


if __name__ == "__main__":
    main()
