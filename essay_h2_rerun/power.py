"""Power calculation for the T6 H2 re-run (pre-registration; no study data used).

Simulates the design: N items per arm, E explainers per item, explainers grouped into calls of K items.
Item i has a latent probability p_i ~ Beta(mu*kappa, (1-mu)*kappa) of being rated A; each call has a logit
shift ~ N(0, tau^2). Delta-hat = mean item A-score (neurons) - mean item A-score (dense controls).
The CI uses the item-level s.e. inflated by a design factor estimated on the same simulated data
(calls bootstrap), approximated here by the empirical sd of Delta-hat across simulations.
usage: python power.py  -> prints a table; writes power.json
"""
import json

import numpy as np

rng = np.random.default_rng(0)
N_NEU, N_CTL, E, K, SIMS, MARGIN = 280, 280, 2, 14, 2000, 0.30


def sim(mu_n, mu_c, kappa, tau):
    n_items = N_NEU + N_CTL
    mu = np.r_[np.full(N_NEU, mu_n), np.full(N_CTL, mu_c)]
    out = []
    for _ in range(SIMS):
        p = rng.beta(mu * kappa, (1 - mu) * kappa)
        score = np.zeros(n_items)
        for e in range(E):
            order = rng.permutation(n_items)
            shift = np.repeat(rng.normal(0, tau, -(-n_items // K)), K)[:n_items]
            lp = np.log(p[order] + 1e-9) - np.log1p(-p[order] + 1e-9) + shift
            score[order] += rng.random(n_items) < 1 / (1 + np.exp(-lp))
        score /= E
        out.append(score[:N_NEU].mean() - score[N_NEU:].mean())
    return np.array(out)


rows = []
for kappa, tau in [(2, 0.4), (10, 0.4), (2, 0.8)]:
    for d in [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        mu_c = 0.20
        dh = sim(mu_c + d, mu_c, kappa, tau)
        se = dh.std()
        lo, hi = dh - 1.96 * se, dh + 1.96 * se
        rows.append(dict(kappa=kappa, tau=tau, true_delta=d, se=float(se),
                         p_support=float((lo > MARGIN).mean()), p_refute=float((hi < MARGIN).mean()),
                         p_inconclusive=float(((lo <= MARGIN) & (hi >= MARGIN)).mean()),
                         p_diff_gt0=float((lo > 0).mean())))
        r = rows[-1]
        print(f"kappa={kappa:>2} tau={tau} Delta={d:.2f} se={se:.3f} support={r['p_support']:.2f} "
              f"refute={r['p_refute']:.2f} inconcl={r['p_inconclusive']:.2f} Delta>0={r['p_diff_gt0']:.2f}")
# original design for comparison: 84 vs 28, 1 explainer
N_NEU, N_CTL, E = 84, 28, 1
dh = sim(0.61, 0.36, 2, 0.4)
print(f"original design (84 vs 28, 1 explainer), Delta=.25: se={dh.std():.3f}")
rows.append(dict(original_design_se=float(dh.std())))
json.dump(rows, open("essay_h2_rerun/power.json", "w"), indent=1)
