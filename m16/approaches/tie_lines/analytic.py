"""Precise compute-adjusted efficiency per model from per-stratum variances (large samples), instead of
noisy rep-MSEs. For a stratified design with n_h = a_h * B draws in stratum h at cost c_h each:
   Var = sum_h P_h^2 s_h^2 / (a_h B),  cost = B * sum_h a_h c_h
   gain vs plain sampling at equal cost = acc(1-acc) / (Var * cost).
Designs: strat (stratified.py allocation, plain terms), strat_swap (estimators.SW_N), strat_line (exact tie
lines in strata 1,2; tangent pieces cost 2), and the oracle Neyman optimum of each (per-model allocation;
not achievable without a per-model pilot, shown as a ceiling).  usage: python analytic.py [n_per_stratum]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import estimators as E
from stratified import draw, N_STRATA, EDGES
from tielines import Z_stratum
from arc_eval import noisy_models
from common import Wi, Wh, Wo
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
NL = 1500
GEN = 200 / 6e3
models = {"orig": (Wi, Wh, Wo)}
for s in (0.0025, 0.005):
    for k, m in enumerate(noisy_models(6, s)):
        models[f"s{s}_{k}"] = m
P = E.P


def design(sig, cost, alloc):
    var = (P ** 2 * sig ** 2 / alloc).sum(); c = (alloc * cost).sum()
    return var * c


rows = []
for name, m in models.items():
    rng = np.random.default_rng(42)
    mu = np.zeros(N_STRATA); sp = np.zeros(N_STRATA); ss = np.zeros(N_STRATA)
    sl = np.zeros(N_STRATA); cl = np.zeros(N_STRATA)
    for h in range(N_STRATA):
        x = draw(h, NS, rng); g = E.g_of(m, x); gs = E.g_of(m, E.swap_minpair(x)[0])
        mu[h] = g.mean(); sp[h] = g.std(); ss[h] = ((g + gs) / 2).std()
        if h in (1, 2):
            Z, pc = Z_stratum(m, x[:NL], EDGES); sl[h] = Z.std(); cl[h] = 2 * pc.mean()
    acc = (P * mu).sum(); ref = acc * (1 - acc)
    a0 = E.ALLOC0
    sw_sig = np.where(E.USE_SWAP, ss, sp); sw_c = np.where(E.USE_SWAP, 2.0, 1.0)
    ln_sig = sp.copy(); ln_c = np.ones(N_STRATA); ln_sig[[1, 2]] = sl[[1, 2]]; ln_c[[1, 2]] = cl[[1, 2]]
    ln_alloc = P * ln_sig / np.sqrt(ln_c)   # neyman on this model (ceiling for the line design)
    r = dict(model=name, acc=float(acc),
             strat=ref / design(sp, np.ones(N_STRATA), a0),
             strat_swap=ref / design(sw_sig, sw_c, E.SW_N),
             strat_swap_gen=None,
             strat_oracle=ref / ((P * sp).sum()) ** 2,
             strat_swap_oracle=ref / ((P * sw_sig * np.sqrt(sw_c)).sum()) ** 2,
             strat_line_oracle=ref / ((P * ln_sig * np.sqrt(ln_c)).sum()) ** 2,
             sd_plain=sp.round(3).tolist(), sd_swap=ss.round(3).tolist(), sd_line12=sl[[1, 2]].round(3).tolist(),
             cost_line12=cl[[1, 2]].round(2).tolist())
    # with input-generation cost (pooled draw: candidates = max_h n_h / P_h per unit budget)
    cand = (E.SW_N / P).max() * 1.05
    r["strat_swap_gen"] = r["strat_swap"] * (E.SW_N * sw_c).sum() / ((E.SW_N * sw_c).sum() + GEN * cand)
    cand0 = (a0 / P).sum()                    # stratified.py's per-stratum rejection
    r["strat_gen"] = r["strat"] / (1 + GEN * cand0)
    rows.append(r); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(rows, open(os.path.join(HERE, "analytic.json"), "w"), indent=1)
