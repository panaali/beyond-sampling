"""T446 (iv) safe-use rating at L = 15 (pre-registered in README). usage: python rate.py MODEL
r(k) = sum_t P(dist_t >= k, p_ill_t > tau_r) / sum_t P(dist_t >= k); numerator by CE-IS from the weights
(1e5 FPE per k), denominator model-free (exact language enumeration). Check: the exact r(k)."""
import json
import sys
import zlib

import numpy as np

from est import ce_is, phi_all
from s3lib import DATA, Dyck2, load, stack_info

L, KS = 15, (2, 4, 6, 8, 10, 12)
name = sys.argv[1]
ex = json.load(open(f"{DATA}/{name}_L15_exact.json"))
rn, rd = np.array(ex["r_num"]), np.array(ex["r_den"])
cand = [j for j, t in enumerate(ex["r_taus"]) if all(rn[j, k] > 0 for k in KS)]
out = dict(model=name, ks=KS)
if not cand:
    out["result"] = "no tau_r with nonzero exact r at all k"
else:
    j = cand[0]  # r_taus are (.5, .1, .01): the first candidate is the largest tau_r
    tau_r = ex["r_taus"][j]
    model, _ = load(name)
    lang = Dyck2(L=L)
    rows = []
    for k in KS:
        def score(y, k=k):
            ph = phi_all(model, lang, y)[:, :L]
            dep, _, opos = stack_info(y)
            far = (dep > 0) & (np.arange(L)[None, :] - opos >= k)
            return np.where(far, ph, 0).max(1), ((ph > tau_r) & far).sum(1)
        rng = np.random.default_rng(zlib.crc32(f"{name}|rate|{k}".encode()))
        num, cost = ce_is(model, lang, L, tau_r, 1e5, rng, score=score)
        r_est, r_ex = num / rd[k], rn[j, k] / rd[k]
        rows.append(dict(k=k, r_est=r_est, r_exact=r_ex, ratio=r_est / r_ex, cost=cost))
        print(name, f"tau_r {tau_r} k {k}: est {r_est:.3e} exact {r_ex:.3e} ratio {r_est / r_ex:.3f}", flush=True)
    out.update(tau_r=tau_r, rows=rows, within2x=int(sum(.5 <= r["ratio"] <= 2 for r in rows)))
json.dump(out, open(f"{DATA}/rate_{name}.json", "w"))
