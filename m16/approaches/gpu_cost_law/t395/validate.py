"""T395 pre-registered validation: census all-input k = 4, 5 exp(H) and N(delta) vs the exact region masses
(../results/prefix/k{4,5}_regionmass.npz), pass iff |relative difference| <= 1% where resolved; k = 6 distinct D <=
19,740,596 (T270 region count). usage: python validate.py census10.json -> prints the table, writes validate.json"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(sys.argv[1]))["results"]; out = {}
for k in (4, 5):
    m = np.load(os.path.join(HERE, "..", "results", "prefix", f"k{k}_regionmass.npz"))["mass"]
    p = np.sort(m[m > 0])[::-1] / m.sum(); eH = math.exp(-(p * np.log(p)).sum()); cs = np.cumsum(p)
    c = r[f"all_k{k}"]; row = dict(exact_exp_H=eH, census_exp_H=c["exp_H"], census_exp_H_cs=c["exp_H_chao_shen"],
                                   rel_exp_H=c["exp_H"] / eH - 1, N_delta={})
    ok = abs(row["rel_exp_H"]) <= .01
    for dl, v in c["N_delta"].items():
        ex = int(np.searchsorted(cs, 1 - float(dl)) + 1); rel = v["N"] / ex - 1
        row["N_delta"][dl] = dict(exact=ex, census=v["N"], rel=rel, resolved=v["resolved"])
        if v["resolved"]:
            ok &= abs(rel) <= .01
    row["pass"] = bool(ok); out[f"k{k}"] = row
    print(f"k={k} exp(H) exact {eH:.1f} census {c['exp_H']:.1f} ({row['rel_exp_H']:+.4f}) CS {c['exp_H_chao_shen']:.1f}",
          {d: (v["exact"], v["census"], "%+.4f" % v["rel"], v["resolved"]) for d, v in row["N_delta"].items()}, "PASS" if ok else "FAIL")
D6 = r["all_k6"]["distinct"]; out["k6"] = dict(distinct=D6, t270_regions=19740596, pass_=D6 <= 19740596)
print("k=6 distinct", D6, "<= 19,740,596:", D6 <= 19740596)
json.dump(out, open(os.path.join(HERE, "validate.json"), "w"), indent=1)
