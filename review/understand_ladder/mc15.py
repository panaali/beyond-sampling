"""1e7-sample MC of the actual alg_zoo torch modules (all 15 zoo seeds at H = 4, 5, 6, T = 3) vs own exact enumeration.
usage: OMP_NUM_THREADS=1 python mc15.py [n]  -> mc15.json"""
import json
import math
import sys

import rtruth as R

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
out = {}
for H in (4, 5, 6):
    for s in range(5):
        m, mod = R.load(H, 3, s)
        ex, nr, _ = R.exact(m)
        mc = R.mc_torch(mod, n, 1000 + 10 * H + s)
        se = math.sqrt(ex * (1 - ex) / n)
        out[f"H{H}_seed{s}"] = dict(exact=ex, mc=mc, se=se, z=(mc - ex) / se, regions=nr)
        print(f"H={H} seed={s} exact {ex:.6f} mc {mc:.6f} z {(mc - ex) / se:+.2f}", flush=True)
json.dump(out, open("mc15.json", "w"), indent=1)
