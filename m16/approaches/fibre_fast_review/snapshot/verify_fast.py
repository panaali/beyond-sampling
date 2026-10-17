"""T95 reference check: fibre_fast.est_fast vs the reference F2L2 (fibre.est_fibre2, LOOK = 2) and vs exact truth,
on every model in results_H{H}_T3.json. usage: python verify_fast.py H [n_models]"""
import json
import sys
import time

import numpy as np

import fibre
import fibre_fast
from fiber3 import Ops
from run_t3 import group
from zoo import best_seed, leaderboard_models, noisy_abs, zoo_model

if __name__ == "__main__":
    H = int(sys.argv[1]); nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    R = json.load(open(f"results_H{H}_T3.json")); rows = R["rows"]
    ms = leaderboard_models(H, 3)
    base = zoo_model(H, 3, best_seed(H, 3))
    for ns in (0, 1):
        for k, m in enumerate(noisy_abs(base, R["K"], 0.05, ns)):
            ms[f"abs0.05_n{ns}_{k}"] = m
    d_ref = []; d_tr = []; ops_f = []; ops_r = []; out = {}
    for name in list(ms)[:nmax]:
        m = ms[name]
        o = Ops(); inf = {}; t = time.time(); e = fibre_fast.est_fast(m, o, inf); sec = time.time() - t
        ref = rows[name]["F2L2"]
        d_ref.append(e - ref["est"]); d_tr.append(e - rows[name]["truth"]); ops_f.append(o.n); ops_r.append(ref["ops"])
        out[name] = {"est": e, "ops": o.n, "traces": inf["traces"], "cons": inf["cons"], "sec": sec}
    d_ref = np.array(d_ref)
    print(f"H={H}: {len(d_ref)} models; max |fast - reference F2L2| = {np.abs(d_ref).max():.2e}; "
          f"max |fast - truth| = {np.abs(d_tr).max():.2e}; ops fast {np.mean(ops_f):.0f} vs reference {np.mean(ops_r):.0f} "
          f"(x{np.mean(ops_r) / np.mean(ops_f):.2f})")
    if len(out) == len(ms):
        for name, v in out.items():
            rows[name]["F2L2fast"] = v
        json.dump(R, open(f"results_H{H}_T3.json", "w"), indent=1)
