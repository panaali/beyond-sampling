"""T95 scaling data point (not a rung): zoo_2nd_argmax(16, 3) seeds 0-4 and 6 ARC-protocol copies of the best seed;
exact truth (exact3), region count, F2L2 via fibre_fast. Writes results_H16_T3.json. usage: python run_h16.py"""
import json
import time

import numpy as np

import fibre_fast
from exact3 import count_regions, exact_acc
import m43
from fiber3 import Ops
from zoo import best_seed, noisy_abs, zoo_model

if __name__ == "__main__":
    H = 16
    ms = {f"seed{s}": zoo_model(H, 3, s) for s in range(5)}
    for k, m in enumerate(noisy_abs(zoo_model(H, 3, best_seed(H, 3)), 6, 0.05, 0)):
        ms[f"abs0.05_n0_{k}"] = m
    rows = {}
    for name, m in ms.items():
        oe = m43.Ops(); st = {}; t = time.time(); ex = exact_acc(m, oe, st); te = time.time() - t
        o = Ops(); inf = {}; t = time.time(); e = fibre_fast.est_fast(m, o, inf); tf = time.time() - t
        rows[name] = {"truth": ex, "exact_ops": oe.n, "exact_sec": te, "regions": count_regions(m),
                      "F2L2fast": {"est": e, "ops": o.n, "traces": inf["traces"], "cons": inf["cons"], "sec": tf}}
        print(name, f"truth {ex:.6f} err {e - ex:+.1e} regions {rows[name]['regions']} ops {o.n} traces {inf['traces']}", flush=True)
    json.dump({"H": H, "T": 3, "rows": rows}, open("results_H16_T3.json", "w"), indent=1)
