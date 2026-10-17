"""T100 rung (H=4, T=4) scoring for the fibre recipe (frozen F2L3 = fibre4b.est_F2L2_4 with LOOK = 3; strict op
count) and the full trace FX (fibre4, exact up to cubature).
Models: zoo seeds 0-4; ARC protocol (absolute sigma 0.05, K copies of the best-of-5 seed, noise seed 0);
leaderboard protocol (relative sigma 0.0025-0.02, 6 copies each = leaderboard.models(4,4)).
Truth: 1e8-sample MC per model (1e7 for the off-regime (4,4) data) (own; rng seed from the model name), with its standard error; T87's exact values
for the zoo seeds are used as a cross-check where available.
Counts: distinct gate patterns in 4e6 samples (lower bound on the linear regions) and exact 2-D boundary-face cells.
Results are saved after every model (resumable). usage: OMP_NUM_THREADS=1 python run_t4.py --H 8 --K 8 --nrel 3 [--only seed] [--tiers F2L3,FX] [--nmc 1e8]"""
import json
import os
import sys
import time
import zlib

import numpy as np

import fibre4
import fibre4b
from count_regions import patterns
from fiber3 import Ops
from zoo import best_seed, leaderboard_models, mc_acc, noisy_abs, zoo_model

HERE = os.path.dirname(os.path.abspath(__file__))



def models(H, K, nrel):
    ms = {k: v for k, v in leaderboard_models(H, 4).items() if k.startswith("seed") or int(k.rsplit("_", 1)[1]) < nrel}
    for k, m in enumerate(noisy_abs(zoo_model(H, 4, best_seed(H, 4)), K, 0.05, 0)):
        ms[f"abs0.05_n0_{k}"] = m
    return ms


def main(H, K, nrel, tiers, only, nmc):
    OUT = os.path.join(HERE, f"results_H{H}_T4.json")
    R = json.load(open(OUT)) if os.path.exists(OUT) else {"rows": {}}
    rows = R["rows"]
    for name, m in models(H, K, nrel).items():
        if only and not name.startswith(only):
            continue
        r = rows.setdefault(name, {})
        if "truth" not in r:
            rng = np.random.default_rng(zlib.crc32(name.encode()))
            n = nmc; p = mc_acc(m, n, rng)
            r["truth"] = p; r["truth_se"] = float(np.sqrt(p * (1 - p) / n))
        if "patterns" not in r:
            r["patterns"] = patterns(m, 4_000_000, np.random.default_rng(1))
            r["face_cells"] = sum(len(fibre4.cells_on_face(m, poly, fam["B"], Ops())) for fam in fibre4.FAMS4
                                  for poly in fam["polys"])
        for tier in tiers:
            if tier in r:
                continue
            o = Ops(); inf = {}; t = time.time()
            e = fibre4b.est_F2L2_4(m, o, inf) if tier == "F2L3" else fibre4.est_fibre4(m, tier, o, inf)
            r[tier] = {"est": e, "ops": o.n, "traces": inf["traces"], "pieces": inf["pieces"], "sec": time.time() - t}
            print(f"{name:14s} {tier}: est {e:.6f} truth {r['truth']:.6f} (se {r['truth_se']:.1e}) err {e - r['truth']:+.2e} "
                  f"ops {o.n / 1e6:.1f}M patterns {r['patterns']} face cells {r['face_cells']}", flush=True)
        json.dump(R, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=8); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--nrel", type=int, default=3); ap.add_argument("--tiers", default="F2L3")
    ap.add_argument("--only", default=""); ap.add_argument("--nmc", type=int, default=10 ** 8)
    a = ap.parse_args()
    main(a.H, a.K, a.nrel, a.tiers.split(","), a.only, a.nmc)
