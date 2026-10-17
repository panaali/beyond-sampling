"""T100 rung (H=4, T=4) scoring for the fibre recipe (frozen F2L3 = fibre4b.est_F2L2_4 with LOOK = 3; strict op
count) and the full trace FX (fibre4, exact up to cubature).
Models: zoo seeds 0-4; ARC protocol (absolute sigma 0.05, K copies of the best-of-5 seed, noise seed 0);
leaderboard protocol (relative sigma 0.0025-0.02, 6 copies each = leaderboard.models(4,4)).
Truth: 1e7-sample MC per model (own; rng seed from the model name), with its standard error; T87's exact values
for the zoo seeds are used as a cross-check where available.
Counts: distinct gate patterns in 4e6 samples (lower bound on the linear regions) and exact 2-D boundary-face cells.
Results are saved after every model (resumable). usage: OMP_NUM_THREADS=1 python run_t4.py [K] [tiers]"""
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
OUT = os.path.join(HERE, "results_H4_T4.json")


def models(K):
    ms = leaderboard_models(4, 4)
    for k, m in enumerate(noisy_abs(zoo_model(4, 4, best_seed(4, 4)), K, 0.05, 0)):
        ms[f"abs0.05_n0_{k}"] = m
    return ms


def main(K, tiers):
    R = json.load(open(OUT)) if os.path.exists(OUT) else {"rows": {}}
    rows = R["rows"]
    for name, m in models(K).items():
        r = rows.setdefault(name, {})
        if "truth" not in r:
            rng = np.random.default_rng(zlib.crc32(name.encode()))
            n = 10 ** 7; p = mc_acc(m, n, rng)
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
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    tiers = sys.argv[2].split(",") if len(sys.argv) > 2 else ["F2L3", "FX"]
    main(K, tiers)
