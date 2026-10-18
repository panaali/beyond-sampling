"""1e7-sample MC truth (rng seed 2026, chunks of 5e5) for zoo(8,4,s) s=0..4 and final3's 4 noisy copies of zoo0. -> mc_H8_T4.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
from zoo44 import zoo, mc_acc
from leaderboard import _noisy
ms = [(f"zoo{s}", zoo(8, 4, s)) for s in range(5)] + [(f"zoo0_noisy{i}", m) for i, m in enumerate(_noisy(zoo(8, 4, 0), 4, 0.005, 7))]
out = {n: mc_acc(m, 10_000_000, np.random.default_rng(2026), chunk=500_000) for n, m in ms}
json.dump(out, open(os.path.join(HERE, "mc_H8_T4.json"), "w"), indent=1); print(out)
