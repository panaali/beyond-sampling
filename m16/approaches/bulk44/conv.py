"""Convergence check on clean (4,4) seed 0: Hopf nu=256 and MC 2e8 (rng 2024). -> conv.json"""
import sys, os, json; import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "understand_m43")); from zoo44 import zoo, mc_acc; import hopf
m = zoo(4, 4, 0); which = sys.argv[1]
r = hopf.est_hopf(m, 256) if which == "hopf" else mc_acc(m, 200_000_000, np.random.default_rng(2024))
print(which, r); json.dump({which: r}, open(f"conv_{which}.json", "w"))
