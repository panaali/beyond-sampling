"""T386 check: dealias.est_grad with the K3 rule reproduces T372's K3 estimate and gradient (dirs_<model>.json)."""
import sys
import time

import numpy as np

import common
import dealias

for nm in sys.argv[1].split(","):
    m, t0, U, D = common.case(nm); t = time.time(); r = dealias.est_grad(m, t0)
    print(nm, "est", r["est"], "T372", D["est"], "max rel grad diff", float(np.abs(r["grad"] - D["grad"]).max() / np.abs(D["grad"]).max()),
          "C_walk", round(r["C_walk"]), "n_end", r["n_end"], "sec", round(time.time() - t), flush=True)
