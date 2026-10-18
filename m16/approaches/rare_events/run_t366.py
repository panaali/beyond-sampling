"""T366: run the T218 child's frozen estimator (claude/t218-tie-tail-labels @5b4a1250 r1_cont/estimator_t366.py, hashes
checked; T340_SRC = claude/t286-cheap-frontier @620eb47b) on the fresh models -> results/t366_est.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = "/tmp/claude-0/-home-user-clean-neurons/18e70fa5-7ff5-5fac-af53-4fea2351128f/scratchpad"
os.environ["T340_SRC"] = os.path.join(S, "src620")
sys.path.insert(0, os.path.join(S, "r1c366", "m16", "approaches", "r1_cont"))
import estimator_t366 as E  # noqa: E402

z = np.load(os.path.join(HERE, "ext", "dev3", "weights.npz")); names = list(map(str, z["names"])); out = {}
for nm in ("dev3_s528", "dev3_s529", "dev3_s530", "dev3_s531"):
    i = names.index(nm); m = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
    t = time.time(); r, f = E.est(m, (2.35, 2.75))
    out[nm] = dict(est={str(k): float(v) for k, v in r.items()}, C=f / 5948.0, sec=time.time() - t)
    print(nm, out[nm], flush=True)
json.dump(out, open(os.path.join(HERE, "results", "t366_est.json"), "w"), indent=1)
