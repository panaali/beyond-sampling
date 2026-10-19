"""T401: frozen K3 (H-R1; claude/t218-tie-tail-labels @e433919d t340c/t354/walkfix, sha256 asserted) on the rating grid.
Deductive. -> results/t401_est.json"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R1C = "/tmp/claude-0/-home-user-clean-neurons/18e70fa5-7ff5-5fac-af53-4fea2351128f/scratchpad/r1c/m16/approaches/r1_cont"
SHA = {"t340c.py": "7e6a749d67284d4e634bfb302087b9559da4956a6d5e6a1d29dd381959ec9236",
       "t354.py": "abed36c8db9dbecfe36b6e5ab41090363778a8d1485f0e7599e6f23298232dde",
       "walkfix.py": "8e21b6a11c7d9151d2f02f84d1e09f5225b1e605501aa0e57aeb398a1d432d7e"}
for f, h in SHA.items():
    assert hashlib.sha256(open(os.path.join(R1C, f), "rb").read()).hexdigest() == h, f
sys.path.insert(0, HERE)
from deep import S, TC, models  # noqa: E402  (deep patches walk3 with walkfix)

T1 = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
T2 = (0.5, 0.75, 1.0, 1.25, 1.5)
ms = models(); out = {}
for nm in ("orig", "dev3_s529", "dev3_s531"):
    t = time.time(); r, r2, f = S.est_seg(tuple(np.asarray(w, float) for w in ms[nm]), 3, t0s=T1, t2s=T2, tmax=6.0)
    out[nm] = dict(E1={str(k): float(v) for k, v in r.items()}, E2={str(k): float(v) for k, v in r2.items()},
                   C=f / TC.cf.FPE, sec=time.time() - t)
    print(nm, out[nm], flush=True)
json.dump(dict(sha256=SHA, commit="e433919d", rows=out), open(os.path.join(HERE, "results", "t401_est.json"), "w"), indent=1)
