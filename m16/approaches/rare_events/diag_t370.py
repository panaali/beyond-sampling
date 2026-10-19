"""T370-diag (diagnostic, not a go): S14 E1 on M_16,10 (claude/t218-tie-tail-labels @87028552 t370.py, sha256 asserted).
usage: python diag_t370.py t0,t0,... -> results/t370diag_<t0s>.json"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S0 = "/tmp/claude-0/-home-user-clean-neurons/18e70fa5-7ff5-5fac-af53-4fea2351128f/scratchpad"
R1C = os.path.join(S0, "r1c370", "m16", "approaches", "r1_cont")
SHA = {"t370.py": "155ce0f643c69e8a7f72b17f5ad92bfd44372e252aee8b60a91864c6d1531e2d",
       "t366.py": "997a1b60052757b3048c9baf846245654e8bdc6d2c75f6e5224a746b18064774",
       "t340c.py": "7e6a749d67284d4e634bfb302087b9559da4956a6d5e6a1d29dd381959ec9236",
       "t354.py": "abed36c8db9dbecfe36b6e5ab41090363778a8d1485f0e7599e6f23298232dde",
       "walkfix.py": "8e21b6a11c7d9151d2f02f84d1e09f5225b1e605501aa0e57aeb398a1d432d7e",
       "cont_e.py": "50dd1c9b85512e23f037171d19de03527f8f5989758cfb74d93b441de48fda2b"}
for f, h in SHA.items():
    assert hashlib.sha256(open(os.path.join(R1C, f), "rb").read()).hexdigest() == h, f
os.environ["T340_SRC"] = os.path.join(S0, "src620")
sys.path.insert(0, R1C)
import t370  # noqa: E402

t0s = tuple(float(t) for t in sys.argv[1].split(","))
m = t370.TC.cf.lb3_models()["orig"]
t = time.time(); est, flops = t370.est_e1(tuple(np.asarray(w, np.float64) for w in m), t0s, None, closure="S14")
out = dict(commit="87028552", sha256=SHA, t0s=t0s, est={str(k): v for k, v in est.items()}, C=flops / 5948, sec=time.time() - t)
print(out, flush=True)
json.dump(out, open(os.path.join(HERE, "results", f"t370diag_{'_'.join(map(str, t0s))}.json"), "w"), indent=1)
