"""T371 estimates: the frozen H-R1 estimator (K3 = T340c with the walk fix and T354 segment start, T359 cut T_hi = 3.5),
called exactly as T364's deep.py calls it: est_seg(model, 3, t0s, t2s=(), tmax=T_hi). DEDUCTIVE: no random inputs.
Code is read-only from claude/t218-tie-tail-labels @5b4a1250 (env T371_R1C = its m16/approaches/r1_cont) and
T340_SRC = claude/t286-cheap-frontier @620eb47b; sha256s asserted against r1_cont/t366_freeze_sha256.txt.
usage: python est_run.py SHARD NSHARD -> results/est_SHARD.json"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as MP  # noqa: E402

T0S = (1.0, 1.25, 1.5)
T_HI = 3.5
R1C = os.environ["T371_R1C"]; SRC = os.environ["T340_SRC"]
SHA = {os.path.join(R1C, "t340c.py"): "7e6a749d67284d4e634bfb302087b9559da4956a6d5e6a1d29dd381959ec9236",
       os.path.join(R1C, "t354.py"): "abed36c8db9dbecfe36b6e5ab41090363778a8d1485f0e7599e6f23298232dde",
       os.path.join(R1C, "walkfix.py"): "8e21b6a11c7d9151d2f02f84d1e09f5225b1e605501aa0e57aeb398a1d432d7e",
       os.path.join(SRC, "m16/approaches/rare_events/rare.py"): "09f89a8b9eb23c589976e6f52220ca66a2cad9f24a818dbaa9353e8643cb4fa9",
       os.path.join(SRC, "m16/approaches/closure_floor/cont_g.py"): "d34dde3699c62d5784b2051d6ce8446ca03a0180e1ef6558ef117e3525b71605"}
for fn, h in SHA.items():
    assert hashlib.sha256(open(fn, "rb").read()).hexdigest() == h, fn
sys.path.insert(0, R1C)
import t340c as TC  # noqa: E402
import t354 as SG  # noqa: E402
import walkfix as WF  # noqa: E402

TC.rare.walk3 = lambda model, x0, d, k, k3, tmax, max_cells=5000: WF.walk3_fixed(TC.G, model, x0, d, k, k3, tmax, max_cells)[:5]
DEDUCTIVE = True

if __name__ == "__main__":
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    P = MP.panel(); names = sorted(P)[shard::nsh]
    fn = os.path.join(HERE, "results", f"est_{shard}.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    for nm in names:
        if nm in out:
            continue
        t = time.time(); r, _, f = SG.est_seg(P[nm], 3, t0s=T0S, t2s=(), tmax=T_HI)
        out[nm] = dict(est={str(k): float(v) for k, v in r.items()}, C_fpe=f / TC.cf.FPE, T_hi=T_HI, sec=time.time() - t)
        json.dump(out, open(fn, "w"), indent=1)
        print(nm, {k: f"{v:.3e}" for k, v in out[nm]["est"].items()}, "C %.0f" % out[nm]["C_fpe"], "%.0fs" % out[nm]["sec"], flush=True)
