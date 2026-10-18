"""T359 probe: cost share of the walk beyond T_hi. Fixed-walk S3 (t354.est_seg, K=3) with tmax = T_hi.
usage: python thi_probe.py MODEL THI -> results_thi_probe_MODEL_THI.json"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t340c as TC  # noqa: E402
import t354 as S  # noqa: E402
import walkfix as WF  # noqa: E402

TC.rare.walk3 = lambda model, x0, d, k, k3, tmax, max_cells=5000: WF.walk3_fixed(TC.G, model, x0, d, k, k3, tmax, max_cells)[:5]
nm, thi = sys.argv[1], float(sys.argv[2]); t0 = time.time()
r, r2, f = S.est_seg(TC.cf.lb3_models()[nm], 3, tmax=thi)
out = dict(model=nm, T_hi=thi, est={str(k): v for k, v in r.items()}, est2={str(k): v for k, v in r2.items()}, C=f / TC.cf.FPE, sec=time.time() - t0)
print(json.dumps(out)); json.dump(out, open(os.path.join(HERE, f"results_thi_probe_{nm}_{thi}.json"), "w"), indent=1)
