"""T354: rerun with the simultaneous-flip fix (walkfix.walk3_fixed patched in for T340's rare.walk3): T340's B, T340c
(K3, K5) and T354's segment walks (S3, S5). usage: python run_fixed.py B,C3,S3 -> results_fixed_<tags>.json"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t340c as TC  # noqa: E402
import t354 as S  # noqa: E402
import walkfix as WF  # noqa: E402

rare = TC.rare; G = TC.G
rare.walk3 = lambda model, x0, d, k, k3, tmax, max_cells=5000: WF.walk3_fixed(G, model, x0, d, k, k3, tmax, max_cells)[:5]

tags = sys.argv[1].split(","); ms = TC.cf.lb3_models(); fn = os.path.join(HERE, f"results_fixed_{'_'.join(tags)}.json"); out = {}
for tag in tags:
    for nm in TC.NAMES:
        t0 = time.time()
        if tag == "B":
            r, r2, f, _ = rare.est(ms[nm], variant="B")
        elif tag[0] == "C":
            r, r2, f = TC.est_c(ms[nm], int(tag[1:]))
        else:
            r, r2, f = S.est_seg(ms[nm], int(tag[1:]))
        out[f"{nm}:{tag}"] = dict(est={str(k): v for k, v in r.items()}, est2={str(k): v for k, v in r2.items()}, C=f / TC.cf.FPE)
        print(tag, nm, {k: f"{v:.3e}" for k, v in r.items()}, {k: f"{v:.3e}" for k, v in r2.items()}, f"C {f / TC.cf.FPE:.0f}",
              round(time.time() - t0), flush=True)
        json.dump(out, open(fn, "w"), indent=1)
