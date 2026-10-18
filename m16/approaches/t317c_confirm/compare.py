"""T320: my instrumented estimator vs frozen estimator_c (imported only here) on the 10 T297 DEV2 models."""
import json
import os
import sys
import numpy as np
import cont_indep as MINE
if len(sys.argv) > 1:
    MINE.ROOT_RULE = sys.argv[1]

Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
truth = {k: v["acc"] for k, v in json.load(open("/home/user/dl/dev2/truths.json"))["truths"].items()}
here = os.getcwd()
FROZEN = "/home/user/t218/m16/approaches/r1_cont"
sys.path.insert(0, FROZEN); os.chdir(FROZEN)
import estimator_c as F  # noqa: E402
os.chdir(here)
rows = {}
for i in range(0, 80, 8):
    nm = str(Z["names"][i]); W = tuple(Z[k][i].astype(float) for k in ("Wi", "Wh", "Wo"))
    a, f, c = MINE.est(W, return_count=True)
    os.chdir(FROZEN); b, g = F.est(W, None); os.chdir(here)
    rows[nm] = dict(mine=a, mine_flops=f, frozen=b, frozen_flops=g, truth=truth[nm], mine_breakdown=c.by)
    print(nm, "est mine %.6f frozen %.6f diff %+.5f pp | FPE mine %.0f frozen %.0f ratio %.3f" % (a, b, 100 * (a - b), f / 5948, g / 5948, f / g), flush=True)
json.dump(rows, open("results/compare.json" if MINE.ROOT_RULE == "direction" else "results/compare_v2.json", "w"), indent=1)
d = np.array([100 * (r["mine"] - r["frozen"]) for r in rows.values()])
fm = np.mean([r["mine_flops"] for r in rows.values()]); ff = np.mean([r["frozen_flops"] for r in rows.values()])
print(f"max |est diff| {np.abs(d).max():.2e} pp; mean C mine {fm/5948:.0f} FPE, frozen {ff/5948:.0f} FPE, ratio {fm/ff:.3f}")
