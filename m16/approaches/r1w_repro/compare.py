"""T312 comparison: my re-implementation vs the frozen R1W V1 (imported only here), all 80 DEV2 models."""
import json
import os
import sys
import time
import numpy as np
import r1w_indep as MINE

FROZEN_DIR = "/home/user/t218/m16/approaches/r1_walk"
Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
truth = {k: v["acc"] for k, v in json.load(open("/home/user/dl/dev2/truths.json"))["truths"].items()}
sys.path.insert(0, FROZEN_DIR)
os.chdir(FROZEN_DIR)
import estimator as FROZEN  # noqa: E402
rows = {}
for k, nm in enumerate(Z["names"]):
    m = (Z["Wi"][k].astype(float), Z["Wh"][k].astype(float), Z["Wo"][k].astype(float))
    t0 = time.time(); e1, f1 = MINE.est(m); t1 = time.time()
    e2, f2 = FROZEN.est(m, None); t2 = time.time()
    rows[str(nm)] = dict(mine=e1, mine_flops=f1, frozen=e2, frozen_flops=f2, truth=truth[str(nm)],
                         cpu_mine=t1 - t0, cpu_frozen=t2 - t1)
    print(nm, "mine %.5f frozen %.5f diff %+.3f pp | FPE mine %.0f frozen %.0f" % (e1, e2, 100 * (e1 - e2), f1 / 5948, f2 / 5948), flush=True)
os.chdir(os.path.dirname(os.path.abspath(MINE.__file__)))
json.dump(rows, open("results/compare.json", "w"), indent=1)
d = np.array([100 * (r["mine"] - r["frozen"]) for r in rows.values()])
fr = np.array([r["mine_flops"] / r["frozen_flops"] for r in rows.values()])
for lab, key in (("mine", "mine"), ("frozen", "frozen")):
    e = np.array([100 * (r[key] - r["truth"]) for r in rows.values()])
    C = np.mean([r[key + "_flops"] for r in rows.values()]) / 5948
    print(f"{lab}: bias {e.mean():+.3f} RMS {np.sqrt((e**2).mean()):.3f} pp, mean C {C:.0f} FPE, F {np.mean(e**2)*1e-4*C/0.0725:.2f}")
print(f"diff mine-frozen: max |d| {np.abs(d).max():.3f} pp, mean {d.mean():+.3f}, RMS {np.sqrt((d**2).mean()):.3f}; within .1 pp: {(np.abs(d) <= .1).sum()}/80")
print(f"FLOP ratio mine/frozen: mean {fr.mean():.3f}, range {fr.min():.3f}-{fr.max():.3f}")
