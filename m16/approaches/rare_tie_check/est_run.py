"""T403 step 2: the frozen estimator (via frozen_est.est_split, unmodified walk_seg / rare.mass) on every panel model at
the fixed t0 grid; one call per model gives all t0. -> results/est_SHARD.json. DEDUCTIVE: no random model inputs.
usage: python est_run.py SHARD NSHARD [T0,T0,...]"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frozen_est as FE  # noqa: E402
import models as MP  # noqa: E402

GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)

if __name__ == "__main__":
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    grid = tuple(float(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else GRID
    tag = "" if grid == GRID else "_ext"
    fn = os.path.join(HERE, "results", f"est{tag}_{shard}.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    for nm in MP.NAMES[shard::nsh]:
        if nm in out:
            continue
        t = time.time(); tot, tie, f = FE.est_split(MP.load(nm), grid)
        out[nm] = dict(est={str(k): tot[k] for k in grid}, tie={str(k): tie[k] for k in grid}, C_fpe=f / FE.FPE, sec=time.time() - t)
        json.dump(out, open(fn, "w"), indent=1)
        print(nm, {k: "%.2e/%.2f" % (v, out[nm]["tie"][k] / v if v > 0 else -1) for k, v in out[nm]["est"].items()},
              "C %.0f" % out[nm]["C_fpe"], "%.0fs" % out[nm]["sec"], flush=True)
