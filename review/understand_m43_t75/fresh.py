"""T84 items 1 and 4a: frozen T75 recipe on never-used noise seeds, all 5 zoo seeds.
Truth: arc_m43_repro/exact.py (checked against own 1e7 MC in truth_mc.py). Estimate: the reviewed est75 (frozen
RECIPE, its strict charge) and this review's independent re-implementation recipe.py (must agree to ~1e-12).
usage: [ZOO=4] OMP_NUM_THREADS=1 python fresh.py SIGMA N SEED [SEED ...]  -> fresh_<sigma>_<seeds>.json, prints table"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_m43"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import exact as EX
import est75 as E75
from m43 import Ops
import recipe as R
from mine import zoo_w, noisy_w

OPS_PER_SAMPLE = 72


def ratio(e, tr, ops):
    return float(np.mean(e ** 2) / (np.mean(tr * (1 - tr)) / (ops / OPS_PER_SAMPLE)))


def main():
    sigma = float(sys.argv[1]); n = int(sys.argv[2]); seeds = [int(s) for s in sys.argv[3:]]
    t0 = time.time(); out = {"sigma": sigma, "n": n, "seeds": seeds, "zoo": {}}
    zoos = [int(z) for z in os.environ.get("ZOO", "0,1,2,3,4").split(",")]
    for zs in zoos:
        base = zoo_w(zs)
        models = [m for ns in seeds for m in noisy_w(base, n, sigma, ns)]
        tr, est, mine, ops, tcpu = [], [], [], [], 0.0
        for m in models:
            tr.append(EX.exact_acc(m))
            o = Ops(); t = time.process_time(); est.append(E75.est75(m, o, **E75.RECIPE)); tcpu += time.process_time() - t
            ops.append(o.n); mine.append(R.recipe(R.as_lists(m)))
        tr, est, mine, ops = map(np.array, (tr, est, mine, ops))
        e = est - tr; om = float(ops.mean())
        boot = np.random.default_rng(1)             # analysis-only bootstrap over copies
        bs = [ratio(e[i], tr[i], om) for i in (boot.integers(0, len(e), len(e)) for _ in range(1000))]
        per = [ratio(e[k * n:(k + 1) * n], tr[k * n:(k + 1) * n], om) for k in range(len(seeds))]
        r = dict(mean_truth=float(tr.mean()), pq=float(np.mean(tr * (1 - tr))), rms=float(np.sqrt(np.mean(e ** 2))),
                 bias=float(e.mean()), max_abs=float(np.abs(e).max()), ops=om, mse_ratio=ratio(e, tr, om),
                 ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], per_seed=per,
                 reimpl_maxdiff=float(np.abs(mine - est).max()), cpu_ms=1000 * tcpu / len(models),
                 truths=tr.tolist(), errs=e.tolist(), opsl=ops.tolist())
        out["zoo"][zs] = r
        print(f"zoo {zs}: acc {r['mean_truth']:.3f} rms {100*r['rms']:.2f}% bias {100*r['bias']:+.3f}% max {100*r['max_abs']:.1f}% "
              f"ops {om:.0f} ratio {r['mse_ratio']:.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}] per seed {[round(x, 4) for x in per]} "
              f"| reimpl maxdiff {r['reimpl_maxdiff']:.1e} | est75 cpu {r['cpu_ms']:.0f} ms/model", flush=True)
        tag = "" if len(zoos) == 5 else "_zoo" + "".join(map(str, zoos))
        json.dump(out, open(os.path.join(HERE, f"fresh_{sigma}_{'-'.join(map(str, seeds))}{tag}.json"), "w"))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
