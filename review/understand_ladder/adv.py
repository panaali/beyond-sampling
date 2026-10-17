"""T97 adversarial search against F2L2: find weights whose error set has islands that F2L2's fibre walks miss
(F2L2 is one-sided: a missed island makes the estimate too HIGH). Objective = per-model MSE ratio vs sampling at
F2L2's own (author-counted) ops, i.e. (est - exact)^2 / (p(1-p) / (ops / ops_per_sample)).
Phase 1: random absolute perturbations of each zoo seed (sigma in 0.05, 0.1, 0.2, 0.4).
Phase 2: greedy hill-climb from the worst phase-1 model (sigma 0.03 steps, accept if objective grows).
Also records LOOK = 1 and 3 on the final worst models.
usage: OMP_NUM_THREADS=1 python adv.py H n1 n2 seed  -> adv_H{H}.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_ladder"))
import fibre  # noqa: E402
import rtruth as R  # noqa: E402


def evaluate(m, look=2):
    H = len(m[0])
    ex, nr, _ = R.exact(m)
    fibre.LOOK = look; o = fibre.Ops(); e = fibre.est_fibre2(m, o, {})
    o1 = 3 * H * (H + 1) + 3 * H
    ratio = (e - ex) ** 2 / (max(ex * (1 - ex), 1e-12) / (o.n / o1))
    return dict(truth=ex, est=e, err=e - ex, ops=o.n, ratio=ratio, regions=nr)


if __name__ == "__main__":
    H, n1, n2, sd = (int(a) for a in sys.argv[1:5])
    rng = np.random.default_rng(sd)
    t0 = time.time()
    zoo = [R.load(H, 3, s)[0] for s in range(5)]
    log = []
    best = None
    for i in range(n1):
        s = i % 5; sig = (0.05, 0.1, 0.2, 0.4)[(i // 5) % 4]
        m = tuple(w + sig * rng.standard_normal(w.shape) for w in zoo[s])
        r = evaluate(m); r.update(phase=1, seed=s, sigma=sig); log.append(r)
        if best is None or r["ratio"] > best[1]["ratio"]:
            best = (m, r)
    print(f"H={H} phase1 {n1} models {time.time() - t0:.0f}s: worst ratio {best[1]['ratio']:.3g} err {best[1]['err']:+.2e} "
          f"(seed {best[1]['seed']} sigma {best[1]['sigma']}); n ratio>=1: {sum(r['ratio'] >= 1 for r in log)}; "
          f"n |err|>1e-10: {sum(abs(r['err']) > 1e-10 for r in log)}; n err<-1e-10: {sum(r['err'] < -1e-10 for r in log)}", flush=True)
    m, r = best
    for j in range(n2):
        m2 = tuple(w + 0.03 * rng.standard_normal(w.shape) for w in m)
        r2 = evaluate(m2); r2.update(phase=2, step=j); log.append(r2)
        if r2["ratio"] > r["ratio"]:
            m, r = m2, r2
    print(f"H={H} phase2 {n2} steps {time.time() - t0:.0f}s: worst ratio {r['ratio']:.3g} err {r['err']:+.2e} acc {r['truth']:.4f}", flush=True)
    looks = {L: evaluate(m, L) for L in (1, 2, 3, 4, 6)}
    for L, x in looks.items():
        print(f"   final worst model LOOK={L}: err {x['err']:+.2e} ops {x['ops']} ratio {x['ratio']:.3g}")
    json.dump(dict(H=H, log=log, worst=dict(model=[w.tolist() for w in m], **r),
                   looks={str(L): x for L, x in looks.items()}), open(os.path.join(HERE, f"adv_H{H}.json"), "w"), indent=1)
