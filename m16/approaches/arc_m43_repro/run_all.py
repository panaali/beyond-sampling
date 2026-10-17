"""T36: score ARC's two M_4,3 estimators, plain sampling and exact enumeration under ARC's protocol.

Protocol (Hilton notes, Dec 2024): 256 noisy copies of zoo_2nd_argmax(4,3), zero-mean Gaussian noise of absolute
s.d. 0.05 on all 32 weights; error = estimate - exact accuracy; compute in ops (mult-add = 1; 72 per sample).

usage: OMP_NUM_THREADS=1 python run_all.py [--seeds 0,1,2,3] [--mc 1000000]
writes results.json next to this file and prints the tables quoted in README.md.
"""
import argparse
import json
import os
import time

import numpy as np

from m43 import OPS_PER_SAMPLE, Ops, Wi0, Wh0, Wo0, forward, mc_acc, noisy_models, HERE
from exact import exact_acc
from est_arc import ARC_OPS_A, ARC_OPS_B, est_A, est_B


def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--mc", type=int, default=1_000_000, help="MC samples per model to cross-check exact truth")
    ap.add_argument("--k", type=int, default=256)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {"protocol": {"sigma_abs": 0.05, "n_models": args.k, "ops_per_sample": OPS_PER_SAMPLE}, "seeds": {}}
    base = (Wi0, Wh0, Wo0)
    ob = Ops(); out["base"] = {"exact": exact_acc(base, ob), "exact_ops": ob.n, "A": est_A(base), "B": est_B(base)}
    for sd in seeds:
        t0 = time.time()
        models = noisy_models(args.k, 0.05, seed=sd)
        rows = []
        for i, m in enumerate(models):
            oe, oa = Ops(), Ops(); st = {}; ia = {}; ib = {}
            tr = exact_acc(m, oe, st)
            a = est_A(m, ops=oa, info=ia)
            b, b0 = est_B(m, info=ib, return_parts=True)
            r = {"truth": tr, "exact_ops": oe.n, "leaves": st["leaves"], "A": a, "A_ops": oa.n,
                 "A_checks_ok": ia["checks_ok"], "B": b, "B_reduced_only": b0,
                 "A_q": {str(q): est_A(m, q=q) for q in (0.05, 0.1, 0.3, 0.5)}}
            if args.mc and sd == seeds[0]:
                r["mc"] = mc_acc(m, args.mc, np.random.default_rng(10_000 + i))
            rows.append(r)
        tr = np.array([r["truth"] for r in rows])
        A = np.array([r["A"] for r in rows]); B = np.array([r["B"] for r in rows])
        B0 = np.array([r["B_reduced_only"] for r in rows])
        pq = tr * (1 - tr)
        summ = {
            "mean_acc": float(tr.mean()), "rms_sqrt_pq": float(np.sqrt(pq.mean())),
            "A_rms": rms(A - tr), "A_bias": float((A - tr).mean()), "A_sd": float((A - tr).std()),
            "B_rms": rms(B - tr), "B_bias": float((B - tr).mean()), "B_sd": float((B - tr).std()),
            "B_reduced_only_rms": rms(B0 - tr), "B_reduced_only_bias": float((B0 - tr).mean()),
            "A_checks_ok_frac": float(np.mean([r["A_checks_ok"] for r in rows])),
            "A_ops_measured_mean": float(np.mean([r["A_ops"] for r in rows])),
            "exact_ops_mean": float(np.mean([r["exact_ops"] for r in rows])),
            "exact_ops_max": int(max(r["exact_ops"] for r in rows)),
            "leaves_mean": float(np.mean([r["leaves"] for r in rows])),
            "A_q_rms": {q: rms(np.array([r["A_q"][q] for r in rows]) - tr) for q in rows[0]["A_q"]},
        }
        if "mc" in rows[0]:
            mc = np.array([r["mc"] for r in rows])
            z = (mc - tr) / np.sqrt(pq / args.mc)
            summ["mc_check"] = {"n": args.mc, "z_mean": float(z.mean()), "z_sd": float(z.std()),
                                "max_abs_z": float(np.abs(z).max())}
        # sampling under the same protocol: analytic RMSE at equal ops and an empirical check
        for name, ops in (("A_arc", ARC_OPS_A), ("B_arc", ARC_OPS_B), ("A_measured", summ["A_ops_measured_mean"]),
                          ("exact_measured", summ["exact_ops_mean"])):
            k = ops / OPS_PER_SAMPLE
            summ[f"sampling_rmse_at_{name}"] = float(np.sqrt(pq.mean() / k))
        rng = np.random.default_rng(777 + sd); emp = {}
        for k in (16, 107, 986, 4096):
            est = np.array([(forward(m, x := rng.standard_normal((k, 3))).argmax(1) == np.argsort(x, 1)[:, -2]).mean()
                            for m in models])
            emp[k] = {"rmse": rms(est - tr), "pred": float(np.sqrt(pq.mean() / k))}
        summ["sampling_empirical"] = emp
        summ["seconds"] = time.time() - t0
        out["seeds"][sd] = {"summary": summ, "rows": rows}
        print(f"seed {sd}: " + json.dumps({k: v for k, v in summ.items() if k != "A_q_rms"}, default=float))
        print("   A rms by q:", {q: round(v, 4) for q, v in summ["A_q_rms"].items()}, "(q=0.2:", round(summ["A_rms"], 4), ")")
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
