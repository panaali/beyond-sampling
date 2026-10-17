"""T85 (copy of eval75.py): frozen T85 recipe vs T75 recipe on FRESH noise seeds 40-43 (never used by T63, T74 or T75 development) for every zoo seed
0-4 of M_4,3, plus the clean models. Truth: arc_m43_repro/exact.py. Compares the frozen T75 recipe with T63 tiers B, C.
Ops: est75 counts strictly (T74 convention); exact.py is also recounted strictly for the comparison.
usage: OMP_NUM_THREADS=1 python eval75.py [n_per_seed=256]   -> eval85_43.json, eval85_43.log"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
import numpy as np
import exact as EX
import est75 as E75
import est_mech as EM
from m43 import Ops, OPS_PER_SAMPLE
from zoo43 import zoo_model, noisy_copies

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
SEEDS = (40, 41, 42, 43)
B = dict()
C = dict(strips="all", resolve=("c01", "c02", "c12"), pair_rule="arc", arc_fams=("c01", "c02", "c12"))


def exact_strict(m):
    o = Ops(); oc, of = EX.clip, EX.frac
    EX.clip, EX.frac = E75.clip, E75.frac
    try:
        a = EX.exact_acc(m, o)
    finally:
        EX.clip, EX.frac = oc, of
    return a, o.n


def main():
    t0 = time.time(); out = {"noise_seeds": SEEDS, "n_per_seed": N, "zoo": {}}
    for zs in range(5):
        base = zoo_model(zs)
        models = [base] + [m for ns in SEEDS for m in noisy_copies(base, N, 0.05, ns)]
        rows = []
        for m in models:
            o1 = Ops(); tr = EX.exact_acc(m, o1); _, ostrict = exact_strict(m)
            E75.STATS.update(cut=0, resolved=0, cells=0)
            o = Ops(); r = E75.est75(m, o, **E75.RECIPE)
            E75.STATS.update(cut=0, resolved=0, cells=0)
            ob = Ops(); b = E75.est75(m, ob, **E75.RECIPE85); st = dict(E75.STATS); oc_ = Ops(); c = r
            rows.append(dict(truth=tr, exact_ops=o1.n, exact_ops_strict=ostrict, t75=r, t75_ops=o.n, stats=st,
                             B=b, B_ops=ob.n, C=c, C_ops=oc_.n))
        clean, noisy = rows[0], rows[1:]
        tr = np.array([r["truth"] for r in noisy]); pq = float(np.mean(tr * (1 - tr)))
        summ = {"clean": clean, "mean_truth": float(tr.mean()), "pq": pq}
        for name, key, okey in (("T75", "t75", "t75_ops"), ("T85", "B", "B_ops")):
            e = np.array([r[key] for r in noisy]) - tr; om = float(np.mean([r[okey] for r in noisy]))
            mse_s = pq / (om / OPS_PER_SAMPLE)
            per = []
            for i, ns in enumerate(SEEDS):
                ee = e[i * N:(i + 1) * N]; tt = tr[i * N:(i + 1) * N]
                per.append(float(np.mean(ee ** 2) / (np.mean(tt * (1 - tt)) / (om / OPS_PER_SAMPLE))))
            boot = np.random.default_rng(0)            # bootstrap CI of the ratio (analysis only, not the estimator)
            bs = [np.mean(e[idx] ** 2) / (np.mean(tr[idx] * (1 - tr[idx])) / (om / OPS_PER_SAMPLE))
                  for idx in (boot.integers(0, len(e), len(e)) for _ in range(500))]
            ce = clean[key] - clean["truth"]; pc = clean["truth"] * (1 - clean["truth"])
            summ[name] = {"rms": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()), "max_abs": float(np.abs(e).max()),
                          "ops": om, "mse_ratio": float(np.mean(e ** 2) / mse_s), "ratio_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                          "ratio_per_noise_seed": per, "clean_err": float(ce),
                          "clean_ratio": float(ce ** 2 / (pc / (clean[okey] / OPS_PER_SAMPLE)))}
        summ["exact_ops"] = float(np.mean([r["exact_ops"] for r in noisy])); summ["exact_ops_strict"] = float(np.mean([r["exact_ops_strict"] for r in noisy]))
        st = [r["stats"] for r in noisy]
        summ["t85_cells_mean"] = float(np.mean([s["cells"] for s in st]))
        summ["t85_resolved_frac"] = float(np.sum([s["resolved"] for s in st]) / max(1, np.sum([s["cut"] for s in st])))
        out["zoo"][zs] = summ
        print(f"zoo seed {zs}: mean acc {tr.mean():.3f}; exact ops {summ['exact_ops']:.0f} (strict {summ['exact_ops_strict']:.0f}); "
              f"T85 cells/model {summ['t85_cells_mean']:.0f}, resolved {summ['t85_resolved_frac']:.0%} of cutting gates")
        for name in ("T75", "T85"):
            s = summ[name]
            print(f"   {name:4s} clean err {100*s['clean_err']:+.2f}pp (ratio {s['clean_ratio']:.2f}) | noisy rms {100*s['rms']:.2f}% bias {100*s['bias']:+.2f}% "
                  f"max {100*s['max_abs']:.1f}% ops {s['ops']:.0f} MSE ratio {s['mse_ratio']:.3f} [{s['ratio_ci'][0]:.3f},{s['ratio_ci'][1]:.3f}] "
                  f"per noise seed {[round(x, 3) for x in s['ratio_per_noise_seed']]}", flush=True)
        json.dump(out, open(os.path.join(HERE, "eval85_43.json"), "w"), indent=1, default=float)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
