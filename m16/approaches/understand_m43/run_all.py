"""T63: score the mechanism-derived estimator (est_mech) under ARC's M_4,3 protocol and on zoo seeds 0-4.

Truth: exact enumeration (arc_m43_repro/exact.py); noisy-copy truths are read from arc_m43_repro/results.json
(same noisy_models draws). Sampling RMSE at equal ops is sqrt(mean p(1-p) / (ops/72)) (checked empirically in
arc_m43_repro). Writes results.json and prints the README tables.
usage: OMP_NUM_THREADS=1 python run_all.py
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
import numpy as np
from m43 import Ops, Wi0, Wh0, Wo0, noisy_models, OPS_PER_SAMPLE, mc_acc
from exact import exact_acc
from est_mech import est_mech
from diag_family import exact_by_half

TIERS = {
    "T1 pair, anchor strip": dict(strips="anchor", resolve=()),
    "T2 pair, all strips": dict(strips="all", resolve=()),
    "T3 pair + resolved crossover": dict(strips="all", resolve=("c12",)),
    "T4 resolved everywhere": dict(strips="all", resolve=("c01", "c02", "c12")),
}


def zoo_model(seed):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(4, 3, seed=seed)
    sd = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
    return sd["rnn.weight_ih_l0"][:, 0], sd["rnn.weight_hh_l0"], sd["linear.weight"]


def main():
    t0 = time.time()
    R = json.load(open(os.path.join(HERE, "..", "arc_m43_repro", "results.json")))
    out = {"tiers": {}, "zoo": {}, "families": {}}
    noisy = []; truth = []
    for sd in range(4):
        noisy += noisy_models(256, 0.05, seed=sd)
        truth += [r["truth"] for r in R["seeds"][str(sd)]["rows"]]
    truth = np.array(truth); pq = float(np.mean(truth * (1 - truth)))
    seeds = list(range(4))
    zoo = {}
    for s in range(5):
        m = zoo_model(s)
        zoo[s] = (m, exact_acc(m), mc_acc(m, 200_000, np.random.default_rng(5 + s)))
    print("zoo seeds: exact truth / 2e5-sample MC check:", {s: (round(v[1], 5), round(v[2], 5)) for s, v in zoo.items()})
    for name, kw in TIERS.items():
        est = []; ops = []; bad = 0
        for m in noisy:
            o = Ops(); inf = {}
            est.append(est_mech(m, o, inf, **kw)); ops.append(o.n); bad += inf["bad_pairs"] > 0
        est = np.array(est); e = est - truth; ops_m = float(np.mean(ops))
        per_seed = [float(np.sqrt(np.mean(e[256 * s:256 * (s + 1)] ** 2))) for s in seeds]
        mse_s = pq / (ops_m / OPS_PER_SAMPLE); mse_arc = pq / (ops_m / OPS_PER_SAMPLE)
        row = {"rms": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()), "sd": float(e.std()),
               "max_abs": float(np.abs(e).max()), "rms_per_seed": per_seed, "ops_mean": ops_m,
               "ops_max": int(max(ops)), "models_with_bad_anchor_pair": int(bad),
               "sampling_rmse_equal_ops": float(np.sqrt(mse_s)), "mse_ratio": float(np.mean(e ** 2) / mse_s)}
        ob = Ops(); row["base"] = {"est": est_mech((Wi0, Wh0, Wo0), ob, **kw), "ops": ob.n}
        row["zoo"] = {}
        for s, (m, tr, _) in zoo.items():
            oz = Ops(); ez = est_mech(m, oz, **kw)
            row["zoo"][s] = {"truth": tr, "est": ez, "err": ez - tr, "ops": oz.n}
        out["tiers"][name] = row
        print(f"{name:32s} rms {row['rms']:.4f} bias {row['bias']:+.4f} max|e| {row['max_abs']:.4f} ops {ops_m:.0f} "
              f"(max {row['ops_max']}) samp_rmse@ops {row['sampling_rmse_equal_ops']:.4f} MSE ratio {row['mse_ratio']:.4f} "
              f"bad {bad} per-seed {[round(x, 4) for x in per_seed]}")
        print(f"{'':32s} base err {row['base']['est'] - R['base']['exact']:+.5f}; zoo errs "
              f"{ {s: round(v['err'], 4) for s, v in row['zoo'].items()} } ops {[v['ops'] for v in row['zoo'].values()]}")
    # where the residual of the default tier comes from (exact mass per family, as a diagnostic)
    fam_ex = {}; fam_es = {}
    for m in noisy:
        d = exact_by_half(m); inf = {}; est_mech(m, info=inf, **TIERS["T3 pair + resolved crossover"])
        for k in d:
            fam_ex.setdefault(str(k), []).append(d[k]); fam_es.setdefault(str(k), []).append(inf["err_by"].get(k, 0.0))
    for k in sorted(fam_ex):
        a, b = np.array(fam_ex[k]), np.array(fam_es[k])
        out["families"][k] = {"exact_mean": float(a.mean()), "model_mean": float(b.mean()),
                              "bias": float((b - a).mean()), "rms": float(np.sqrt(np.mean((b - a) ** 2)))}
        print(f"  family {k:16s} exact {a.mean():.4f} model {b.mean():.4f} bias {(b-a).mean():+.5f} rms {np.sqrt(np.mean((b-a)**2)):.4f}")
    out["pq_mean"] = pq; out["mean_truth"] = float(truth.mean()); out["seconds"] = time.time() - t0
    out["constant_baseline"] = {"est": R["base"]["exact"], "rms": float(np.sqrt(np.mean((R["base"]["exact"] - truth) ** 2)))}
    print("constant baseline (original model's accuracy for every copy): rms", round(out["constant_baseline"]["rms"], 4))
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
