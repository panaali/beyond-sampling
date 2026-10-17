"""T85b frozen evaluation (rung H=4, T=3): RECIPE85B (scale-invariant step-2 impact + margin-aware step 3) on FRESH
noise seeds 60-63 (not used by T63/T74/T75/T84/T85 development) for zoo seeds 0-4, clean + noisy (abs sigma 0.05).
Truth: arc_m43_repro/exact.py. Strict op count. Includes the permanent invariance test: every 8th model is also
evaluated after a random positive rescaling + permutation of its hidden neurons (must not change the estimate).
usage: OMP_NUM_THREADS=1 python eval85b.py [n_per_seed=256]  -> eval85b.json, eval85b.log"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
import numpy as np
import exact as EX
import est75 as E75
from m43 import Ops, OPS_PER_SAMPLE
from zoo43 import zoo_model, noisy_copies
from rescale_test import rescale

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
SEEDS = (60, 61, 62, 63)


def main():
    t0 = time.time(); out = {"noise_seeds": SEEDS, "n_per_seed": N, "recipe": repr(E75.RECIPE85B), "zoo": {}}
    rng = np.random.default_rng(999)
    for zs in range(5):
        base = zoo_model(zs)
        models = [base] + [m for ns in SEEDS for m in noisy_copies(base, N, 0.05, ns)]
        rows = []; inv = 0.0
        for i, m in enumerate(models):
            tr = EX.exact_acc(m)
            E75.STATS.update(cut=0, resolved=0, cells=0)
            o = Ops(); r = E75.est75(m, o, **E75.RECIPE85B)
            if i % 8 == 0:
                inv = max(inv, abs(E75.est75(rescale(m, rng), Ops(), **E75.RECIPE85B) - r))
            rows.append(dict(truth=tr, est=r, ops=o.n, cells=E75.STATS["cells"], cut=E75.STATS["cut"], resolved=E75.STATS["resolved"]))
        clean, noisy = rows[0], rows[1:]
        tr = np.array([q["truth"] for q in noisy]); e = np.array([q["est"] for q in noisy]) - tr
        om = float(np.mean([q["ops"] for q in noisy])); pq = float(np.mean(tr * (1 - tr)))
        mse_s = pq / (om / OPS_PER_SAMPLE)
        per = [float(np.mean(e[k * N:(k + 1) * N] ** 2) / (np.mean(tr[k * N:(k + 1) * N] * (1 - tr[k * N:(k + 1) * N])) / (om / OPS_PER_SAMPLE)))
               for k in range(len(SEEDS))]
        b = np.random.default_rng(0)
        bs = [np.mean(e[ix] ** 2) / (np.mean(tr[ix] * (1 - tr[ix])) / (om / OPS_PER_SAMPLE)) for ix in (b.integers(0, len(e), len(e)) for _ in range(500))]
        ce = clean["est"] - clean["truth"]; pc = clean["truth"] * (1 - clean["truth"])
        s = dict(mean_truth=float(tr.mean()), rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), max_abs=float(np.abs(e).max()),
                 ops=om, ratio=float(np.mean(e ** 2) / mse_s), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 per_noise_seed=per, clean_truth=clean["truth"], clean_err=float(ce), clean_ops=clean["ops"],
                 clean_ratio=float(ce ** 2 / (pc / (clean["ops"] / OPS_PER_SAMPLE))), max_rescale_change=float(inv),
                 cells=float(np.mean([q["cells"] for q in noisy])),
                 resolved_frac=float(np.sum([q["resolved"] for q in noisy]) / max(1, np.sum([q["cut"] for q in noisy]))))
        out["zoo"][zs] = s
        print(f"zoo seed {zs}: clean err {100*ce:+.3f}pp (ratio {s['clean_ratio']:.3f}) | noisy acc {tr.mean():.3f} rms {100*s['rms']:.3f}% "
              f"bias {100*s['bias']:+.3f}% max {100*s['max_abs']:.2f}% ops {om:.0f} MSE ratio {s['ratio']:.4f} [{s['ci'][0]:.4f},{s['ci'][1]:.4f}] "
              f"per noise seed {[round(x, 4) for x in per]} | cells {s['cells']:.0f}, resolved {s['resolved_frac']:.0%} | "
              f"max change under rescaling {inv:.1e}", flush=True)
        json.dump(out, open(os.path.join(HERE, "eval85b.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
