"""T112: one ladder rung (H, T=3): frozen RECIPE112 on zoo_2nd_argmax(H,3) seeds 0-4 (clean) + noisy copies (abs
sigma 0.05, FRESH noise seeds 70-73). Truth: arc_m43_repro/exact.py (generic in H). Strict ops; sampling costs
3H(H+1)+3H ops. Every 8th model is re-run after a random rescaling + permutation (invariance test). Also reports
cells (recipe) vs exact leaves vs linear regions.
usage: OMP_NUM_THREADS=1 python evalH.py H [n_per_noise_seed=64]  -> evalH_<H>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro")); sys.path.insert(0, os.path.join(HERE, "..", "understand_ladder"))
import numpy as np
import exact as EX
import est75 as E75
from m43 import Ops
from zoo44 import zoo, ops_per_sample
from zoo43 import noisy_copies
from rescale_test import rescale

H = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
SEEDS = (70, 71, 72, 73); OPS_S = ops_per_sample(H, 3); REC = E75.RECIPE112


def main():
    fn = os.path.join(HERE, f"evalH_{H}.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {"H": H, "noise_seeds": SEEDS, "n_per_seed": N, "recipe": repr(REC), "zoo": {}}
    from exact3 import count_regions
    for zs in range(5):
        if str(zs) in out["zoo"]:
            continue
        t0 = time.time(); rng = np.random.default_rng(1000 + zs)
        base = zoo(H, 3, zs)
        models = [base] + [m for ns in SEEDS for m in noisy_copies(base, N, 0.05, ns)]
        rows = []; inv = 0.0
        for i, m in enumerate(models):
            oe = Ops(); st = {}; tr = EX.exact_acc(m, oe, st)
            E75.STATS.update(cut=0, resolved=0, cells=0); o = Ops(); r = E75.est75(m, o, **REC)
            if i % 8 == 0:
                inv = max(inv, abs(E75.est75(rescale(m, rng), Ops(), **REC) - r))
            rows.append(dict(truth=tr, est=r, ops=o.n, exact_ops=oe.n, leaves=st["leaves"], cells=E75.STATS["cells"],
                             cut=E75.STATS["cut"], resolved=E75.STATS["resolved"]))
        clean, noisy = rows[0], rows[1:]
        tr = np.array([q["truth"] for q in noisy]); e = np.array([q["est"] for q in noisy]) - tr
        om = float(np.mean([q["ops"] for q in noisy])); pq = float(np.mean(tr * (1 - tr)))
        ratio = float(np.mean(e ** 2) / (pq / (om / OPS_S)))
        b = np.random.default_rng(0)
        bs = [np.mean(e[ix] ** 2) / (np.mean(tr[ix] * (1 - tr[ix])) / (om / OPS_S)) for ix in (b.integers(0, len(e), len(e)) for _ in range(500))]
        ce = clean["est"] - clean["truth"]; pc = clean["truth"] * (1 - clean["truth"])
        s = dict(clean_truth=clean["truth"], clean_err=float(ce), clean_ops=clean["ops"],
                 clean_ratio=float(ce ** 2 / max(pc / (clean["ops"] / OPS_S), 1e-300)), regions=count_regions(base),
                 clean_cells=clean["cells"], clean_leaves=clean["leaves"], clean_exact_ops=clean["exact_ops"],
                 mean_truth=float(tr.mean()), rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), max_abs=float(np.abs(e).max()),
                 ops=om, exact_ops=float(np.mean([q["exact_ops"] for q in noisy])), ratio=ratio,
                 ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 cells=float(np.mean([q["cells"] for q in noisy])), leaves=float(np.mean([q["leaves"] for q in noisy])),
                 resolved_frac=float(np.sum([q["resolved"] for q in noisy]) / max(1, np.sum([q["cut"] for q in noisy]))),
                 max_rescale_change=float(inv), seconds=time.time() - t0)
        out["zoo"][str(zs)] = s
        print(f"H={H} seed {zs}: clean acc {clean['truth']:.4f} err {100*ce:+.4f}pp (ratio {s['clean_ratio']:.3f}); regions {s['regions']}, "
              f"exact leaves {clean['leaves']}, recipe cells {clean['cells']} | noisy acc {tr.mean():.4f} rms {100*s['rms']:.4f}% "
              f"bias {100*s['bias']:+.4f}% max {100*s['max_abs']:.3f}% ops {om:.0f} (exact {s['exact_ops']:.0f}) MSE ratio {ratio:.4f} "
              f"[{s['ci'][0]:.4f},{s['ci'][1]:.4f}] resolved {s['resolved_frac']:.0%} rescale {inv:.1e} ({s['seconds']:.0f}s)", flush=True)
        json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()
