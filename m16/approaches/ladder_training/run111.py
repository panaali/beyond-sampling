"""T111: T87's frozen recipe (understand_m43/estT.py, imported UNCHANGED, eps=0.1) on better-trained (4,4) models.
Same loop as understand_m43/run87.py; only the weights differ (ours, weights/m_4_4_<tag>_<seed>.npz).
Truth: estT exact mode checked against 1e7-sample MC; 8 noisy copies (abs sigma 0.05, noise seed 50) with 1e7 MC truth.
usage: OMP_NUM_THREADS=1 python run111.py TAG [n_noisy=8] [seeds=0,1,2,3,4]   (TAG = 30sL2 or 31s; 'zoo' re-runs the zoo models)
       -> run111_<TAG>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
M43 = os.path.join(HERE, "..", "understand_m43")
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
sys.path.insert(0, M43)
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo, mc_acc, ops_per_sample
from zoo43 import noisy_copies
from models import load

H, T = 4, 4
TAG = sys.argv[1]
NN = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEEDS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else range(5)
OPS_S = ops_per_sample(H, T)


def run(m, with_exact):
    out = {}
    if with_exact:
        estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0); o = Ops(); t0 = time.time()
        out["exact"] = estT.estT(m, o, mode="exact"); out["exact_ops"] = o.n; out["exact_cells"] = estT.STATS["cells"]
        out["exact_sec"] = time.time() - t0
    estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0); o = Ops(); t0 = time.time()
    out["est"] = estT.estT(m, o); out["ops"] = o.n; out["cells"] = estT.STATS["cells"]
    out["resolved_frac"] = estT.STATS["resolved"] / max(1, estT.STATS["cut"]); out["sec"] = time.time() - t0
    return out


def main():
    fn = os.path.join(HERE, f"run111_{TAG}.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {"clean": {}, "noisy": {}}
    for s in SEEDS:
        if str(s) in res["noisy"]:
            continue
        m = zoo(H, T, s) if TAG == "zoo" else load(H, T, s, n_tag=TAG)
        r = run(m, True)
        r["mc1e7"] = mc_acc(m, 10_000_000, np.random.default_rng(100 + s))  # T111b seeds > 4 use rng 100 + seed as well
        p = r["exact"]; e = r["est"] - p
        r["ratio"] = e ** 2 / (p * (1 - p) / (r["ops"] / OPS_S))
        res["clean"][str(s)] = r
        print(f"seed {s}: exact {p:.5f} (MC1e7 {r['mc1e7']:.5f}, z {(r['mc1e7']-p)/np.sqrt(p*(1-p)/1e7):+.1f}; {r['exact_cells']} cells, "
              f"{r['exact_ops']/1e6:.1f}M ops) | recipe err {100*e:+.3f}pp, {r['cells']} cells, {r['ops']/1e6:.1f}M ops, "
              f"resolved {r['resolved_frac']:.0%}; sampling RMSE at equal ops {100*np.sqrt(p*(1-p)/(r['ops']/OPS_S)):.3f}% -> ratio {r['ratio']:.2f}",
              flush=True)
        rows = []
        for m2 in noisy_copies(m, NN, 0.05, 50):
            r2 = run(m2, False); r2["truth"] = mc_acc(m2, 10_000_000, np.random.default_rng(7)); rows.append(r2)
        tr = np.array([q["truth"] for q in rows]); e = np.array([q["est"] for q in rows]) - tr
        om = np.mean([q["ops"] for q in rows]); pq = np.mean(tr * (1 - tr))
        # MC truth noise (1e7) adds ~p(1-p)/1e7 to the MSE; subtracted for the ratio (as run87)
        mse = max(np.mean(e ** 2) - pq / 1e7, 0.0)
        res["noisy"][str(s)] = dict(rows=rows, rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), ops=float(om),
                                    ratio=float(mse / (pq / (om / OPS_S))), mean_truth=float(tr.mean()))
        print(f"   noisy x{NN}: acc {tr.mean():.3f} rms {100*np.sqrt(np.mean(e**2)):.3f}% bias {100*e.mean():+.3f}% ops {om/1e6:.1f}M "
              f"ratio {res['noisy'][str(s)]['ratio']:.2f}", flush=True)
        json.dump(res, open(fn, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
