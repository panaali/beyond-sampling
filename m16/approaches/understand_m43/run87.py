"""T87 rung (H=4, T=4): frozen T85 recipe (estT, eps=0.1) on zoo_2nd_argmax(4,4) seeds 0-4 (clean) and noisy copies
(abs sigma 0.05, fresh noise seed 50). Truth: estT exact mode (enumeration) checked against 1e7-sample MC.
usage: OMP_NUM_THREADS=1 python run87.py [n_noisy=8]   -> run87.json, run87.log"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo, mc_acc, ops_per_sample
from zoo43 import noisy_copies

H, T = 4, 4
NN = int(sys.argv[1]) if len(sys.argv) > 1 else 8
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
    res = {"clean": {}, "noisy": {}}
    fn = os.path.join(HERE, "run87.json")
    if os.path.exists(fn):
        res = json.load(open(fn))
    for s in range(5):
        if str(s) in res["noisy"]:
            continue
        m = zoo(H, T, s); r = run(m, True)
        r["mc1e7"] = mc_acc(m, 10_000_000, np.random.default_rng(100 + s))
        p = r["exact"]; e = r["est"] - p
        r["ratio"] = e ** 2 / (p * (1 - p) / (r["ops"] / OPS_S))
        res["clean"][s] = r
        print(f"seed {s}: exact {p:.5f} (MC1e7 {r['mc1e7']:.5f}, z {(r['mc1e7']-p)/np.sqrt(p*(1-p)/1e7):+.1f}; {r['exact_cells']} cells, "
              f"{r['exact_ops']/1e6:.1f}M ops) | recipe err {100*e:+.3f}pp, {r['cells']} cells, {r['ops']/1e6:.1f}M ops, "
              f"resolved {r['resolved_frac']:.0%}; sampling RMSE at equal ops {100*np.sqrt(p*(1-p)/(r['ops']/OPS_S)):.3f}% -> ratio {r['ratio']:.2f}",
              flush=True)
        rows = []
        for m2 in noisy_copies(m, NN, 0.05, 50):
            r2 = run(m2, False); r2["truth"] = mc_acc(m2, 10_000_000, np.random.default_rng(7)); rows.append(r2)
        tr = np.array([q["truth"] for q in rows]); e = np.array([q["est"] for q in rows]) - tr
        om = np.mean([q["ops"] for q in rows]); pq = np.mean(tr * (1 - tr))
        # MC truth noise (1e7) adds ~p(1-p)/1e7 to the MSE; subtracted for the ratio
        mse = max(np.mean(e ** 2) - pq / 1e7, 0.0)
        res["noisy"][s] = dict(rows=rows, rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), ops=float(om),
                               ratio=float(mse / (pq / (om / OPS_S))), mean_truth=float(tr.mean()))
        print(f"   noisy x{NN}: acc {tr.mean():.3f} rms {100*np.sqrt(np.mean(e**2)):.3f}% bias {100*e.mean():+.3f}% ops {om/1e6:.1f}M "
              f"ratio {res['noisy'][s]['ratio']:.2f}", flush=True)
        json.dump(res, open(os.path.join(HERE, "run87.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
