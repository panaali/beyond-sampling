"""T113 items 1, 2b, 4, 5: frozen RECIPE112 (the reviewed est75.est75(**RECIPE112), its own strict op charge) on
never-used noise seeds. For every model: truth from arc_m43_repro/exact.py AND from this review's own enumeration
(mine.exact_mine; built differently); the independent re-implementation rec112.py (must agree); its cell / cut /
resolved counts (own counters); linear regions (own count) and pieces of the regions in the 24 half-chambers.
Every 4th model is re-run after a random function-preserving rescaling (c = exp(U(-4, 4)), wider than the author's
U(-2, 2)) and permutation.
usage: OMP_NUM_THREADS=1 python fresh.py H MODE SIGMA N SEED [SEED ...] [--light]
  MODE abs (ARC: absolute sigma) | rel (m16/leaderboard.py: sigma x RMS of each matrix)
  --light: skip own exact truth, half-chamber pieces and rescaling (for the extra noise levels)
-> out/fresh_H<H>_<mode><sigma>_<seeds>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_m43"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import exact as EX
import est75 as E75
from m43 import Ops
import mine as M
import rec112 as R


def ratio(e, tr, ops, ops_s):
    return float(np.mean(e ** 2) / (np.mean(tr * (1 - tr)) / (ops / ops_s)))


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]; light = "--light" in sys.argv
    H, mode, sigma, n = int(a[0]), a[1], float(a[2]), int(a[3]); seeds = [int(s) for s in a[4:]]
    ops_s = 3 * H * (H + 1) + 3 * H
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    fn = os.path.join(HERE, "out", f"fresh_H{H}_{mode}{sigma}_{'-'.join(map(str, seeds))}.json")
    out = json.load(open(fn)) if os.path.exists(fn) else dict(H=H, mode=mode, sigma=sigma, n=n, seeds=seeds, zoo={})
    noisy = M.noisy_abs if mode == "abs" else M.noisy_rel
    for zs in range(5):
        if str(zs) in out["zoo"]:
            continue
        t0 = time.time(); base = M.zoo_w(H, zs); rng = np.random.default_rng(777 + 10 * H + zs)
        models = [base] + [m for ns in seeds for m in noisy(base, n, sigma, ns)]
        rows = []
        for i, m in enumerate(models):
            oe = Ops(); tr = EX.exact_acc(m, oe)
            o = Ops(); est = E75.est75(m, o, **E75.RECIPE112)
            st = {}; rr = R.rec112(R.as_lists(m), R.P, st)
            row = dict(truth=tr, est=est, ops=o.n, exact_ops=oe.n, reimpl_diff=rr - est, **st)
            if not light:
                ms = {}; row["truth_mine"] = M.exact_mine(m, ms); row["regions"] = ms["regions"]
                row["half_pieces"] = M.half_pieces(m)
                if i % 4 == 0:
                    row["rescale_change"] = E75.est75(M.rescale_perm(m, rng), Ops(), **E75.RECIPE112) - est
            rows.append(row)
        clean, nz = rows[0], rows[1:]
        tr = np.array([q["truth"] for q in nz]); e = np.array([q["est"] for q in nz]) - tr
        om = float(np.mean([q["ops"] for q in nz]))
        b = np.random.default_rng(0)
        bs = [ratio(e[ix], tr[ix], om, ops_s) for ix in (b.integers(0, len(e), len(e)) for _ in range(1000))]
        per = [ratio(e[k * n:(k + 1) * n], tr[k * n:(k + 1) * n], om, ops_s) for k in range(len(seeds))]
        s = dict(clean=clean, ratio=ratio(e, tr, om, ops_s), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 per_seed=per, mean_truth=float(tr.mean()), rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()),
                 max_abs=float(np.abs(e).max()), ops=om, exact_ops=float(np.mean([q["exact_ops"] for q in nz])),
                 max_reimpl_diff=float(max(abs(q["reimpl_diff"]) for q in rows)),
                 cells=float(np.mean([q["cells"] for q in nz])),
                 resolved_frac=float(sum(q["resolved"] for q in nz) / max(1, sum(q["cut"] for q in nz))),
                 rows=rows, seconds=time.time() - t0)
        if not light:
            s["max_truth_diff"] = float(max(abs(q["truth_mine"] - q["truth"]) for q in rows))
            s["regions"] = float(np.mean([q["regions"] for q in nz])); s["half_pieces"] = float(np.mean([q["half_pieces"] for q in nz]))
            s["max_rescale_change"] = float(max(abs(q["rescale_change"]) for q in rows if "rescale_change" in q))
        out["zoo"][str(zs)] = s
        c = clean
        print(f"H={H} {mode} {sigma} zoo {zs}: clean acc {c['truth']:.5f} err {100*(c['est']-c['truth']):+.4f}pp cells {c['cells']}"
              + ("" if light else f" regions {c['regions']} half-pieces {c['half_pieces']}")
              + f" | noisy acc {s['mean_truth']:.4f} rms {100*s['rms']:.3f}% max {100*s['max_abs']:.2f}% ops {om:.0f} "
              f"(exact {s['exact_ops']:.0f}) ratio {s['ratio']:.4f} [{s['ci'][0]:.4f},{s['ci'][1]:.4f}] per seed "
              f"{[round(x, 4) for x in per]} reimpl {s['max_reimpl_diff']:.0e}"
              + ("" if light else f" truthdiff {s['max_truth_diff']:.0e} rescale {s['max_rescale_change']:.0e}")
              + f" ({s['seconds']:.0f}s)", flush=True)
        json.dump(out, open(fn, "w"))


if __name__ == "__main__":
    main()
