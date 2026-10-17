"""T97: score the author's frozen recipe (fibre.est_fibre2 with LOOK = 1, 2 (= F2L2), 3) against the reviewer's own
exact truth (rtruth.exact) on zoo seeds and FRESH noisy copies (noise seeds never used by the author).
Models per H: zoo seeds 0-4; ARC abs sigma 0.05, noise seeds 101, 102 x 64 copies of the best-of-5 seed;
abs sigma 0.10, seed 103 x 32 copies (stress, beyond ARC's sigma); relative sigma 0.0025/0.005/0.01/0.02 x RMS(W),
6 copies each, rng seed 7 (author/leaderboard used seed 0).
usage: OMP_NUM_THREADS=1 python score.py H [looks]  -> score_H{H}.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAD = os.path.join(HERE, "..", "..", "m16", "approaches", "understand_ladder")
sys.path.insert(0, LAD)
import fibre  # noqa: E402  (author's code, read-only)
import rtruth as R  # noqa: E402

from alg_zoo.zoo import best_of_5_seed_2nd_argmax  # noqa: E402


def models(H, T=3):
    ms = {f"seed{s}": R.load(H, T, s)[0] for s in range(5)}
    base = R.load(H, T, best_of_5_seed_2nd_argmax(hidden_size=H, seq_len=T))[0]
    for sig, ns, k in ((0.05, 101, 64), (0.05, 102, 64), (0.10, 103, 32)):
        rng = np.random.default_rng(ns)
        for i in range(k):
            ms[f"abs{sig}_n{ns}_{i}"] = tuple(w + sig * rng.standard_normal(w.shape) for w in base)
    for sig in (0.0025, 0.005, 0.01, 0.02):
        rng = np.random.default_rng(7)
        for i in range(6):
            ms[f"rel{sig}_{i}"] = tuple(w + sig * np.sqrt(np.mean(w ** 2)) * rng.standard_normal(w.shape) for w in base)
    return ms


def ops1(H, T=3):
    return T * H * (H + 1) + T * H


def run(H, looks):
    rows = {}
    t0 = time.time()
    for name, m in models(H).items():
        ex, nr, _ = R.exact(m)
        r = {"truth": ex, "regions": nr}
        for L in looks:
            fibre.LOOK = L
            o = fibre.Ops(); inf = {}
            e = fibre.est_fibre2(m, o, inf)
            r[f"L{L}"] = {"est": e, "ops": o.n, "traces": inf["traces"]}
        rows[name] = r
    print(f"H={H} {len(rows)} models {time.time() - t0:.0f}s", flush=True)
    return rows


def group(n):
    if n.startswith("seed"):
        return "seed"
    if n.startswith("abs"):
        return n.split("_")[0] + ("" if "0.1" in n else "")
    return n.rsplit("_", 1)[0]


def summarize(H, rows, key):
    o1 = ops1(H)
    out = {}
    for g in sorted(set(group(n) for n in rows)):
        ns = [n for n in rows if group(n) == g]
        tr = np.array([rows[n]["truth"] for n in ns]); e = np.array([rows[n][key]["est"] for n in ns]) - tr
        ops = np.array([rows[n][key]["ops"] for n in ns], float)
        pq = tr * (1 - tr)
        grp = float(np.mean(e ** 2) / (np.mean(pq) / (np.mean(ops) / o1)))
        per = (e ** 2) / (np.maximum(pq, 1e-12) / (ops / o1))
        out[g] = dict(n=len(ns), mean_truth=float(tr.mean()), rmse=float(np.sqrt(np.mean(e ** 2))),
                      max_abs=float(np.abs(e).max()), max_over=float(e.max()), ops=float(ops.mean()),
                      group_ratio=grp, worst_model_ratio=float(per.max()), worst=ns[int(per.argmax())],
                      n_fail=int((per >= 1).sum()))
    return out


if __name__ == "__main__":
    H = int(sys.argv[1]); looks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 3]
    rows = run(H, looks)
    summ = {f"L{L}": summarize(H, rows, f"L{L}") for L in looks}
    json.dump({"H": H, "rows": rows, "summary": summ}, open(os.path.join(HERE, f"score_H{H}.json"), "w"), indent=1)
    for k, S in summ.items():
        for g, d in S.items():
            print(f"H={H} {k} {g:12s} n={d['n']:3d} acc {d['mean_truth']:.5f} rmse {d['rmse']:.2e} maxabs {d['max_abs']:.2e} "
                  f"ops {d['ops']:.0f} group {d['group_ratio']:.2e} worst {d['worst_model_ratio']:.2e} ({d['worst']}) fails {d['n_fail']}")
