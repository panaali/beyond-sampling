"""T101: score one training-amount rung k (n_seqs = 2**k) of M_4,3: zoo seeds 0-4 + frozen noisy panels.
Per zoo seed: clean model + 1024 ARC copies (noise seeds 20-23 x 256, abs sigma 0.05). Truth: exact.py.
T75 (frozen est75.RECIPE) on all 1024 copies; fibre F2L3 (R4 fix, strict count) and region counts on the first
n_fib copies (noise seed 20). No estimator sees the truth; nothing is fitted.
usage: OMP_NUM_THREADS=1 python run_rung.py k [n_fib=256]  -> rung_k{k}.json"""
import json
import os
import sys
import time

import numpy as np

import lt

HERE = os.path.dirname(os.path.abspath(__file__))


def ratio_stats(e, tr, ops, nboot=1000):
    o = float(np.mean(ops)); pq = tr * (1 - tr)
    agg = float(np.mean(e ** 2) / (np.mean(pq) / (o / lt.OPS_PER_SAMPLE)))
    rng = np.random.default_rng(0)       # bootstrap over copies (analysis only)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(e), len(e))
        bs.append(np.mean(e[i] ** 2) / (np.mean(pq[i]) / (o / lt.OPS_PER_SAMPLE)))
    per = e ** 2 / (np.maximum(pq, 1e-12) / (np.asarray(ops, float) / lt.OPS_PER_SAMPLE))
    return dict(ratio=agg, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], rms=float(np.sqrt(np.mean(e ** 2))),
                bias=float(e.mean()), max_abs=float(np.abs(e).max()), ops=o, frac_copies_ratio_ge1=float(np.mean(per >= 1)),
                worst_copy_ratio=float(per.max()))


def score_seed(k, s, n_fib):
    base = lt.zoo(k, s)
    pan = lt.panel(base)
    out = {"k": k, "seed": s}
    tb, eops = lt.truth(base); r, o, st = lt.t75(base); f, fo, ftr = lt.f2l(base)
    c = dict(truth=float(tb), exact_ops=eops, regions=lt.regions(base), t75=float(r), t75_ops=o, t75_stats=st,
             f2l3=float(f), f2l3_ops=fo, f2l3_traces=ftr)
    for nm, est, ops in (("t75", r, o), ("f2l3", f, fo)):
        pq = tb * (1 - tb); c[nm + "_err"] = float(est - tb)
        c[nm + "_ratio"] = float((est - tb) ** 2 / (pq / (ops / lt.OPS_PER_SAMPLE)))
    out["clean"] = c
    tr = np.empty(len(pan)); t7 = np.empty(len(pan)); t7o = np.empty(len(pan)); eo = np.empty(len(pan)); cells = np.empty(len(pan))
    for i, m in enumerate(pan):
        tr[i], eo[i] = lt.truth(m)
        t7[i], t7o[i], st = lt.t75(m); cells[i] = st["cells"]
    nf = min(n_fib, len(pan))
    fe = np.empty(nf); fo_ = np.empty(nf); rg = np.empty(nf)
    for i in range(nf):
        fe[i], fo_[i], _ = lt.f2l(pan[i]); rg[i] = lt.regions(pan[i])
    out["panel"] = dict(n=len(pan), n_fib=nf, mean_truth=float(tr.mean()), exact_ops=float(eo.mean()),
                        regions_mean=float(rg.mean()), t75_cells=float(cells.mean()),
                        t75=ratio_stats(t7 - tr, tr, t7o), f2l3=ratio_stats(fe - tr[:nf], tr[:nf], fo_),
                        t75_on_fib_subset=ratio_stats(t7[:nf] - tr[:nf], tr[:nf], t7o[:nf]))
    out["panel"]["t75_errs"] = (t7 - tr).tolist(); out["panel"]["truths"] = tr.tolist(); out["panel"]["f2l3_errs"] = (fe - tr[:nf]).tolist()
    return out


def main():
    k = int(sys.argv[1]); n_fib = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    t0 = time.time(); res = {"k": k, "n_fib": n_fib, "seeds": {}}
    for s in range(5):
        r = score_seed(k, s, n_fib); res["seeds"][s] = r
        c, p = r["clean"], r["panel"]
        print(f"k={k} seed {s}: clean acc {c['truth']:.4f} regions {c['regions']} | T75 clean err {100*c['t75_err']:+.3f}pp "
              f"ops {c['t75_ops']} | F2L3 clean err {100*c['f2l3_err']:+.1e}pp ops {c['f2l3_ops']} || panel acc {p['mean_truth']:.3f} "
              f"regions {p['regions_mean']:.1f} | T75 ratio {p['t75']['ratio']:.4f} {[round(x, 4) for x in p['t75']['ci']]} "
              f"rms {100*p['t75']['rms']:.2f}% max {100*p['t75']['max_abs']:.1f}% ops {p['t75']['ops']:.0f} | F2L3 ratio "
              f"{p['f2l3']['ratio']:.2e} ops {p['f2l3']['ops']:.0f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(os.path.join(HERE, f"rung_k{k}.json"), "w"), default=float)
    print(f"done k={k} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
