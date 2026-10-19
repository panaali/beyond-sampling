"""T410 summary: python summarize.py -> results/summary.json and a printed table (pre-registered statistics only)."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
T402 = dict(ratio=(.059, .004), dacc=(-1.16e-4, 2e-6), shift=(.093, np.sqrt(679e3) / 7.26e6))


def jk_ratio(num, den):
    """Delete-one-block jackknife of sum(num) / sum(den) (num, den: per-block sums)."""
    n = len(num); N, D = num.sum(), den.sum(); r = N / D
    rj = (N - num) / (D - den); return r, np.sqrt((n - 1) / n * ((rj - rj.mean()) ** 2).sum())


def load(kind):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, f"{kind}_*.npz"))):
        z = np.load(f); out[f] = {k: z[k] for k in z.files}
    return out


def main():
    S, A = load("slab_2.0"), load("acc")
    names = max((list(z["names"]) for z in S.values()), key=len)
    # pair (base, t402): every file; controls: the "all" files
    def cat(D, key, j, need_all):
        return np.concatenate([z[key][:, j] for z in D.values() if (len(z["names"]) > 2) or not need_all])
    res = {}
    for j, nm in enumerate(names):
        allonly = j >= 2
        eb, ej = cat(S, "sE", 0, allonly), cat(S, "sE", j, allonly)
        rb, rj = cat(S, "sR1", 0, allonly), cat(S, "sR1", j, allonly)
        cb, cj = cat(A, "sC", 0, allonly), cat(A, "sC", j, allonly)
        nS = len(eb) * 1e6; nA = len(cb) * 1e6
        r, rse = jk_ratio(ej, eb); sh, shse = jk_ratio(rj, rb)
        d = (cj - cb) / 1e6; dacc, dse = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        res[nm] = dict(n_slab=nS, n_acc=nA, P_base=eb.sum() / nS, P_edit=ej.sum() / nS, ratio=r, ratio_se=rse,
                       dacc=dacc, dacc_se=dse, acc_base=cb.sum() / nA, acc_edit=cj.sum() / nA,
                       shift=sh - 1, shift_se=shse, success=bool(r <= .1 and dacc >= -2e-4))
    t = res["t402"]
    sw = np.concatenate([z["sw"] for z in S.values()]); pgap = sw.sum() / (len(sw) * 1e6)
    agree = {k: dict(mine=(t[k], t[k + "_se"]), t402=T402[k],
                     z=(t[k] - T402[k][0]) / np.hypot(t[k + "_se"], T402[k][1])) for k in ("ratio", "dacc", "shift")}
    for v in agree.values():
        v["agree"] = bool(abs(v["z"]) <= 2)
    cnt = {nm: np.sum([z["cnt"][j] for z in S.values() if len(z["names"]) > j], 0).tolist() for j, nm in enumerate(names[:2])}

    def group(prefix, k=None):
        g = [res[n] for n in names if n.startswith(prefix)][:k]; r = np.array([x["ratio"] for x in g])
        return dict(n=len(g), n_success=int(sum(x["success"] for x in g)), n_ratio_le_0p1=int((r <= .1).sum()),
                    ratio_min_q25_med_q75_max=np.quantile(r, [0, .25, .5, .75, 1]).tolist(),
                    dacc_pp_min_med_max=(np.quantile([x["dacc"] for x in g], [0, .5, 1]) * 100).tolist(),
                    shift_min_med_max=np.quantile([x["shift"] for x in g], [0, .5, 1]).tolist())
    rand40, rand20, sign = group("rand_"), group("rand_", 20), group("sign_")
    rmin = rand40["ratio_min_q25_med_q75_max"][0]
    verdict = ("targeting matters" if rand40["n_success"] <= 2 and t["ratio"] < rmin else
               "targeting not needed" if rand40["n_success"] >= 10 else "partial")
    fwd = sum(len(z["sE"]) * 1e6 * len(z["names"]) for z in S.values()) + sum(len(z["sC"]) * 1e6 * len(z["names"]) for z in A.values())
    out = dict(pgap_2=pgap, t402=t, base=res["base"], agree=agree, picked_rank_counts_unweighted=cnt, rand40=rand40,
               rand20=rand20, sign=sign, verdict=verdict, forward_passes=fwd, per_model=res)
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)
    print(f"pgap(2.0) {pgap:.4e}; base P(E1(2)) {t['P_base']:.3e} edit {t['P_edit']:.3e}; acc base {res['base']['acc_base']:.6f}")
    for k, v in agree.items():
        print(f"{k}: mine {v['mine'][0]:.5g} +- {v['mine'][1]:.2g}  T402 {v['t402'][0]:.4g} +- {v['t402'][1]:.2g}  z {v['z']:+.2f} agree {v['agree']}")
    for nm in names:
        x = res[nm]
        print(f"{nm:8s} ratio {x['ratio']:.4f} +- {x['ratio_se']:.4f}  dacc {100 * x['dacc']:+.4f} +- {100 * x['dacc_se']:.4f} pp"
              f"  shift {100 * x['shift']:+.2f}%  {'SUCCESS' if x['success'] else ''}")
    print("rand40", rand40); print("rand20", rand20); print("sign", sign); print("verdict:", verdict, "fwd", f"{fwd:.3g}")


if __name__ == "__main__":
    main()
