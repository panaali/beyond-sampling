"""T195 analysis: identity check vs ref/, and split / class-changing mass by trigger rank, step, neuron.
Driver of a split = argmax_i con_i (contrast attribution, attr.py); rank = position of that input in the cell's
ordering chamber (1 = largest). 'frac' variants spread each split over inputs by max(con_i, 0) / sum.
Class-changing (exact, within the recipe): absdE > THR. usage: python analyze.py [names...] -> analysis.json"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
THR = 1e-13


def load(name):
    z = np.load(os.path.join(HERE, "out", name + ".npz"))
    return json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}


def ranks(r):
    perm = r["perm"]; T = perm.shape[1]
    rk = np.empty_like(perm)
    rk[np.arange(len(perm))[:, None], perm] = np.arange(1, T + 1)[None]      # rk[n, i] = rank of input i
    return rk


def summarise(meta, r):
    T = r["perm"].shape[1]; n = len(r["t"]); rk = ranks(r)
    drv = np.argmax(r["con"], 1); drank = rk[np.arange(n), drv]
    cp = np.maximum(r["con"], 0); cp = cp / np.maximum(cp.sum(1, keepdims=True), 1e-300)
    gradrank = rk[np.arange(n), np.argmax(r["mag"], 1)]
    cls = r["absdE"] > THR
    out = dict(n_splits=n, acc=meta["acc"], ops=meta["ops"], cells=meta["cells"], sec=meta["sec"],
               abs_mass=float(r["absdE"].sum()), net_mass=float(r["dE"].sum()), n_cls=int(cls.sum()),
               s_cls=int((r["s_cls"] == 1).sum()), s_err=int((r["s_err"] == 1).sum()), m_min=float(r["m_min"].sum()))
    by = {}
    for key, lab in (("rank", drank), ("gradrank", gradrank), ("step", r["t"].astype(int)), ("neuron", r["k"].astype(int)),
                     ("newest", (drv == r["t"]).astype(int))):
        d = {}
        for v in np.unique(lab):
            m = lab == v
            d[int(v)] = dict(n=int(m.sum()), n_cls=int(cls[m].sum()), absdE=float(r["absdE"][m].sum()),
                             dE=float(r["dE"][m].sum()), m_min=float(r["m_min"][m].sum()), s_err=int((r["s_err"][m] == 1).sum()))
        by[key] = d
    # fractional attribution over ranks
    fr = {}
    for q in range(1, T + 1):
        wq = (cp * (rk == q)).sum(1)
        fr[q] = dict(n=float(wq.sum()), absdE=float((wq * r["absdE"]).sum()), m_min=float((wq * r["m_min"]).sum()))
    by["rank_frac"] = fr
    # step x rank
    sr = {}
    for t in (int(v) for v in np.unique(r["t"])):
        m = r["t"] == t
        sr[t] = {int(q): dict(n=int((m & (drank == q)).sum()), absdE=float(r["absdE"][m & (drank == q)].sum()))
                 for q in range(1, T + 1)}
    by["step_rank"] = sr
    # uniform baseline: the step-t gate depends on inputs 0..t; share of rank>=4 among them, averaged over splits
    base = np.array([(rk[j, :int(r["t"][j]) + 1] >= 4).mean() for j in range(n)]) if n else np.zeros(0)
    out["low_rank"] = dict(
        share_splits=float((drank >= 4).mean()) if n else 0.0,
        share_cls_splits=float((drank[cls] >= 4).mean()) if cls.any() else 0.0,
        share_absdE=float(r["absdE"][drank >= 4].sum() / max(r["absdE"].sum(), 1e-300)),
        share_absdE_frac=float(sum(fr[q]["absdE"] for q in fr if q >= 4) / max(r["absdE"].sum(), 1e-300)),
        share_absdE_gradrank=float(r["absdE"][gradrank >= 4].sum() / max(r["absdE"].sum(), 1e-300)),
        uniform_baseline=float(base.mean()) if n else 0.0,
        uniform_baseline_absdE=float((base * r["absdE"]).sum() / max(r["absdE"].sum(), 1e-300)))
    # first-order pruning view: splits whose own exact effect is below delta (effects of joint pruning not additive)
    out["prunable"] = {f"{d:g}": dict(share=float((r["absdE"] < d).mean()) if n else 0.0,
                                      share_rank4=float((drank[r["absdE"] < d] >= 4).mean()) if (r["absdE"] < d).any() else 0.0,
                                      sum_absdE=float(r["absdE"][r["absdE"] < d].sum()))
                       for d in (1e-10, 1e-8, 1e-6, 1e-4)}
    out["by"] = by
    return out


def main():
    names = sys.argv[1:] or sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(HERE, "out", "*.npz")))
    res = {}
    for nm in names:
        meta, r = load(nm); s = summarise(meta, r)
        ref = json.load(open(os.path.join(HERE, "ref", nm + ".json"))) if os.path.exists(os.path.join(HERE, "ref", nm + ".json")) else None
        s["identical"] = None if ref is None else bool(ref["acc"] == meta["acc"] and ref["ops"] == meta["ops"] and
                                                      ref["cells"] == meta["cells"] and ref["resolved"] == meta["resolved"])
        res[nm] = s
        lr = s["low_rank"]
        print(f"{nm} ident {s['identical']} splits {s['n_splits']} cls {s['n_cls']} absdE {s['abs_mass']:.3g} | "
              f"rank>=4: splits {lr['share_splits']:.3f} (unif {lr['uniform_baseline']:.3f}) absdE {lr['share_absdE']:.3f} "
              f"frac {lr['share_absdE_frac']:.3f} grad {lr['share_absdE_gradrank']:.3f}")
        print("   by rank:", {q: (v["n"], v["n_cls"], f"{v['absdE']:.2g}") for q, v in s["by"]["rank"].items()})
    json.dump(res, open(os.path.join(HERE, "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
