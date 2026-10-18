"""T218 analysis of results/*.json (from tail.py). usage: python analyse.py > analyse.out; writes summary.json."""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["z6s0", "z6s1", "z6s2", "z6s3", "z6s4", "orig", "s005", "s010"]
LS = (10, 30, 100)


def derived(p, T):
    if not p.get("kind"):
        return p
    a, b = p["a"], p["b"]
    if p["kind"] == "23":
        mx = p["p1"]
    else:
        mx = None                                       # max is one of the tied pair
    p["dist"] = abs(a - b)
    p["age"] = T - 1 - (mx if mx is not None else max(a, b))
    p["x9"] = ("max" if mx == T - 1 else "pair" if T - 1 in (a, b) else "third" if (p["kind"] == "12" and
               p["p1"] == T - 1) else "below")
    p["ft"] = tuple(tuple(f) for f in p["flips"])
    return p


SCHEMES = {
    "mech": lambda p: (p["kind"], p["a"], p["b"], p["p1"], p["ub"], p["wt"], p["ft"]),     # primary (pre-registered)
    "pos": lambda p: (p["kind"], p["a"], p["b"], p["p1"]),
    "pos_flips": lambda p: (p["kind"], p["a"], p["b"], p["p1"], p["ft"]),
    "nopos": lambda p: (p["kind"], p["ub"], p["wt"], p["ft"]),
    "relpos": lambda p: (p["kind"], p["dist"], p["age"], p["x9"], p["ub"], p["wt"], p["ft"]),
    "coarse": lambda p: (p["kind"], p["ub"], p["wt"], min(p["nflips"], 3)),
}


def mass_by(tail, f):
    d = defaultdict(float)
    for p in tail:
        d[f(p)] += p["g2"]
    return d


def top_labels(tail, f, L):
    d = mass_by(tail, f)
    return set(sorted(d, key=d.get, reverse=True)[:L])


def coverage(tail, f, labs):
    tot = sum(p["g2"] for p in tail)
    return sum(p["g2"] for p in tail if f(p) in labs) / tot


def within_tail(tail, f, k=10, minpts=20):
    """For the top-k labels with >= minpts tail points: share of the label's tail mass in its top 5% of points."""
    groups = defaultdict(list)
    for p in tail:
        groups[f(p)].append(p["g2"])
    big = sorted((g for g in groups.values() if len(g) >= minpts), key=sum, reverse=True)[:k]
    def share(v):
        v = np.sort(v)[::-1]; return v[:max(1, int(np.ceil(0.05 * len(v))))].sum() / v.sum()
    allv = [p["g2"] for p in tail]
    return dict(pooled=float(share(allv)), labels=[float(share(g)) for g in big], npts=[len(g) for g in big])


def main():
    R = {}
    for m in MODELS:
        fn = os.path.join(HERE, "results", f"{m}.json")
        if not os.path.exists(fn):
            continue
        d = json.load(open(fn)); T = d["T"]
        A = [derived(p, T) for p in d["rules"]["700"]["tail"] if p.get("kind")]
        B = [derived(p, T) for p in d["rules"]["701"]["tail"] if p.get("kind")]
        r = dict(T=T, n=d["n"], top5=[d["rules"][k]["top5pct_share"] for k in ("700", "701")],
                 top1=[d["rules"][k]["top1pct_share"] for k in ("700", "701")],
                 err=[2 * d["rules"][k]["err"] for k in ("700", "701")], ntail=len(A), schemes={})
        rng = np.random.default_rng(0)
        for name, f in SCHEMES.items():
            nl = len(mass_by(A + B, f))
            s = dict(nlabels_A=len(mass_by(A, f)), nlabels_AB=nl)
            for L in LS:
                s[f"in_A@{L}"] = coverage(A, f, top_labels(A, f, L))
                s[f"oos_AB@{L}"] = coverage(B, f, top_labels(A, f, L))
                s[f"oos_BA@{L}"] = coverage(A, f, top_labels(B, f, L))
                # n-check: in-sample share on a random half of A (~2^13 faces) and on pooled A+B (~2^15)
                half = [p for p in A if rng.random() < 0.5]
                s[f"in_half@{L}"] = coverage(half, f, top_labels(half, f, L))
                s[f"in_AB@{L}"] = coverage(A + B, f, top_labels(A + B, f, L))
            s["within"] = within_tail(A, f)
            r["schemes"][name] = s
        # nulls
        gA = np.sort([p["g2"] for p in A])[::-1]
        r["null_singleton"] = {L: float(gA[:L].sum() / gA.sum()) for L in LS}
        K = r["schemes"]["mech"]["nlabels_AB"]
        ra = {id(p): int(rng.integers(K)) for p in A + B}
        fr = lambda p: ra[id(p)]
        r["null_random"] = {L: coverage(B, fr, top_labels(A, fr, L)) for L in LS}
        # composition (rule A tail mass shares)
        tot = sum(p["g2"] for p in A)
        comp = {}
        for key in ("kind", "wt", "ub", "nflips", "dist", "age", "x9", "side", "ncomp"):
            dd = mass_by(A, lambda p: p[key] if key != "nflips" else min(p[key], 5))
            comp[key] = {str(k): round(v / tot, 4) for k, v in sorted(dd.items(), key=lambda kv: -kv[1])}
        comp["blind_npos<=1"] = round(sum(p["g2"] for p in A if p["npos"] <= 1) / tot, 4)
        fl = defaultdict(float)
        for p in A:
            for f_ in set(p["ft"]):
                fl[f_] += p["g2"]
        comp["flip_in_band(step,neuron)_top12"] = {str(k): round(v / tot, 4) for k, v in
                                                   sorted(fl.items(), key=lambda kv: -kv[1])[:12]}
        r["composition"] = comp
        R[m] = r
    # cross-copy: orig's top-L (rule A) labels covering each copy's rule-B tail
    if all(m in R for m in ("orig", "s005", "s010")):
        T = 10
        tails = {m: ([derived(p, T) for p in json.load(open(os.path.join(HERE, "results", f"{m}.json")))
                      ["rules"][k]["tail"] if p.get("kind")]) for m in ("orig", "s005", "s010") for k in ["700"]}
        tailsB = {m: [derived(p, T) for p in json.load(open(os.path.join(HERE, "results", f"{m}.json")))
                      ["rules"]["701"]["tail"] if p.get("kind")] for m in ("orig", "s005", "s010")}
        cc = {}
        for name in ("mech", "pos", "relpos", "nopos"):
            f = SCHEMES[name]
            cc[name] = {f"{m}@{L}": coverage(tailsB[m], f, top_labels(tails["orig"], f, L))
                        for m in ("orig", "s005", "s010") for L in LS}
        R["cross_copy_from_orig"] = cc
    json.dump(R, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=str)
    for m, r in R.items():
        if m == "cross_copy_from_orig":
            print("\ncross-copy coverage (labels from orig rule A, evaluated on copy rule B):")
            for name, v in r.items():
                print(f"  {name:9s}", " ".join(f"{k}={x:.3f}" for k, x in v.items()))
            continue
        print(f"\n== {m} (T={r['T']}, n={r['n']}/kind, tail points={r['ntail']}) top5% share "
              f"{r['top5'][0]:.3f}/{r['top5'][1]:.3f}, top1% {r['top1'][0]:.3f}/{r['top1'][1]:.3f}")
        print("  null singleton (in-sample top-L points):", {L: round(v, 3) for L, v in r["null_singleton"].items()},
              " null random labels (oos):", {L: round(v, 3) for L, v in r["null_random"].items()})
        for name, s in r["schemes"].items():
            print(f"  {name:9s} labels A={s['nlabels_A']:5d} A+B={s['nlabels_AB']:5d} | " +
                  " ".join(f"L{L}: in {s[f'in_A@{L}']:.3f} oosAB {s[f'oos_AB@{L}']:.3f} oosBA "
                           f"{s[f'oos_BA@{L}']:.3f} half {s[f'in_half@{L}']:.3f} AB {s[f'in_AB@{L}']:.3f} |"
                           for L in LS))
            w = s["within"]
            print(f"            within-label top5% share: pooled {w['pooled']:.3f}, top labels "
                  f"{[round(x, 2) for x in w['labels']]} (pts {w['npts']})")
        for k, v in r["composition"].items():
            print(f"  comp {k}: {dict(list(v.items())[:10]) if isinstance(v, dict) else v}")


if __name__ == "__main__":
    main()
