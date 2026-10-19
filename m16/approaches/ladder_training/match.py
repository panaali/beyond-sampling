"""T86 match test (pre-registered in README): M1 accuracy, M2 storage pattern, M3 error geometry, ours vs zoo.
usage: python match.py [tag=30s]   reads results.json, prints a markdown table, writes match.json"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "30s"
res = json.load(open(os.path.join(HERE, "results.json")))
REPRO = {(4, 3): [(4, 3)], (5, 3): [(5, 3)]}
NEW = {(7, 3): [(6, 3), (8, 3)], (4, 7): [(4, 6), (4, 8)]}


def get(src, H, T, tag):
    return [res[k] for k in (f"{src}_{H}_{T}_{tag}_{s}" for s in range(5)) if k in res]


def widen(lo, hi, minw):
    w = hi - lo
    return lo - max(w / 2, minw), hi + max(w / 2, minw)


def store(r):
    return (r["n_store_pos"], r["n_store_neg"])


out = {}
print("| size | ref | our acc | zoo acc range | R | M1 | our n_store | zoo n_store | M2 | our bnd1 median | bnd1 window | M3 | match |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (H, T), refs in {**REPRO, **NEW}.items():
    ours = get("ours", H, T, TAG)
    zoo = sum((get("zoo", h, t, "30") for h, t in refs), [])
    if len(ours) < 5 or not zoo:
        print(f"| ({H},{T}) | incomplete: {len(ours)} of ours, {len(zoo)} zoo |")
        continue
    a = np.array([r["acc"] for r in ours]); za = np.array([r["acc"] for r in zoo])
    lo, hi = widen(za.min(), za.max(), 0.005)
    m1 = bool(((a >= lo) & (a <= hi)).all() and abs(np.median(a) - np.median(za)) <= max(za.max() - za.min(), 0.01))
    if (H, T) in REPRO:
        mode = Counter(r["n_store"] for r in zoo).most_common(1)[0][0]
        m2 = sum(r["n_store"] == mode for r in ours) >= 3
        zs = mode
    else:  # per chart, between the neighbours (inclusive), sorted pattern compared elementwise
        pats = [tuple(sorted(store(r))) for r in zoo]
        lo2 = tuple(min(p[i] for p in pats) for i in range(2)); hi2 = tuple(max(p[i] for p in pats) for i in range(2))
        m2 = all(all(lo2[i] <= tuple(sorted(store(r)))[i] <= hi2[i] for i in range(2)) for r in ours)
        zs = f"{lo2[0]}/{lo2[1]}..{hi2[0]}/{hi2[1]}"
    zb = np.array([r["bnd1"] for r in zoo]); blo, bhi = widen(zb.min(), zb.max(), 0.05)
    bm = float(np.median([r["bnd1"] for r in ours])); m3 = blo <= bm <= bhi
    ok = m1 and m2 and m3
    out[f"{H}_{T}"] = dict(acc=a.tolist(), zoo_acc=za.tolist(), R=[lo, hi], M1=m1, M2=bool(m2), M3=bool(m3), match=ok,
                           bnd1_median=bm, bnd1_window=[blo, bhi])
    yn = lambda b: "yes" if b else "**no**"
    print(f"| ({H},{T}) | {'zoo' if (H, T) in REPRO else 'zoo ' + '+'.join(map(str, refs))} | {' '.join('%.4f' % x for x in a)} | "
          f"{za.min():.4f}-{za.max():.4f} | {lo:.4f}-{hi:.4f} | {yn(m1)} | {' '.join(r['n_store'] for r in ours)} | {zs} | {yn(m2)} | "
          f"{bm:.3f} | {max(blo, 0):.3f}-{min(bhi, 1):.3f} | {yn(m3)} | {yn(ok)} |")
json.dump(out, open(os.path.join(HERE, "match.json"), "w"), indent=1)
