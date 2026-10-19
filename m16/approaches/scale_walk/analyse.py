"""Score (pre-registered): F per grid point and budget vs plain MC at equal FPE, bootstrap over models.
-> analysis.json, analysis.log (stdout)."""
import json
from collections import defaultdict

import numpy as np

import common as Cm

B = (4000, 16000, 64000); R0 = 31415; NB = 10_000
tr = {r["tag"]: r for r in Cm.jl_read("truths.jsonl")}
reg = {r["tag"]: r for r in Cm.registry()}
est = defaultdict(dict)
for r in Cm.jl_read("est.jsonl"):
    est[r["tag"]][(r["rule"], r["budget"])] = r
pts = defaultdict(list)
for t in reg:
    if t in tr and (R0, 64000) in est[t]:
        pts[(reg[t]["H"], reg[t]["T"])].append(t)
rng = np.random.default_rng(423)


def F_of(tags, b, rule=R0):
    p = np.array([tr[t]["acc"] for t in tags]); se = np.array([tr[t]["se"] for t in tags])
    e = np.array([est[t][(rule, b)]["est"] for t in tags])
    num = (e - p) ** 2 - se ** 2; den = p * (1 - p) / b
    return num, den


def score(tags, b):
    num, den = F_of(tags, b)
    F = num.sum() / den.sum()
    idx = rng.integers(0, len(tags), (NB, len(tags)))
    bs = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return F, lo, hi


def verdict(lo, hi, bar):
    return "holds" if hi < bar else ("fails" if lo > bar else "inconcl.")


out = {"points": {}, "validation": {}}
print("validation at 1M FPE (frozen rule): z = (est - truth) / sqrt(p(1-p)/1e6); cap hits")
for t in sorted(tr, key=lambda t: (reg[t]["H"], reg[t]["T"], t)):
    v = est[t].get((R0, 10 ** 6))
    if v:
        p = tr[t]["acc"]; z = (v["est"] - p) / np.sqrt(p * (1 - p) / 1e6)
        out["validation"][t] = {"z": z, "cap": v["cap"], "flag": bool(abs(z) > 5 or v["cap"] > 0),
                                "cells_per_line": v["cells"] / v["lines"], "maxcells": v["maxcells"]}
        print(f"  {t:14s} acc {p:.5f}  z {z:+6.2f}  cap {v['cap']}  cells/line {v['cells'] / v['lines']:.1f}"
              f"  max {v['maxcells']}{'  FLAG' if out['validation'][t]['flag'] else ''}")
print("\nF = MSE_walk / MSE_MC at equal FPE, frozen rule seed, panel = all registered models (95% bootstrap CI)")
print(f"{'H':>3} {'T':>3} {'n':>2} {'1-acc':>7} | " + " | ".join(f"{b // 1000}k: F [lo, hi] strict/lenient" for b in B))
for (H, T) in sorted(pts):
    tags = pts[(H, T)]; err = np.mean([1 - tr[t]["acc"] for t in tags])
    row = {"tags": tags, "err": err}; cells = []
    for b in B:
        F, lo, hi = score(tags, b)
        alt = []
        for k in range(1, 9):
            rule = R0 + 101 * 1000 * k
            if all((rule, b) in est[t] for t in tags):
                n, d = F_of(tags, b, rule); alt.append(n.sum() / d.sum())
        st = {s: score([t for t in tags if reg[t]["stream"] == s], b)[0] for s in ("cpu", "zoo")
              if any(reg[t]["stream"] == s for t in tags)}
        row[b] = {"F": F, "lo": lo, "hi": hi, "strict": verdict(lo, hi, 1), "lenient": verdict(lo, hi, 10),
                  "alt_rules": alt, "per_stream": st,
                  "lines": float(np.mean([est[t][(R0, b)]["lines"] for t in tags])),
                  "cells_per_line": float(np.mean([est[t][(R0, b)]["cells"] / est[t][(R0, b)]["lines"] for t in tags]))}
        cells.append(f"{F:6.2f} [{lo:5.2f}, {hi:6.2f}] {row[b]['strict'][:5]}/{row[b]['lenient'][:5]}")
    out["points"][f"{H},{T}"] = row
    print(f"{H:>3} {T:>3} {len(tags):>2} {err:7.4f} | " + " | ".join(cells))
print("\nsecondary: F over 8 alternative rule seeds (median [min, max]); per stream; lines per call; cells per line")
for k, row in out["points"].items():
    for b in B:
        r = row[b]; a = r["alt_rules"]
        s = f"med {np.median(a):.2f} [{min(a):.2f}, {max(a):.2f}]" if a else "-"
        print(f"  ({k}) {b // 1000:>2}k  alt {s}  streams {', '.join(f'{x} {v:.2f}' for x, v in r['per_stream'].items())}"
              f"  lines {r['lines']:.0f}  cells/line {r['cells_per_line']:.1f}")
# where each bar stops holding
print("\nfirst grid point (in H at fixed T, and in T at fixed H) where a bar no longer holds")
for bar in ("strict", "lenient"):
    for b in B:
        byT = defaultdict(list); byH = defaultdict(list)
        for k, row in out["points"].items():
            H, T = map(int, k.split(",")); byT[T].append((H, row[b][bar])); byH[H].append((T, row[b][bar]))
        sT = {T: next((H for H, v in sorted(l) if v != "holds"), None) for T, l in byT.items()}
        sH = {H: next((T for T, v in sorted(l) if v != "holds"), None) for H, l in byH.items()}
        out.setdefault("stops", {})[f"{bar}:{b}"] = {"H_at_T": sT, "T_at_H": sH}
        print(f"  {bar:7s} {b // 1000:>2}k  in H (per T): {sT}   in T (per H): {sH}")
# scaling fit
keys = list(out["points"])
HT = np.array([list(map(int, k.split(","))) for k in keys])
if len(keys) >= 5 and len(set(HT[:, 0])) > 1 and len(set(HT[:, 1])) > 1:
    print("\nscaling fit: log F = a + b log H + c T")
    for b in B:
        y = np.log([max(out["points"][k][b]["F"], 1e-3) for k in keys])
        X = np.c_[np.ones(len(keys)), np.log(HT[:, 0]), HT[:, 1]]
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        out.setdefault("fit", {})[b] = coef.tolist()
        print(f"  {b // 1000:>2}k: a {coef[0]:+.2f}  b(log H) {coef[1]:+.2f}  c(T) {coef[2]:+.3f}")
else:
    print("\nscaling fit: not enough grid points varying both H and T (pre-registered: descriptive only)")
json.dump(out, open(f"{Cm.HERE}/analysis.json", "w"), indent=1, default=float)
