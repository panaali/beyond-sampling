"""T388 scoring (pre-registered): graded-g12 (T366) est/truth on T371's 16 noisy misses and its 128 noisy hits (K3 within
2x), with T371's truths. Go: >= 12/16 misses within 2x AND <= 6/128 former hits outside 2x. Reported by truth tie share.
usage: python score_t388.py -> results/t388_score.json"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "results", "score.json")))
G = {}
for fn in glob.glob(os.path.join(HERE, "results", "t388_est_*.json")):
    G.update(json.load(open(fn)))
ok = lambda x: .5 <= x <= 2
rows = []
for r in S["rows"]:
    g = G[r["model"]]["est"][str(r["t0"])]
    rows.append(dict(model=r["model"], sigma=r["sigma"], t0=r["t0"], truth=r["truth"], tie=r["tie_share"], k3=r["ratio"],
                     graded=g / r["truth"], C_k3=r["C_fpe"], C_graded=G[r["model"]]["C_fpe"]))
noisy = [r for r in rows if r["sigma"] > 0]
miss = [r for r in noisy if not ok(r["k3"])]; hit = [r for r in noisy if ok(r["k3"])]
assert len(miss) == 16 and len(hit) == 128, (len(miss), len(hit))
fixed = sum(ok(r["graded"]) for r in miss); broken = sum(not ok(r["graded"]) for r in hit)


def summ(rs, key):
    x = np.array([r[key] for r in rs])
    return dict(n=len(rs), within2=int(sum(ok(v) for v in x)), median=float(np.median(x)) if len(x) else None,
                n_high=int((x > 1).sum()), n_over2=int((x > 2).sum()), n_under_half=int((x < .5).sum()))


out = dict(misses_fixed=int(fixed), hits_broken=int(broken), verdict="GO" if fixed >= 12 and broken <= 6 else "NO GO",
           all_noisy=dict(k3=summ(noisy, "k3"), graded=summ(noisy, "graded")),
           by_tie={f"[{lo},{hi})": dict(k3=summ([r for r in noisy if lo <= r["tie"] < hi], "k3"),
                                        graded=summ([r for r in noisy if lo <= r["tie"] < hi], "graded"))
                   for lo, hi in ((0, .5), (.5, .8), (.8, 1.01))},
           clean=[r for r in rows if r["sigma"] == 0], misses=miss, broken_hits=[r for r in hit if not ok(r["graded"])],
           C_ratio_median=float(np.median([r["C_graded"] / r["C_k3"] for r in rows])), C_graded_range=[min(r["C_graded"] for r in rows), max(r["C_graded"] for r in rows)])
json.dump(out, open(os.path.join(HERE, "results", "t388_score.json"), "w"), indent=1)
print(f"{'model':22s} {'t0':>5s} {'tie%':>5s} {'K3':>7s} {'graded':>7s}")
for r in miss + out["broken_hits"]:
    print(f"{r['model']:22s} {r['t0']:5.2f} {100 * r['tie']:5.1f} {r['k3']:7.3f} {r['graded']:7.3f}")
for k in ("misses_fixed", "hits_broken", "all_noisy", "by_tie", "C_ratio_median", "C_graded_range"):
    print(k, out[k])
print("clean", [(r["model"], r["t0"], round(r["k3"], 3), round(r["graded"], 3)) for r in out["clean"]])
print("VERDICT", out["verdict"])
