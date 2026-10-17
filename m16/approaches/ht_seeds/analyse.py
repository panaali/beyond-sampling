"""Tables and statistics for T2 from scan_seeds.json. usage: python analyse.py [--boot 2000]
Gain = MSE gain over plain sampling at equal cost: p(1-p)/cost / MSE, with p = independent truth.
  var-gain: MSE = variance over rule seeds (chi-square CI, R-1 d.o.f.), as sci_tao scan_HT.py;
  mse-gain: MSE = mean (est - truth)^2 - truth s.e.^2 (bias included); CI = bootstrap over rules and truth.
"""
import argparse, json, os
import numpy as np
from scipy.stats import chi2, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--wellacc", type=float, default=0.97)
A = ap.parse_args()
D = json.load(open(os.path.join(HERE, "scan_seeds.json")))
rng = np.random.default_rng(0)


def gains(blk, tr):
    e = np.array(blk["est"]); c = np.mean(blk["cost"]); R = len(e)
    pt, se = tr["acc"], tr["se"]; s0 = pt * (1 - pt) / c
    v = e.var(ddof=1); g = s0 / v
    vci = [g * chi2.ppf(0.025, R - 1) / (R - 1), g * chi2.ppf(0.975, R - 1) / (R - 1)]
    mse = max(np.mean((e - pt) ** 2) - se ** 2, v / R)
    bs = []
    for _ in range(A.boot):
        eb = e[rng.integers(0, R, R)]; pb = pt + se * rng.standard_normal()
        bs.append(s0 / max(np.mean((eb - pb) ** 2) - se ** 2, eb.var(ddof=1) / R, 1e-30))
    bias = e.mean() - pt; bz = bias / np.sqrt(v / R + se ** 2)
    return dict(var=g, var_ci=vci, mse=s0 / mse, mse_ci=list(np.percentile(bs, [2.5, 97.5])),
                bias=bias, bias_z=bz, tail=float(np.mean(blk["tail"])), cost=c, sd=float(np.sqrt(v)))


rows = []
for k, r in D.items():
    if "truth" not in r: continue
    H, T, sd = map(int, k.split(":"))
    s = gains(r["sobol"], r["truth"]); i = gains(r["iid"], r["truth"]) if "iid" in r else None
    rows.append(dict(H=H, T=T, seed=sd, acc=r["truth"]["acc"], tse=r["truth"]["se"], ntruth=r["truth"]["n"],
                     s=s, i=i, cancel=r["cancel"], R=len(r["sobol"]["est"])))
rows.sort(key=lambda x: (x["H"], x["T"], x["seed"]))
f = lambda x: f"{x:.0f}" if x >= 10 else f"{x:.1f}"
print("| H | T | seed | acc (truth) | Sobol MSE gain [95% CI] | Sobol var gain [CI] | bias z | iid var gain | tail | cancel |")
print("|---|---|---|---|---|---|---|---|---|---|")
for x in rows:
    s, i = x["s"], x["i"]
    print(f"| {x['H']} | {x['T']} | {x['seed']} | {x['acc']:.4f} | {f(s['mse'])} [{f(s['mse_ci'][0])}, {f(s['mse_ci'][1])}] | "
          f"{f(s['var'])} [{f(s['var_ci'][0])}, {f(s['var_ci'][1])}] | {s['bias_z']:+.1f} | {f(i['var']) if i else '-'} | "
          f"{s['tail']:.2f} | {x['cancel']:.0f} |")

# ---- collapse points: first T (in the scanned list) whose Sobol MSE-gain CI lies wholly below 10x
print("\nCollapse per (H, seed): first T with MSE-gain CI upper < 10 ('lost'), last T with CI lower > 10 ('won')")
coll = {}
for H in sorted({x["H"] for x in rows}):
    for sd in range(5):
        xs = [x for x in rows if x["H"] == H and x["seed"] == sd]
        if not xs: continue
        lost = [x["T"] for x in xs if x["s"]["mse_ci"][1] < 10]
        won = [x["T"] for x in xs if x["s"]["mse_ci"][0] > 10]
        coll[(H, sd)] = (min(lost) if lost else None, max(won) if won else None)
        seq = " ".join(f"T{x['T']}:{f(x['s']['mse'])}({x['acc']:.3f})" for x in xs)
        print(f"H={H:2d} seed={sd}: lost at T={coll[(H, sd)][0]}, won up to T={coll[(H, sd)][1]} | {seq}")


# ---- rank correlations, bootstrap over models (and over rules inside each model via its CI-free point est.)
def spear_ci(xs, key_x, key_y):
    X = np.array([key_x(x) for x in xs]); Y = np.array([key_y(x) for x in xs]); n = len(X)
    rho = spearmanr(X, Y).correlation; bs = []
    for _ in range(A.boot):
        ii = rng.integers(0, n, n); bs.append(spearmanr(X[ii], Y[ii]).correlation)
    return rho, np.nanpercentile(bs, [2.5, 97.5]), n


lg = lambda x: np.log(x["s"]["mse"])
print("\nSpearman rank correlation with log Sobol MSE-gain [95% bootstrap CI over models]")
for lab, sel in (("all", lambda x: True), (f"acc>{A.wellacc}", lambda x: x["acc"] > A.wellacc),
                 (f"acc<={A.wellacc}", lambda x: x["acc"] <= A.wellacc)):
    xs = [x for x in rows if sel(x)]
    if len(xs) < 5: continue
    for nm, kx in (("tail", lambda x: x["s"]["tail"]), ("T", lambda x: x["T"]),
                   ("log err", lambda x: np.log(1 - x["acc"])), ("log cancel", lambda x: np.log(x["cancel"]))):
        rho, ci, n = spear_ci(xs, kx, lg)
        print(f"  {lab:10s} n={n:3d} {nm:10s} rho={rho:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")

# within-T (fixed face dimension): does tail still rank the gain across H and seeds?
print("\nWithin fixed T (same quadrature dimension), well-trained models: Spearman(tail, log gain)")
for T in sorted({x["T"] for x in rows}):
    xs = [x for x in rows if x["T"] == T and x["acc"] > A.wellacc]
    if len(xs) < 5: continue
    r1, c1, n = spear_ci(xs, lambda x: x["s"]["tail"], lg)
    r2, c2, _ = spear_ci(xs, lambda x: np.log(1 - x["acc"]), lg)
    print(f"  T={T:2d} n={n:2d}: tail rho={r1:+.2f} [{c1[0]:+.2f}, {c1[1]:+.2f}]  log err rho={r2:+.2f} [{c2[0]:+.2f}, {c2[1]:+.2f}]")

# stratified (pooled within-T) rank correlation: rank-transform within each T, then pool
def pooled_within(xs, kx):
    X, Y = [], []
    for T in sorted({x["T"] for x in xs}):
        g = [x for x in xs if x["T"] == T]
        if len(g) < 3: continue
        from scipy.stats import rankdata
        X += list((rankdata([kx(x) for x in g]) - 0.5) / len(g)); Y += list((rankdata([lg(x) for x in g]) - 0.5) / len(g))
    return np.array(X), np.array(Y)

for lab, sel in (("all", lambda x: True), (f"acc>{A.wellacc}", lambda x: x["acc"] > A.wellacc)):
    xs = [x for x in rows if sel(x)]
    for nm, kx in (("tail", lambda x: x["s"]["tail"]), ("log err", lambda x: np.log(1 - x["acc"])),
                   ("log cancel", lambda x: np.log(x["cancel"]))):
        X, Y = pooled_within(xs, kx); n = len(X)
        rho = np.corrcoef(X, Y)[0, 1]
        bs = [np.corrcoef(X[ii], Y[ii])[0, 1] for ii in (rng.integers(0, n, n) for _ in range(A.boot))]
        print(f"pooled within-T {lab:9s} n={n:3d} {nm:10s} rho={rho:+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")

json.dump([{**{k: v for k, v in x.items() if k not in ('s', 'i')}, "sobol": x["s"], "iid": x["i"]} for x in rows],
          open(os.path.join(HERE, "table.json"), "w"), indent=1, default=float)
