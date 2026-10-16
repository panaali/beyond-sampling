"""Compute-adjusted MSE of great-circle estimators vs plain sampling and stratified.py, from the pools."""
import sys, json, glob
import numpy as np
HERE = "/home/user/clean-neurons/m16/approaches/great_circle"
sys.path.insert(0, "/home/user/clean-neurons/m16")
from stratified import stratum, N_STRATA
D = json.load(open("/home/user/clean-neurons/m16/stratified_design.json")); P = np.array(D["P"]); AL = np.array(D["alloc"])
BUD = [16_000, 64_000, 256_000]

# CV coefficients fitted once on an independent pilot of the ORIGINAL model (pilot_0.npz), then frozen.
pl = np.load(f"{HERE}/pilot_0.npz"); G = 512; th = np.arange(G) / G * 2 * np.pi
Fp = np.stack([np.bincount(stratum(np.cos(th)[:, None] * pl["U"][i] + np.sin(th)[:, None] * pl["V"][i]), minlength=N_STRATA) / G
               for i in range(len(pl["f"]))])
X = Fp[:, :-1] - P[:-1]
beta = np.linalg.lstsq(np.c_[np.ones(len(X)), X], pl["f"], rcond=None)[0][1:]

rows = []; emp = {b: {"gc": [], "pred": []} for b in BUD}
for fn in sorted(glob.glob(f"{HERE}/pool_*.npz"), key=lambda s: int(s.split("_")[-1][:-4])):
    d = np.load(fn); idx = int(fn.split("_")[-1][:-4]); tr = float(d["truth"])
    f, p, pc, F = d["f"], d["p"], d["pc"], d["feat"]
    nf = len(f) // 5
    fcv = f - (F[:, :-1] - P[:-1]) @ beta
    K, KP = p.mean(), pc.mean()
    z = (f.mean() - tr) / (f.std() / np.sqrt(len(f)))
    zcv = (fcv.reshape(nf, 5).mean(1).mean() - tr) / (fcv.reshape(nf, 5).mean(1).std() / np.sqrt(nf))
    s_samp = tr * (1 - tr)                                   # var per eval
    n_h = np.maximum(2, np.round(AL * 1e6)); s_strat = (P**2 * d["sv"] / n_h).sum() * n_h.sum()   # var * evals
    v_ind = f.var() * K
    v_fr = f.reshape(nf, 5).mean(1).var() * 5 * K
    v_cv = fcv.reshape(nf, 5).mean(1).var() * 5 * K
    r = dict(model=idx, truth=tr, bias_z=z, bias_z_cv=zcv, pieces=K, partial=KP,
             gain_strat=s_samp / s_strat, gain_gc=s_samp / v_ind, gain_gc_frame=s_samp / v_fr,
             gain_gc_frame_cv=s_samp / v_cv, gain_gc_frame_cv_partial=s_samp / (v_cv * KP / K))
    for b in BUD:
        r[f"rmse_samp_{b}"] = np.sqrt(s_samp / b); r[f"rmse_strat_{b}"] = np.sqrt(s_strat / b)
        r[f"rmse_gcfcv_{b}"] = np.sqrt(v_cv / b)
        # empirical: disjoint blocks of frames using ~b pieces each
        k = max(1, int(round(b / (5 * K)))); g = fcv.reshape(nf, 5).mean(1)
        for j in range(nf // k):
            emp[b]["gc"].append((g[j * k:(j + 1) * k].mean() - tr) ** 2)
        emp[b]["pred"].append(v_cv / (5 * K * k))
    rows.append(r)
json.dump(rows, open(f"{HERE}/results.json", "w"), indent=1, default=float)
print("model truth    bias_z  z_cv  pieces partial | MSE gain vs sampling: strat   gc   gc+frame  gc+frame+cv  (+partial cost)")
for r in rows:
    print(f"{r['model']:2d} {r['truth']:.5f} {r['bias_z']:6.2f} {r['bias_z_cv']:5.2f} {r['pieces']:6.1f} {r['partial']:6.1f} | "
          f"{r['gain_strat']:5.2f} {r['gain_gc']:5.3f} {r['gain_gc_frame']:5.3f} {r['gain_gc_frame_cv']:6.3f} {r['gain_gc_frame_cv_partial']:6.3f}")
for grp, ids in (("orig", [0]), ("s=.0025", range(1, 7)), ("s=.005", range(7, 13))):
    rr = [r for r in rows if r["model"] in ids]
    if not rr: continue
    print(grp, " ".join(f"b={b}: RMSE pp samp {100*np.sqrt(np.mean([r[f'rmse_samp_{b}']**2 for r in rr])):.3f} "
                        f"strat {100*np.sqrt(np.mean([r[f'rmse_strat_{b}']**2 for r in rr])):.3f} "
                        f"gc+fr+cv {100*np.sqrt(np.mean([r[f'rmse_gcfcv_{b}']**2 for r in rr])):.3f};" for b in BUD))
for b in BUD:
    print(f"empirical check b={b}: {len(emp[b]['gc'])} blocks, empirical RMSE {100*np.sqrt(np.mean(emp[b]['gc'])):.3f} pp "
          f"vs predicted {100*np.sqrt(np.mean(emp[b]['pred'])):.3f} pp")
print("beta (CV, strata 0..5):", beta.round(3))
