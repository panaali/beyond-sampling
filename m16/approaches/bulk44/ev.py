"""T165 scoring. usage: OMP_NUM_THREADS=1 python ev.py 44|54|64|43k20 (H>4: recipe not run)  -> ev_<tag>.json, <=10 summary lines.
Panel: zoo seeds 0-4 clean + 4 ARC noisy copies each (abs sigma 0.05, noise seed 50, as run87).
Truth: MC 1e7 (rng 7 noisy / 100+s clean, as run87) and a high-resolution Hopf reference (validated against MC).
Sampling baseline: RMSE^2 = p(1-p)/(ops/ops_per_sample), ops_per_sample = T H (H+1) + T H (as run87).
(4,4) estT recipe rows are reused from understand_m43/run87.json (same models, same truth draws);
(4,3) k=20: estT recipe (eps 0.1, the T87 recipe) is run here."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("understand_m43", "arc_m43_repro", "ladder_training_axis"):
    sys.path.insert(0, os.path.join(HERE, "..", p))
import numpy as np
from m43 import Ops
from zoo44 import zoo, mc_acc, ops_per_sample
from zoo43 import noisy_copies
import hopf
tag = sys.argv[1]; T = 3 if tag == "43k20" else 4; H = int(tag[0]); OPS_S = ops_per_sample(H, T)
LEVELS = (8, 16, 32, 64) if T == 4 else (16, 64, 256, 1024); REF = 128 if T == 4 else 16384


def base(s):
    if T == 4:
        return zoo(H, 4, s)
    import lt
    return tuple(np.asarray(w, float) for w in lt.zoo(20, s))


def score(m, mc, est_recipe=None):
    r = dict(mc=mc); o = Ops(); r["ref"] = hopf.est_hopf(m, REF, o); r["ref_ops"] = o.n
    r["z_ref_mc"] = (mc - r["ref"]) / np.sqrt(r["ref"] * (1 - r["ref"]) / 1e7)
    for nu in LEVELS:
        o = Ops(); r[f"h{nu}"] = hopf.est_hopf(m, nu, o); r[f"h{nu}_ops"] = o.n
    if est_recipe is None:
        import estT
        o = Ops(); est_recipe = (estT.estT(m, o), o.n)
    r["recipe"], r["recipe_ops"] = est_recipe
    return r


def ratio(rows, key, ops_key, truth="ref"):
    p = np.array([q[truth] for q in rows]); e = np.array([q[key] for q in rows]) - p
    mse = np.mean(e ** 2) - (np.mean(p * (1 - p)) / 1e7 if truth == "mc" else 0)
    return float(max(mse, 0) / (np.mean(p * (1 - p)) / (np.mean([q[ops_key] for q in rows]) / OPS_S)))


R87 = json.load(open(os.path.join(HERE, "..", "understand_m43", "run87.json"))) if (T, H) == (4, 4) else None
out = {}
for s in range(5):
    b = base(s); rows = []
    for i, m in enumerate([b] + noisy_copies(b, 4, 0.05, 50)):
        if R87 is None and T == 4:
            mc = mc_acc(m, 10_000_000, np.random.default_rng(100 + s if i == 0 else 7)); rec = (np.nan, 1)
        elif T == 4:
            q = R87["clean"][str(s)] if i == 0 else R87["noisy"][str(s)]["rows"][i - 1]
            mc = q["mc1e7"] if i == 0 else q["truth"]; rec = (q["est"], q["ops"])
        else:
            mc = mc_acc(m, 10_000_000, np.random.default_rng(100 + s if i == 0 else 7)); rec = None
        rows.append(score(m, mc, rec))
    out[s] = dict(rows=rows, clean={k: ratio(rows[:1], k, k + "_ops") for k in ["recipe"] + [f"h{n}" for n in LEVELS]},
                  noisy={k: ratio(rows[1:], k, k + "_ops") for k in ["recipe"] + [f"h{n}" for n in LEVELS]},
                  noisy_mc_truth={k: ratio(rows[1:], k, k + "_ops", "mc") for k in ["recipe", f"h{LEVELS[-1]}"]})
    zs = [q["z_ref_mc"] for q in rows]
    print(f"s{s} acc {rows[0]['ref']:.4f} |z(ref-MC)|max {max(map(abs, zs)):.1f} | clean ratio recipe {out[s]['clean']['recipe']:.3g} "
          + " ".join(f"h{n}:{out[s]['clean'][f'h{n}']:.2g}" for n in LEVELS) + f" | noisy recipe {out[s]['noisy']['recipe']:.3g} "
          + " ".join(f"h{n}:{out[s]['noisy'][f'h{n}']:.2g}" for n in LEVELS)
          + f" | ops h{LEVELS[-1]} {rows[0][f'h{LEVELS[-1]}_ops']/1e6:.2f}M recipe {rows[0]['recipe_ops']/1e6:.2f}M", flush=True)
    json.dump(out, open(os.path.join(HERE, f"ev_{tag}.json"), "w"), indent=0, default=float)
