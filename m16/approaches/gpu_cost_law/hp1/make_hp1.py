"""T276: sealed held-out panel HP1 (TASKS Directive 7). 6 base models (orig = M_16,10, zoo (16,10) seeds 0-4) + per base
8 fresh noisy copies at each of: ARC absolute sigma .05 (W + .05 N), relative sigma .005 and .01 (W + s rms(W) N, the
leaderboard convention). Noise RNG: numpy default_rng(SEED) with SEED = 276000000 + 1000 * group + 100 * base + copy
(group 0 = abs.05, 1 = rel.005, 2 = rel.01; base 0 = orig, 1..5 = zoo seed 0..4): disjoint from LB3's _noisy seeds
(small integers). Writes hp1/weights.npz (arrays Wi, Wh, Wo stacked in manifest order) and hp1/manifest.json."""
import hashlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..", "..", "..")
sys.path[:0] = [os.path.join(ROOT, "m16"), os.path.join(ROOT, "m16", "approaches", "understand_m43")]
from common import Wi as Wi0, Wh as Wh0, Wo as Wo0
from zoo44 import zoo
bases = [("orig", (Wi0, Wh0, Wo0))] + [(f"zoo{s}", zoo(16, 10, s)) for s in range(5)]
groups = [("abs0.05", 0.05, False), ("rel0.005", 0.005, True), ("rel0.01", 0.01, True)]
names, W, man = [], [], []
for bi, (bn, base) in enumerate(bases):
    names.append(bn); W.append(base); man.append(dict(name=bn, base=bn, kind="clean", sigma=0.0, seed=None))
for gi, (gn, sig, rel) in enumerate(groups):
    for bi, (bn, base) in enumerate(bases):
        for c in range(8):
            seed = 276000000 + 1000 * gi + 100 * bi + c; rng = np.random.default_rng(seed)
            Wn = tuple(w + sig * (np.sqrt((w ** 2).mean()) if rel else 1.0) * rng.standard_normal(w.shape) for w in base)
            nm = f"{bn}_{gn}_c{c}"; names.append(nm); W.append(Wn)
            man.append(dict(name=nm, base=bn, kind=gn, sigma=sig, relative=rel, seed=seed))
Wi = np.stack([w[0] for w in W]); Wh = np.stack([w[1] for w in W]); Wo = np.stack([w[2] for w in W])
np.savez(os.path.join(HERE, "weights.npz"), Wi=Wi, Wh=Wh, Wo=Wo, names=np.array(names))
h = hashlib.sha256(Wi.tobytes() + Wh.tobytes() + Wo.tobytes()).hexdigest()
json.dump(dict(panel="HP1", n_models=len(names), weights="weights.npz", weights_sha256=h, models=man,
               noise_rng="numpy.random.default_rng(seed) standard_normal per matrix in order Wi, Wh, Wo",
               truth="sealed/truths.json (MC >= 1e8 per model, independent torch Philox stream per model)"),
          open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
print(len(names), "models, weights sha256", h)
