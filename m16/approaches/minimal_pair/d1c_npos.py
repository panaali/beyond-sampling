"""D1c: error mass and error-gap tail split by number of positive inputs (k), T=4 vs T=5, seeds 0-4."""
import json
import numpy as np
import core as C
out = {}
for T in (4, 5):
  for s in range(5):
    m = C.zoo(16, T, s); rng = np.random.default_rng(500 + s + 10*T)
    n = 0; E = np.zeros(T + 1); P = np.zeros(T + 1); Wd = np.zeros(T + 1); G2 = np.zeros(T + 1)
    for _ in range(16):
        x = rng.standard_normal((250_000, T)); n += len(x)
        pred = C.forward(m, x).argmax(1)
        order = np.argsort(-x, 1); rk = np.argsort(order, 1)[np.arange(len(x)), pred] + 1
        xs = -np.sort(-x, 1)
        g = np.where(rk == 1, xs[:, 0] - xs[:, 1], np.where(rk == 3, xs[:, 1] - xs[:, 2], 1.0))
        k = (x > 0).sum(1); bad = rk != 2
        P += np.bincount(k, minlength=T + 1); E += np.bincount(k[bad], minlength=T + 1)
        Wd += np.bincount(k[bad & (g > 0.05)], minlength=T + 1)
        G2 += np.bincount(k[bad], weights=g[bad] ** 2, minlength=T + 1)
    out[f"{T}_{s}"] = dict(P=(P / n).tolist(), E=(E / n).tolist(), wide=(Wd / n).tolist(), g2=(G2 / n).tolist())
    print(f"T{T} s{s} err {E.sum()/n:.5f} | by #pos k: err mass " + " ".join(f"{e/n:.2e}" for e in E)
          + " | cond err " + " ".join(f"{e/p:.4f}" for e, p in zip(E, P))
          + " | wide>0.05 " + " ".join(f"{w/n:.1e}" for w in Wd)
          + " | E[g^2;err] share " + " ".join(f"{v/G2.sum():.2f}" for v in G2), flush=True)
json.dump(out, open("d1c_npos.json", "w"), indent=1)
