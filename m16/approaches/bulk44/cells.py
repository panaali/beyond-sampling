"""Mass and error rate of the final all-off / final all-on cells and the error share of each (MC 1e6)."""
import sys, os, json; import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "understand_m43")); from zoo44 import zoo
out = {}
for s in range(5):
    Wi, Wh, Wo = zoo(4, 4, s); x = np.random.default_rng(s).standard_normal((1_000_000, 4)); h = np.zeros((len(x), 4)); ndead = np.zeros(len(x))
    for t in range(4):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    pred = (h @ Wo.T).argmax(1); bad = pred != np.argsort(-x, 1)[:, 1]; off = (h == 0).all(1); on = (h > 0).all(1)
    nact = (h > 0).sum(1)
    out[s] = {f"nact{k}": [float((nact == k).mean()), float(bad[nact == k].mean()) if (nact == k).any() else 0.0] for k in range(5)}
    print(f"s{s} final #active neurons k: mass/err " + " ".join(f"{k}:{out[s][f'nact{k}'][0]:.2f}/{out[s][f'nact{k}'][1]:.2f}" for k in range(5)))
json.dump(out, open("cells_4_4.json", "w"))
