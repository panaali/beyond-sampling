"""Per ordering chamber (positions sorted largest first) error rate and dominant wrong answer. usage: python cham.py H T"""
import sys, os, json, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "understand_m43"))
from zoo44 import zoo
H, T = int(sys.argv[1]), int(sys.argv[2]); out = {}
for s in range(5):
    Wi, Wh, Wo = zoo(H, T, s)
    x = np.random.default_rng(s).standard_normal((1_000_000, T)); h = np.zeros((len(x), H))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    pred = (h @ Wo.T).argmax(1); o = np.argsort(-x, 1); key = o @ (T ** np.arange(T)[::-1])
    allon = (h > 0).all(1); rows = []
    for p in itertools.permutations(range(T)):
        k = np.array(p) @ (T ** np.arange(T)[::-1]); m = key == k; bad = pred[m] != p[1]
        wa = np.bincount(pred[m][bad], minlength=T) / max(1, bad.sum())
        rows.append(dict(order="".join(map(str, p)), err=float(bad.mean()), wrong=int(wa.argmax()), wrong_frac=float(wa.max()),
                         allon_err=float(allon[m][bad].mean()) if bad.any() else 0.0))
    out[s] = rows; bulk = [r for r in rows if r["err"] > 0.8]; good = [r for r in rows if r["err"] < 0.1]
    print(f"s{s}: >80%-wrong chambers " + " ".join(f"{r['order']}->{r['wrong']}" for r in bulk) +
          f" | <10%: {len(good)} | mid: " + " ".join(f"{r['order']}:{r['err']:.2f}" for r in rows if 0.1 <= r['err'] <= 0.8))
json.dump(out, open(f"cham_{H}_{T}.json", "w"), indent=0)
