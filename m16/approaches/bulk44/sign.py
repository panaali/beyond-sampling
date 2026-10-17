"""Error rate by sign pattern of x and by gate pattern sequence (top families). usage: python sign.py H T"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "understand_m43"))
from zoo44 import zoo
H, T = int(sys.argv[1]), int(sys.argv[2]); out = {}
for s in range(5):
    Wi, Wh, Wo = zoo(H, T, s)
    x = np.random.default_rng(s).standard_normal((1_000_000, T)); h = np.zeros((len(x), H)); seq = np.zeros(len(x), int)
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0); seq = seq * 16 + (h > 0) @ (1 << np.arange(H))
    pred = (h @ Wo.T).argmax(1); o = np.argsort(-x, 1); bad = pred != o[:, 1]
    sg = (x > 0) @ (1 << np.arange(T))
    r = {int(k): [float((sg == k).mean()), float(bad[sg == k].mean())] for k in range(2 ** T)}
    v, c = np.unique(seq[bad], return_counts=True); i = np.argsort(-c)[:4]
    vt, ct = np.unique(seq, return_counts=True)
    fam = []
    for j in i:
        q = seq == v[j]; fam.append(dict(seq=[int(v[j]) >> (4 * (T - 1 - t)) & 15 for t in range(T)], share=float(c[j] / bad.sum()),
                                         err_in=float(bad[q].mean()), mass=float(q.mean())))
    out[s] = dict(sign=r, fams=fam, n_seq=int(len(vt)))
    worst = sorted(r.items(), key=lambda kv: -kv[1][0] * kv[1][1])[:4]
    print(f"s{s}: sign(x0..x3 bits) worst " + " ".join(f"{k:04b}[::-1]:{e:.2f}" for k, (m, e) in worst) +
          " | gate seqs " + " ".join(f"{f['seq']} sh{f['share']:.2f} e{f['err_in']:.2f}" for f in fam[:3]))
json.dump(out, open(f"sign_{H}_{T}.json", "w"), indent=0)
