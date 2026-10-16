"""Learning curve: does held-out R^2 keep rising with data (learnable but not compact) or saturate?
Train on data/orig_big.npz subsets, test on data/orig.npz (independent surface points).
usage: python curve.py"""
import numpy as np, json
import feats as F
big = F.load("orig", big=True); te = F.load("orig")
res = {}
for tname in ["delta", "mass"]:
    yte = F.targets(te)[tname]
    for n in [10000, 30000, 100000, 290000]:
        tr = {k: v[:n] for k, v in big.items()}; ytr = F.targets(tr)[tname]
        p, k1 = F.fit_cells(tr, ytr, te, "abp", "raw")
        ep = int(max(12, 60 * 39000 / n))
        q, k2 = F.mlp(tr, ytr, te, epochs=ep, hidden=256 if n < 100000 else 384)
        res[f"{tname}|{n}"] = dict(lin_abp=F.r2(p, yte), mlp=F.r2(q, yte))
        print(tname, n, "per-(a,b,p1) linear R2 %.3f   MLP R2 %.3f (params %d)" % (F.r2(p, yte), F.r2(q, yte), k2), flush=True)
json.dump(res, open("curve.json", "w"), indent=1)
