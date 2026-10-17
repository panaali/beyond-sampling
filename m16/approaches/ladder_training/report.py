"""Print per-(H,T,training) tables from results.json.  usage: python report.py"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "results.json")))
groups = defaultdict(dict)
for k, r in res.items():
    src = k.split("_")[0]
    tag = k.split("_")[3]
    groups[(r["H"], r["T"], src, tag)][r["seed"]] = r

print("| H | T | source | n_train | acc seeds 0-4 | median acc | max acc | t1 (|t-1| worst chart) | bnd1 (err mass <1deg) | "
      "n_store | wall s/model |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (H, T, src, tag), d in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2] != "zoo", kv[0][3])):
    s = sorted(d)
    acc = [d[i]["acc"] for i in s]
    n = "2^" + tag.rstrip("s") + (" scaled-warmup" if tag.endswith("s") else (" default-warmup" if src == "ours" else ""))
    wall = np.mean([d[i].get("wall", np.nan) for i in s])
    print(f"| {H} | {T} | {src} | {n} | {' '.join('%.4f' % a for a in acc)}{'' if len(s) == 5 else ' (seeds %s)' % s} | "
          f"{np.median(acc):.4f} | {max(acc):.4f} | {' '.join('%.2f' % d[i]['t1'] for i in s)} | "
          f"{' '.join('%.2f' % d[i]['bnd1'] for i in s)} | {' '.join(d[i]['n_store'] for i in s)} | "
          f"{'' if np.isnan(wall) else '%.0f' % wall} |")
