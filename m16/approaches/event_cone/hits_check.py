"""Soundness test (DRI): every MC hit (pick = p3, g23 >= 2.5) of triple TRI must have its full 10-step gate history among
the k = 10 survivors (union of all dfs shard files). Also reports the union size and duplicates across shards.
usage: python hits_check.py 9,3,2 [hits.npz]"""
import glob
import sys

import numpy as np

import common_path  # noqa: F401
import common

tri = tuple(int(v) for v in sys.argv[1].split(",")); ts = "-".join(map(str, tri))
hp = sys.argv[2] if len(sys.argv) > 2 else "hits/prob_2.5_hits.npz"
files = sorted(glob.glob(f"certs/surv_dfs_d0.333_{ts}_k4to10_s*.npz"))
G = [np.load(f)["gates"] for f in files]
allg = np.concatenate(G) if G else np.zeros((0, 10, 16), np.int8)
S = {g.tobytes() for g in allg}
print(f"{len(files)} shard files, {len(allg)} survivors, {len(S)} distinct (duplicates {len(allg) - len(S)})")
x = np.load(hp)["x"]
o = np.argsort(-x, 1)
sel = [(tuple(int(v) for v in o[i, :3]) == tri) for i in range(len(x))]
x = x[np.array(sel)]
hs = (common.run(x)[:, 1:] > 0).astype(np.int8)
norm = (x[:, tri[1]] - x[:, tri[2]]) / np.abs(x).max(1)
inside = [h.tobytes() in S for h in hs]
for i in range(len(x)):
    print(f"hit {i}: g23/||x||inf = {norm[i]:.3f} (>= 1/3: {norm[i] >= 1/3}), history in survivors: {inside[i]}")
print(f"hits of {tri}: {len(x)}; in survivors: {sum(inside)}; outside: {len(x) - sum(inside)}")
