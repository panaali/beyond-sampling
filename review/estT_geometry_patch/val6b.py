"""T316 V1 on one k = 6 subtree, memory-bounded: walk with the selected patches, masses in batches.
usage: T270_K=6 python val6b.py SRC POS PATCHES (PATCHES: none | A | AB) -> val6b_{POS}_{PATCHES}.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.abspath(sys.argv[1]); sys.path[:0] = [HERE, SRC]
os.chdir(SRC)
import numpy as np
import prefix6 as Q, prefix as P, gmass, estG  # noqa: E402
import estT_base as B  # noqa: E402
from m43 import Ops  # noqa: E402
import estT_geometry as G  # noqa: E402
pos, which = int(sys.argv[2]), sys.argv[3]
model, front = Q.front_list(11); P._init(model, Q.K, True); root = float(np.load(os.path.join(Q.OUT, "roots.npz"))["mass"][pos])
if "A" in which:
    G.apply_estG(estG, B)
if "B" in which:
    G.apply_clip(B)
t0 = time.time(); stack = [front[pos]]; col = P._Col(); estG.BUF[0] = col; leaf = 0.0; leaves = 0; hists = set()


def flush():
    global leaf, col
    if col.R:
        leaf += float(gmass.grouped(np.array(col.R).reshape(-1, Q.K, Q.K), np.zeros(len(col.R), np.int64), 1, "cpu")[0])
    col = P._Col(); estG.BUF[0] = col


while stack:
    c, g, A, t, hist = stack.pop()
    if t == Q.K:
        leaves += 1; hists.add(hist); B.mass(c, Ops())
        if len(col.R) > 150_000:
            flush()
        continue
    for c2, g2, A2 in P._children(c, g, A, t, Ops()):
        stack.append((c2, g2, A2, t + 1, hist + np.packbits(g2 > 0).tobytes()))
flush()
r = dict(pos=pos, patches=which, root=root, leaf=leaf, diff=leaf - root, rel=(leaf - root) / root, leaves=leaves, regions=len(hists),
         statsA=dict(G.STATS), statsB=dict(G.STATS_B), sec=time.time() - t0)
print(json.dumps(r), flush=True); json.dump(r, open(os.path.join(HERE, f"val6b_{pos}_{which}.json"), "w"), indent=1)
