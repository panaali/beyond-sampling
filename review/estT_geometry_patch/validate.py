"""T316 validation (pre-registered in README). Uses T270's code read-only, extracted at claude/t192-gpu-cost-law (commit in
README) into SRC. usage: T270_K=k python validate.py SRC MODE POSITIONS  (MODE: orig | patched | both; POSITIONS: all |
rand20 | comma list) -> val_k{k}_{tag}.json (incremental)"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.abspath(sys.argv[1]); sys.path[:0] = [HERE, SRC]
os.chdir(SRC)
import numpy as np
import prefix6 as Q, prefix as P, gmass, estG  # noqa: E402
import estT_base as B  # noqa: E402
from m43 import Ops  # noqa: E402
import estT_geometry as G  # noqa: E402
mode, spec = sys.argv[2], sys.argv[3]
model, front = Q.front_list(11); P._init(model, Q.K, True)
pos = (list(range(len(front))) if spec == "all" else sorted(np.random.default_rng(316).choice(len(front), 20, replace=False).tolist())
       if spec == "rand20" else [int(p) for p in spec.split(",")])
ORIG = B.mass
roots6 = np.load(os.path.join(Q.OUT, "roots.npz"))["mass"] if Q.K == 6 else None


def run(which, item):
    B.mass = estG._tri_mass = ORIG
    if which == "patched":
        G.apply_estG(estG, B)
    n, (hists, Rs, Is), o, s, sec = P._dfs(item)
    leaf = float(gmass.grouped(Rs, Is, len(hists), "cpu").sum()) if len(Rs) else 0.0
    col = P._Col(); estG.BUF[0] = col; B.mass(item[0], Ops())
    root = float(gmass.grouped(np.array(col.R).reshape(-1, Q.K, Q.K), np.zeros(len(col.R), np.int64), 1, "cpu")[0])
    B.mass = estG._tri_mass = ORIG
    return dict(leaf=leaf, root=root, leaves=len(hists), simplices=len(Rs))


tag = spec if len(spec) < 20 else "list"; fn = os.path.join(HERE, f"val_k{Q.K}_{mode}_{tag.replace(',', '-')}.json")
out = json.load(open(fn)) if os.path.exists(fn) else {}
for p in pos:
    if str(p) in out:
        continue
    t0 = time.time(); r = {}
    for which in (("orig", "patched") if mode == "both" else (mode,)):
        for k in G.STATS:
            G.STATS[k] = 0
        r[which] = run(which, front[p]); r[which]["stats"] = dict(G.STATS)
    if roots6 is not None:
        r["root_npz"] = float(roots6[p])
    r["sec"] = time.time() - t0; out[str(p)] = r
    print(p, json.dumps(r), flush=True); json.dump(out, open(fn, "w"), indent=1)
