"""T270 diagnosis: per-subtree leaf-mass sum vs root mass for the positions of one chunk (CPU masses).
usage: python diag6.py CHUNK_NPZ [workers] -> results/prefix6/diag_<chunk>.json"""
import json, multiprocessing as mp, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import prefix6 as Q, prefix as P, gmass  # noqa: E402
f = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
pos = np.load(f)["positions"].tolist(); roots = np.load(os.path.join(Q.OUT, "roots.npz"))["mass"]
model, front = Q.front_list(11); out = []
with mp.get_context("fork").Pool(w, P._init, (model, Q.K, True)) as pool:
    for p, (n, pt, o, s, sec) in zip(pos, pool.imap(P._dfs, [front[i] for i in pos])):
        hists, Rs, Is = pt
        cm = gmass.grouped(Rs, Is, len(hists), "cpu")
        out.append(dict(pos=p, cells=n, root=float(roots[p]), leaf=float(cm.sum()), diff=float(cm.sum() - roots[p])))
        print(out[-1], flush=True)
json.dump(out, open(os.path.join(Q.OUT, "diag_" + os.path.basename(f).replace(".npz", ".json")), "w"), indent=1)
