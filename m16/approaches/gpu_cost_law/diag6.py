"""T270 diagnosis: per-subtree leaf-mass sum vs root mass for the positions of one chunk (CPU masses computed in the
worker, only sums returned). usage: python diag6.py CHUNK_NPZ [workers] -> results/prefix6/diag_<chunk>.json"""
import json, multiprocessing as mp, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import prefix6 as Q, prefix as P, gmass  # noqa: E402
DEV = os.environ.get("DIAG_DEV", "cpu")


def one(item):
    n, (hists, Rs, Is), o, s, sec = P._dfs(item)
    cm = gmass.grouped(Rs, Is, len(hists), DEV, chunk=1024, tb=512) if len(Rs) else np.zeros(len(hists))
    return n, len(hists), len(Rs), float(cm.sum()), float(cm.min()) if len(cm) else 0.0


if __name__ == "__main__":
    f = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    pos = np.load(f)["positions"].tolist(); roots = np.load(os.path.join(Q.OUT, "roots.npz"))["mass"]
    model, front = Q.front_list(11); out = []
    pos = sorted(pos, key=lambda i: -roots[i])                       # largest root mass first
    with mp.get_context("fork").Pool(w, P._init, (model, Q.K, True)) as pool:
        for p, (n, nr, ns, leaf, mn) in zip(pos, pool.imap(one, [front[i] for i in pos])):
            out.append(dict(pos=p, cells=n, regions=nr, simplices=ns, root=float(roots[p]), leaf=leaf,
                            diff=leaf - float(roots[p]), min_region=mn))
            print(json.dumps(out[-1]), flush=True)
            json.dump(out, open(os.path.join(Q.OUT, "diag_" + os.path.basename(f).replace(".npz", ".json")), "w"), indent=1)
