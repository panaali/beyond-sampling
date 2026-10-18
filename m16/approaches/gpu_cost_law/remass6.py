"""T270 mass fix: recompute the subtrees of one chunk with the geometry defect repaired (see cx6.py: some k = 6 leaf
cells miss vertices and estT's pulling triangulation drops volume, so their masses are underestimates).
Pass 1: estT masses (as prefix6.py). A subtree whose leaf sum falls short of its exact root mass by more than
TOL = 1e-11 + 1e-8 * root is re-walked (pass 2): every leaf cell's rays are re-enumerated from its facet normals with
qhull (HalfspaceIntersection on the slice x . c = 1, c = normalized sum of inward normals, interior point by Chebyshev
LP) and the cell is triangulated by Delaunay of the slice points; a cell where qhull fails keeps its estT simplices.
Writes results/prefix6/remass_<chunk>.npz (same fields as the chunk + per-subtree pos / root / leaf_estT / leaf_fixed /
fixed / qhull_fail); prefix6_bounds.py uses it in place of the original chunk.
usage: python remass6.py CHUNK_NPZ [workers] [device]"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import Delaunay, HalfspaceIntersection

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import prefix6 as Q, prefix as P, gmass  # noqa: E402
from m43 import Ops  # noqa: E402

TOL_ABS, TOL_REL = 1e-11, 1e-8


def qhull_simplices(c):
    """unit-ray simplices (s, 6, 6) of cone {N x >= 0} from an independent vertex enumeration, or None on failure"""
    N = c.N / np.linalg.norm(c.N, axis=1, keepdims=True); d = N.shape[1]
    cdir = N.sum(0); cdir /= np.linalg.norm(cdir); U = np.linalg.svd(np.eye(d) - np.outer(cdir, cdir))[0][:, :d - 1]
    A = -(N @ U); b = -(N @ cdir); nrm = np.linalg.norm(A, axis=1)
    lp = linprog(np.r_[np.zeros(d - 1), -1], A_ub=np.c_[A, nrm], b_ub=-b, bounds=[(None, None)] * (d - 1) + [(0, None)])
    if lp.status != 0 or lp.x[-1] <= 1e-12:
        return None
    Z = HalfspaceIntersection(np.c_[A, b], lp.x[:d - 1]).intersections
    R = cdir + Z @ U.T; R /= np.linalg.norm(R, axis=1, keepdims=True)
    R = np.unique(np.round(R, 12), axis=0); Z = (R / (R @ cdir)[:, None] - cdir) @ U
    S = R[Delaunay(Z).simplices]
    return S / np.linalg.norm(S, axis=2, keepdims=True)


def fixed(item):
    """pass 2: re-walk the subtree (same DFS order as prefix._dfs) with qhull + Delaunay simplices for every leaf"""
    stack = [item]; hists, Rs, Is, fail = [], [], [], 0
    while stack:
        c, g, A, t, hist = stack.pop()
        if t == Q.K:
            i = len(hists); hists.append(hist)
            try:
                S = qhull_simplices(c)
            except Exception:  # noqa: BLE001  (qhull precision errors on extreme cells)
                S = None
            if S is None:
                fail += 1; col = P._Col(); P.estG.BUF[0] = col; P.B.mass(c, Ops()); S = np.array(col.R).reshape(-1, Q.K, Q.K)
            Rs.append(S); Is.append(np.full(len(S), i, np.int64)); continue
        for c2, g2, A2 in P._children(c, g, A, t, Ops()):
            stack.append((c2, g2, A2, t + 1, hist + np.packbits(g2 > 0).tobytes()))
    return hists, np.concatenate(Rs), np.concatenate(Is), fail


def main(f, workers=11, device="cuda"):
    pos = np.load(f)["positions"].tolist(); roots = np.load(os.path.join(Q.OUT, "roots.npz"))["mass"]
    model, front = Q.front_list(workers); t0 = time.time()
    acc = {}; cells = 0; rec = []; redo = []
    with mp.get_context("fork").Pool(workers, P._init, (model, Q.K, True)) as pool:
        for p, (n, (hists, Rs, Is), o, s, sec) in zip(pos, pool.imap(P._dfs, [front[i] for i in pos], chunksize=1)):
            cm = gmass.grouped(Rs, Is, len(hists), device) if len(Rs) else np.zeros(len(hists))
            cells += n; r = float(roots[p]); short = r - float(cm.sum())
            if short > TOL_ABS + TOL_REL * r:
                redo.append(p)
            else:
                for h, mv in zip(hists, cm):
                    acc[h] = acc.get(h, 0.0) + float(mv)
            rec.append([p, r, float(cm.sum()), float(cm.sum()), 0, 0])
        print("pass 1: %d subtrees, %d short, %.0fs" % (len(pos), len(redo), time.time() - t0), flush=True)
        idx = {row[0]: row for row in rec}
        for p, (hists, Rs, Is, fail) in zip(redo, pool.imap(fixed, [front[i] for i in redo], chunksize=1)):
            cm = gmass.grouped(Rs, Is, len(hists), device)
            for h, mv in zip(hists, cm):
                acc[h] = acc.get(h, 0.0) + float(mv)
            idx[p][3] = float(cm.sum()); idx[p][4] = 1; idx[p][5] = fail
            print("fixed pos %d: root %.12g estT %.12g fixed %.12g (qhull fails %d)" % (p, idx[p][1], idx[p][2], idx[p][3], fail), flush=True)
    rec = np.array(rec); hs = list(acc); ka, kb = Q.keys(hs); leaf = float(rec[:, 3].sum())
    out = os.path.join(Q.OUT, "remass_" + os.path.basename(f))
    np.savez(out + ".tmp.npz", key_a=ka, key_b=kb, mass=np.array([acc[h] for h in hs]), positions=np.array(pos), cells=cells,
             leaf=leaf, sub_pos=rec[:, 0].astype(np.int64), sub_root=rec[:, 1], sub_leaf_estT=rec[:, 2], sub_leaf_fixed=rec[:, 3],
             sub_fixed=rec[:, 4].astype(bool), sub_qhull_fail=rec[:, 5].astype(np.int64))
    os.replace(out + ".tmp.npz", out)
    print("done %s: root %.12g leaf estT %.12g leaf fixed %.12g, %.0fs" % (os.path.basename(f), rec[:, 1].sum(), rec[:, 2].sum(),
          leaf, time.time() - t0), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 11, sys.argv[3] if len(sys.argv) > 3 else "cuda")
