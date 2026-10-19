"""T398b: finish the last step(s) of a triple from a saved survivor file, streaming (bnb.py's k = 10 step ran out of
memory holding all children). Each parent cell is rebuilt exactly from its gates (ordering cone + every gate's sign
row), split exactly (cone_count.split_fast), and its children are tested with relax2.prune_stage at the next k.
usage: python k10_stream.py p1,p2,p3 KFROM SHARD NSHARDS [--delta D]
-> certs/{surv,cert}_d<D>_<tri>_k<KFROM+1>_s<SHARD>.npz, logs/k10_<tri>_s<SHARD>.log"""
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog

import cone_count as cc
import relax2
from bnb import gates_of, step

HERE = os.path.dirname(os.path.abspath(__file__))


def cell_from_gates(tri, gates, k, Wi, Wh):
    G = [r for r in cc.cone(tri)[0]]; A = np.zeros((16, 10))
    for t in range(k):
        Z = Wh @ A; Z[:, t] += Wi; g = gates[t].astype(float)
        G += [(2 * g[j] - 1) * Z[j] for j in range(16) if np.linalg.norm(Z[j]) > 0]
        A = g[:, None] * Z
    G = np.array(G); nr = np.linalg.norm(G, axis=1)
    r = linprog(np.r_[np.zeros(10), -1], A_ub=np.c_[-G, nr], b_ub=np.zeros(len(G)),
                bounds=[(-1, 1)] * 10 + [(0, 1)], method="highs")
    hist = b"".join(np.packbits(gates[t] > 0).tobytes() for t in range(k))
    return (G, None, r.x[:10], A, hist), r.x[10]


def main():
    tri = tuple(int(v) for v in sys.argv[1].split(",")); kf = int(sys.argv[2]); sh, ns = int(sys.argv[3]), int(sys.argv[4])
    if "--delta" in sys.argv:
        relax2.DELTA = float(sys.argv[sys.argv.index("--delta") + 1])
    tag = f"d{relax2.DELTA:.3g}_{'-'.join(map(str, tri))}_k{kf + 1}_s{sh}"
    Wi, Wh, _ = cc.model()
    S = np.load(os.path.join(HERE, "certs", f"surv_d{relax2.DELTA:.3g}_{'-'.join(map(str, tri))}_k{kf}.npz"))
    par = S["gates"][sh::ns]
    surv, cst, cstart, cleaves = [], [], [], []; nch = 0; bad = 0; t0 = time.time()
    log = open(os.path.join(HERE, "logs", f"k10_{tag}.log"), "w")
    for i, gt in enumerate(par):
        cell, rad = cell_from_gates(tri, gt, kf, Wi, Wh)
        if rad <= 1e-12:
            bad += 1; continue
        for c in step([cell], kf, Wi, Wh, 16):
            nch += 1; lv = []
            sg = relax2.prune_stage(c[0], c[3], kf + 1, tri, cert=lv)
            if sg < 0:
                surv.append(gates_of(c[4], kf + 1, {}))
            else:
                cst.append(sg); cstart.append(len(cleaves)); cleaves.extend(gates_of(c[4], kf + 1, f) for f in lv)
        if i % 100 == 0:
            print(i, len(par), nch, len(surv), bad, round(time.time() - t0), file=log, flush=True)
    np.savez_compressed(os.path.join(HERE, "certs", f"surv_{tag}.npz"), tri=np.array(tri), k=kf + 1,
                        gates=np.array(surv, np.int8).reshape(-1, 10, 16), delta=relax2.DELTA)
    np.savez_compressed(os.path.join(HERE, "certs", f"cert_{tag}.npz"), tri=np.array(tri), k=kf + 1,
                        stage=np.array(cst, np.int8), start=np.array(cstart), leaves=np.array(cleaves, np.int8).reshape(-1, 10, 16),
                        delta=relax2.DELTA)
    print("done", len(par), "parents", nch, "children", len(surv), "survivors", bad, "degenerate parents",
          round(time.time() - t0), "s", file=log, flush=True)


if __name__ == "__main__":
    main()
