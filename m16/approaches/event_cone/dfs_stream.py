"""Sharded depth-first branch and bound from a saved survivor file (T398b setting): each shard takes parents
gates[shard::nshards] of certs/surv_d<D>_<tri>_k<KFROM>.npz, rebuilds each cell exactly (k10_stream.cell_from_gates)
and expands its subtree to KMAX depth-first with relax2.prune_stage, so memory stays small.
Outputs (per shard): certs/surv_<tag>.npz (k = KMAX survivors' gates), certs/cert_<tag>.npz (all pruned cells,
T400 format v1 with a per-cell k array), logs/dfs_<tag>.log (per-k children/survivor counts, progress).
usage: python dfs_stream.py p1,p2,p3 KFROM KMAX SHARD NSHARDS [--delta D] [--parents i,j,...]"""
import json
import os
import sys
import time

import numpy as np

import cone_count as cc
import relax2
from bnb import gates_of, step
from k10_stream import cell_from_gates

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    tri = tuple(int(v) for v in sys.argv[1].split(",")); kf, km = int(sys.argv[2]), int(sys.argv[3])
    sh, ns = int(sys.argv[4]), int(sys.argv[5])
    if "--delta" in sys.argv:
        relax2.DELTA = float(sys.argv[sys.argv.index("--delta") + 1])
    ts = "-".join(map(str, tri)); tag = f"dfs_d{relax2.DELTA:.3g}_{ts}_k{kf}to{km}_s{sh}of{ns}"
    Wi, Wh, _ = cc.model()
    allg = np.load(os.path.join(HERE, "certs", f"surv_d{relax2.DELTA:.3g}_{ts}_k{kf}.npz"))["gates"]
    idx = list(range(sh, len(allg), ns))
    if "--parents" in sys.argv:
        idx = [int(v) for v in sys.argv[sys.argv.index("--parents") + 1].split(",")]
    ch = {k: 0 for k in range(kf + 1, km + 1)}; sv = {k: 0 for k in range(kf + 1, km + 1)}
    surv, ck, cst, cstart, cleaves = [], [], [], [], []
    log = open(os.path.join(HERE, "logs", f"{tag}.log"), "w"); t0 = time.time()

    def dfs(cell, k):
        for c in step([cell], k, Wi, Wh, 16):
            ch[k + 1] += 1; lv = []
            sg = relax2.prune_stage(c[0], c[3], k + 1, tri, cert=lv)
            if sg >= 0:
                ck.append(k + 1); cst.append(sg); cstart.append(len(cleaves))
                cleaves.extend(gates_of(c[4], k + 1, f) for f in lv)
                continue
            sv[k + 1] += 1
            if k + 1 == km:
                surv.append(gates_of(c[4], km, {}))
            else:
                dfs(c, k + 1)

    for n, i in enumerate(idx):
        cell, rad = cell_from_gates(tri, allg[i], kf, Wi, Wh)
        if rad > 1e-12:
            dfs(cell, kf)
        print(json.dumps(dict(done=n + 1, of=len(idx), parent=i, children=ch, survivors=sv,
                              sec=round(time.time() - t0))), file=log, flush=True)
    np.savez_compressed(os.path.join(HERE, "certs", f"surv_{tag}.npz"), tri=np.array(tri), k=km,
                        gates=np.array(surv, np.int8).reshape(-1, 10, 16), delta=relax2.DELTA, parents=np.array(idx))
    np.savez_compressed(os.path.join(HERE, "certs", f"cert_{tag}.npz"), tri=np.array(tri), k=np.array(ck, np.int8),
                        stage=np.array(cst, np.int8), start=np.array(cstart), leaves=np.array(cleaves, np.int8).reshape(-1, 10, 16),
                        delta=relax2.DELTA)
    print("DONE", file=log, flush=True)


if __name__ == "__main__":
    main()
