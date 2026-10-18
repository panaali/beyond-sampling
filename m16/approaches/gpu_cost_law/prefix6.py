"""T270: exact Gaussian masses of ALL k = 6 prefix regions of M_16,10, sharded across VMs, resumable.

Same enumeration as prefix.py (simplex fan cut by x0 = 0, every cutting gate resolved, region = gate history) and the
same masses (gmass: estT formulas, sliver bisection, d = 6 via orth6). The subtree list (breadth-first front) is
processed in the seeded random order of prefix.py; shard s of S takes positions i = s (mod S). Every CHUNK subtrees
the shard writes results/prefix6/shard{s}_chunk{c}.npz with, merged within the chunk: pattern keys (6 steps x 2 bytes =
uint64 key_a (bytes 0-7) + uint32 key_b (bytes 8-11)), masses, the subtree positions done, cells, and leaf-mass sum.
At start each shard writes results/prefix6/roots.npz: the exact mass of every subtree root cell, so the mass of the
unprocessed subtrees is known exactly (early-stop bounds, prefix6_bounds.py).
usage: python prefix6.py SHARD NSHARDS [workers] [chunk]"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prefix as P  # noqa: E402
import gmass  # noqa: E402
import estG  # noqa: E402
import estT_base as B  # noqa: E402
from m43 import Ops  # noqa: E402

K = int(os.environ.get("T270_K", "6"))     # prefix length (6 for T270; smaller only for tests)
OUT = os.path.join(HERE, "results", "prefix6")


def front_list(workers):
    """identical to prefix.run's front + permutation (k = 6)"""
    model = P.target(); P._init(model, K, True)
    ops = Ops(); front = [(c, g, A, 1, np.packbits(g > 0).tobytes()) for c, g, A in P.start_cells(K)]
    while front and front[0][3] < K and len(front) < 64 * workers:
        nxt = []
        for c, g, A, t, hist in front:
            for c2, g2, A2 in P._children(c, g, A, t, ops):
                nxt.append((c2, g2, A2, t + 1, hist + np.packbits(g2 > 0).tobytes()))
        front = nxt
    perm = np.random.default_rng(0).permutation(len(front))
    return model, [front[i] for i in perm]


def root_masses(front, device):
    col = P._Col(); estG.BUF[0] = col
    for i, (c, g, A, t, hist) in enumerate(front):
        col.cur = i; B.mass(c, Ops())
    R = np.array(col.R).reshape(-1, K, K); I = np.array(col.I, dtype=np.int64)
    return gmass.grouped(R, I, len(front), device)


def keys(hists):
    hists = [h.ljust(12, b"\0") for h in hists]                       # k < 6 (tests): pad to 12 bytes
    a = np.frombuffer(b"".join(h[:8] for h in hists), dtype=np.uint64)
    b = np.frombuffer(b"".join(h[8:12] for h in hists), dtype=np.uint32)
    return a, b


def main(shard, nshards, workers=11, chunk=40, device="cuda"):
    os.makedirs(OUT, exist_ok=True)
    model, front = front_list(workers)
    rf = os.path.join(OUT, "roots.npz")
    if not os.path.exists(rf):
        rm = root_masses(front, device)
        np.savez(rf, mass=rm, n=len(front))
        print("roots", len(front), "sum %.12f" % rm.sum(), flush=True)
    done = set()
    for f in os.listdir(OUT):
        if f.startswith(f"shard{shard}_chunk"):
            done |= set(np.load(os.path.join(OUT, f))["positions"].tolist())
    todo = [i for i in range(len(front)) if i % nshards == shard and i not in done]
    nchunk = sum(1 for f in os.listdir(OUT) if f.startswith(f"shard{shard}_chunk"))
    print("shard", shard, "subtrees", len(todo), "to do,", len(done), "done", flush=True)
    t0 = time.time()
    acc = {}; pos = []; cells = 0; leaf = 0.0
    with mp.get_context("fork").Pool(workers, P._init, (model, K, True)) as pool:
        for j, (n, p, o, s, sec) in enumerate(pool.imap(P._dfs, [front[i] for i in todo], chunksize=1)):
            hists, Rs, Is = p
            cm = gmass.grouped(Rs, Is, len(hists), device) if len(Rs) else np.zeros(len(hists))
            for h, mv in zip(hists, cm):
                acc[h] = acc.get(h, 0.0) + float(mv)
            cells += n; leaf += float(cm.sum()); pos.append(todo[j])
            if len(pos) == chunk or j == len(todo) - 1:
                hs = list(acc); ka, kb = keys(hs)
                f = os.path.join(OUT, f"shard{shard}_chunk{nchunk:04d}.npz")
                np.savez(f + ".tmp.npz", key_a=ka, key_b=kb, mass=np.array([acc[h] for h in hs]),
                         positions=np.array(pos), cells=cells, leaf=leaf)
                os.replace(f + ".tmp.npz", f)
                print("shard %d chunk %d: %d subtrees done (%d/%d), %.0fs" % (shard, nchunk, len(pos), j + 1, len(todo),
                      time.time() - t0), flush=True)
                nchunk += 1; acc = {}; pos = []; cells = 0; leaf = 0.0


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 11,
         int(sys.argv[4]) if len(sys.argv) > 4 else 40)
