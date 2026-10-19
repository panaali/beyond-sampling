"""Per-model line diagnostics on 2,000 fixed tie-face segments (1,000 per face, Sobol seed 99, Gaussian window):
mean cells per line (fragmentation, M4 proxy), the walk's cost per line (FPE), and, for non-PL walks, the
line-integral error vs a reference numerical walk (G = 1024 nodes, 30 bisections).
usage: python lines.py TAG SEED  -> appends to results/lines.jsonl
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "frozen"))
import core as C  # noqa: E402
import load  # noqa: E402
import walkgen as G  # noqa: E402

if __name__ == "__main__":
    tag, seed = sys.argv[1], int(sys.argv[2])
    net = G.Net(*load.load(tag, seed)); r = {"tag": tag, "seed": seed, "mode": net.mode}
    t0 = time.time()
    for k in ("23", "12"):
        y, a, b, tau = C.face_points(10, k, C.sobol(10, 1000, 99)); tau = np.minimum(tau, 8.5)
        m, c, n = G.walk(net, y, a, b, tau, k)
        r[f"mass_{k}"] = float(m.mean()); r[f"cost_{k}"] = float(c.mean())
        if net.mode != "num":
            r[f"cells_{k}"] = float(n.mean())
        if net.mode != 1:
            mr, cr, _ = G.walk_num(net, y, a, b, tau, k, G=1024, B=30)
            r[f"ref_absdiff_{k}"] = float(np.abs(m - mr).mean()); r[f"ref_meandiff_{k}"] = float((m - mr).mean())
            r[f"ref_mass_{k}"] = float(mr.mean())
    r["wall"] = time.time() - t0
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "lines.jsonl"), "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps(r))
