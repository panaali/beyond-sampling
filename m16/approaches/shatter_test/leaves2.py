"""T239 follow-up (DRI): final exact leaves with rays and the predicted class at every ray (forward pass at the ray; the
network is continuous and linear on each leaf, so the leaf has one predicted class iff all its rays agree).
usage: python leaves2.py H T -> out/rays_H{H}_T{T}.npz (object arrays: rays, classes; err, mass, truth)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "..", "width16_probe")
sys.path[:0] = [W, os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo
H, T = int(sys.argv[1]), int(sys.argv[2]); m = zoo(H, T, 0); Wi, Wh, Wo = m; rows = []; ce0 = estT._cell_error


def classes(R):
    h = np.zeros((len(R), H))
    for t in range(T):
        h = np.maximum(h @ Wh.T + R[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1)


def ce(c, g, A, Wo_, truth, ops):
    e = ce0(c, g, A, Wo_, truth, ops); R = c.R / np.linalg.norm(c.R, axis=1, keepdims=True)
    rows.append((e, estT.mass(c, Ops()), truth, R, classes(R))); return e


estT._cell_error = ce; estT.estT(m, Ops(), mode="exact")
np.savez_compressed(os.path.join(HERE, "out", f"rays_H{H}_T{T}.npz"), err=np.array([r[0] for r in rows]), mass=np.array([r[1] for r in rows]),
                    truth=np.array([r[2] for r in rows]), rays=np.array([r[3] for r in rows], dtype=object), cls=np.array([r[4] for r in rows], dtype=object))
print(H, T, len(rows), flush=True)
