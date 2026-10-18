"""T239: dump exact leaves with (error mass, leaf mass, center). final: estT exact mode (chamber start) on zoo(H,T,0);
prefix: width16_probe/prefix.py leaves of a T232 model at k (no labels; error = nan).
usage: python leaves.py final H T | python leaves.py prefix MODEL k   -> out/<tag>.npz"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "..", "width16_probe")
sys.path[:0] = [W, os.path.join(HERE, "..", "random_control"), os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from m43 import Ops


def center(R):
    u = (R / np.linalg.norm(R, axis=1, keepdims=True)).mean(0); return u / np.linalg.norm(u)


os.makedirs(os.path.join(HERE, "out"), exist_ok=True); rows = []
if sys.argv[1] == "final":
    from zoo44 import zoo
    H, T = int(sys.argv[2]), int(sys.argv[3]); tag = f"final_H{H}_T{T}"; ce0 = estT._cell_error

    def ce(c, g, A, Wo, truth, ops):
        e = ce0(c, g, A, Wo, truth, ops); rows.append((e, estT.mass(c, Ops()), *center(c.R))); return e
    estT._cell_error = ce
    acc = estT.estT(zoo(H, T, 0), Ops(), mode="exact")
else:
    from rc import model
    from prefix import run
    name, k = sys.argv[2], int(sys.argv[3]); tag = f"prefix_{name}_k{k}"
    _, leaves = run(model(name), k, save=True)
    for c, hist in leaves:
        rows.append((np.nan, estT.mass(c, Ops()), *center(c.R)))
A = np.array(rows); np.savez_compressed(os.path.join(HERE, "out", tag + ".npz"), err=A[:, 0], mass=A[:, 1], u=A[:, 2:])
print(tag, len(A), "mass sum - 1 = %.1e" % (A[:, 1].sum() - 1), flush=True)
