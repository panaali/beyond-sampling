"""T75 diagnostic (exact enumeration used only to describe where errors are, per seed): error mass by number of
active step-3 neurons, and by whether the error leaf touches a true boundary plane. usage: python diag75.py [n_noisy]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
from collections import Counter
from m43 import PERMS, clip, frac, order_normals, simplicial
from exact import _split
from zoo43 import zoo_model, noisy_copies
E = np.eye(3)


def describe(model):
    Wi, Wh, Wo = model
    by_n = Counter(); touch = Counter(); cross = []
    for perm in PERMS:
        c = perm[1]; on = order_normals(perm)
        for s in (1.0, -1.0):
            poly = simplicial(np.array(on + [s * E[0]]))
            g1 = ((Wi * s) > 0).astype(float)
            A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1])
            for p2, g2 in _split(poly, A2, None):
                A3 = Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2])
                cross.append(sum(1 for k in range(4) if min(A3[k] @ v for v in p2) < 0 < max(A3[k] @ v for v in p2)))
                for p3, g3 in _split(p2, A3, None):
                    n = int(g3.sum())
                    if n == 0:
                        w = [] if c == 0 else [p3]
                    else:
                        L = Wo @ (g3[:, None] * A3); w = []; q = p3
                        for o in range(3):
                            if o == c: continue
                            q, out = clip(q, L[c] - L[o])
                            if out: w.append(out)
                            if not q: break
                    for piece in w:
                        m = frac(piece); by_n[n] += m
                        t = any(abs(nn @ v) < 1e-9 for v in piece for nn in on)
                        touch[t] += m
    return by_n, touch, np.mean(cross)


if __name__ == "__main__":
    nn = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    for sd in range(5):
        for label, ms in (("clean", [zoo_model(sd)]), ("noisy", noisy_copies(zoo_model(sd), nn, 0.05, 0))):
            B = Counter(); Tt = Counter(); cr = []
            for m in ms:
                b, t, c = describe(m); B.update({k: v / len(ms) for k, v in b.items()}); Tt.update({k: v / len(ms) for k, v in t.items()}); cr.append(c)
            tot = sum(B.values())
            print(f"seed {sd} {label}: error {tot:.3f}; by #active neurons " +
                  ", ".join(f"{k}:{B[k]/tot:.0%}" for k in sorted(B)) + f"; touching a true boundary {Tt[True]/tot:.0%}; "
                  f"mean step-3 gates crossing a step-2 piece {np.mean(cr):.2f}")
