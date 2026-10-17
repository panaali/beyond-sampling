"""Diagnostic only: error mass by family (nearest true boundary to the error piece, side, chart, dead zone)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
from collections import defaultdict
from m43 import PERMS, clip, frac, order_normals, simplicial, Wi0, Wh0, Wo0, noisy_models
from exact import _split

def fam_decomp(model):
    Wi, Wh, Wo = model
    e = np.eye(3); out = defaultdict(float)
    for perm in PERMS:
        c = perm[1]
        for s in (1.0, -1.0):
            poly = simplicial(np.array(order_normals(perm) + [s * e[0]]))
            g1 = ((Wi * s) > 0).astype(float)
            A2 = Wh @ (g1[:, None] * np.outer(Wi, e[0])) + np.outer(Wi, e[1])
            for p2, g2 in _split(poly, A2, None):
                A3 = Wh @ (g2[:, None] * A2) + np.outer(Wi, e[2])
                for p3, g3 in _split(p2, A3, None):
                    if g3.sum() == 0:
                        if c != 0:
                            out[("dead", int(s))] += frac(p3)
                        continue
                    L = Wo @ (g3[:, None] * A3)
                    for w in range(3):
                        if w == c: continue
                        q = p3
                        for o in range(3):
                            if o == w: continue
                            q, _ = clip(q, L[w] - L[o])
                            if not q: break
                        if q:
                            cen = np.mean(q, 0)
                            d = {(i, j): abs(cen[i] - cen[j]) for i, j in ((0, 1), (0, 2), (1, 2))}
                            ij = min(d, key=d.get)
                            k = 3 - sum(ij)
                            side = "+" if cen[k] > max(cen[i] for i in ij) else "-"
                            out[(f"c{ij[0]}{ij[1]}{side}", int(s), "pred", w, "g3", tuple(g3.astype(int)))] += frac(q)
    return out

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    models = [(Wi0, Wh0, Wo0)] if n == 0 else noisy_models(256, 0.05, seed=0)[:n]
    agg = defaultdict(float); agg2 = defaultdict(float); tot = 0
    for m in models:
        d = fam_decomp(m)
        for k, v in d.items():
            agg[k[:2]] += v / len(models); agg2[k] += v / len(models); tot += v / len(models)
    print(f"mean error {tot:.4f}")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(" ", k, f"{v:.4f}")
    print("finer:")
    for k, v in sorted(agg2.items(), key=lambda kv: -kv[1])[:25]:
        print(" ", k, f"{v:.4f}")
