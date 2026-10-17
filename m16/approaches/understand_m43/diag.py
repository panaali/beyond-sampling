"""Diagnostic only (not an estimator): where is the error mass? Uses the exact enumerator's region split."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
from collections import defaultdict
from m43 import PERMS, Ops, clip, frac, order_normals, simplicial, Wi0, Wh0, Wo0, noisy_models
from exact import _split

def err_decomp(model):
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
                    L = Wo @ (g3[:, None] * A3)
                    for w in range(3):
                        if w == c: continue
                        q = p3
                        for o in range(3):
                            if o == w: continue
                            n = L[w] - L[o] if g3.sum() > 0 else (np.zeros(3) if w == 0 else -np.ones(3) * 0)
                            if g3.sum() == 0:
                                if w != 0: q = []; break
                                continue
                            q, _ = clip(q, n)
                            if not q: break
                        if q:
                            key = (perm, int(s), tuple(g2.astype(int)), tuple(g3.astype(int)), w)
                            out[key] += frac(q)
    return out

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    if which == "base":
        models = [(Wi0, Wh0, Wo0)]
    else:
        models = noisy_models(256, 0.05, seed=0)[:int(which)]
    for m in models:
        d = err_decomp(m)
        tot = sum(d.values())
        print(f"total error {tot:.4f}")
        agg = defaultdict(float)
        for (perm, s, g2, g3, w), v in d.items():
            agg[(perm, s, w)] += v
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:12]:
            print("  perm(desc)", k[0], "sx0", k[1], "pred", k[2], f"{v:.4f}")
        for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:10]:
            print("    ", k, f"{v:.4f}")
