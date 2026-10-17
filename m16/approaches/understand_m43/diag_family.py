"""Diagnostic (not an estimator; uses exact enumeration as reference): exact error mass in each of est_mech's
family half-chambers vs est_mech's modelled error, averaged over noisy models. Shows which family is mis-modelled.
usage: python diag_family.py [n_models] [seed]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
from collections import defaultdict
from m43 import clip, frac, Wi0, Wh0, Wo0, noisy_models
from exact import _split
from est_mech import HALVES, est_mech, E


def exact_by_half(model):
    Wi, Wh, Wo = model
    out = defaultdict(float)
    for (sigma, (i, j, k), side, truth, poly) in HALVES:
        key = (int(sigma), f"c{i}{j}{'+' if side == 1 else '-'}")
        g1 = ((Wi * sigma) > 0).astype(float)
        A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1])
        for p2, g2 in _split(poly, A2, None):
            A3 = Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2])
            for p3, g3 in _split(p2, A3, None):
                tot = frac(p3)
                if g3.sum() == 0:
                    out[key] += 0.0 if truth == 0 else tot
                    continue
                L = Wo @ (g3[:, None] * A3)
                q = p3
                for o in range(3):
                    if o != truth:
                        q, _ = clip(q, L[truth] - L[o])
                        if not q:
                            break
                out[key] += tot - (frac(q) if q else 0.0)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    models = [(Wi0, Wh0, Wo0)] if n == 0 else noisy_models(256, 0.05, seed=sd)[:n]
    ex = defaultdict(list); es = defaultdict(list)
    for m in models:
        d = exact_by_half(m); inf = {}; est_mech(m, info=inf)
        for key in d:
            ex[key].append(d[key]); es[key].append(inf["err_by"].get(key, 0.0))
    print(f"{'family':16s} {'exact':>8s} {'model':>8s} {'bias':>8s} {'rms':>8s}")
    tot_e = 0; tot_m = 0
    for key in sorted(ex):
        a, b = np.array(ex[key]), np.array(es[key]); tot_e += a; tot_m += b
        print(f"{str(key):16s} {a.mean():8.4f} {b.mean():8.4f} {(b-a).mean():+8.4f} {np.sqrt(np.mean((b-a)**2)):8.4f}")
    print(f"{'total':16s} {tot_e.mean():8.4f} {tot_m.mean():8.4f} {(tot_m-tot_e).mean():+8.4f} {np.sqrt(np.mean((tot_m-tot_e)**2)):8.4f}")
