"""Diagnostic: exact vs modelled error in each strip piece of one family (exact = split the piece by all four
step-3 gates; used only to locate mis-modelled pieces). usage: python diag_piece.py noise_seed index sigma circle side"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
from m43 import noisy_models, Ops, clip, frac
from exact import _split
from est_mech import HALVES, _strip_pieces, E, _arc_err, _read_point


def piece_exact(piece, A3, Wo, truth):
    err = 0.0
    for p3, g3 in _split(piece, A3, None):
        tot = frac(p3)
        if g3.sum() == 0:
            err += 0.0 if truth == 0 else tot; continue
        L = Wo @ (g3[:, None] * A3); q = p3
        for o in range(3):
            if o != truth:
                q, _ = clip(q, L[truth] - L[o])
                if not q: break
        err += tot - (frac(q) if q else 0.0)
    return err


if __name__ == "__main__":
    sd, idx, sigma, circ, side = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    Wi, Wh, Wo = noisy_models(256, 0.05, sd)[idx]
    g1 = ((sigma * Wi) > 0).astype(float)
    A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1]); ts = np.sort(-A2[:, 0] / Wi)
    b = [-np.inf] + list(ts) + [np.inf]; A3s = []
    for s in range(5):
        r = (b[s] + b[s+1]) / 2 if np.isfinite(b[s]) and np.isfinite(b[s+1]) else (b[s+1] - 1 if np.isfinite(b[s+1]) else b[s] + 1)
        g2 = (sigma * (A2[:, 0] + Wi * r) > 0).astype(float); A3s.append(Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2]))
    for h in HALVES:
        if h["sigma"] == sigma and "".join(map(str, h["circle"][:2])) == circ and h["side"] == side:
            for s, p in _strip_pieces(h, ts, Ops()):
                ex = piece_exact(p, A3s[s], Wo, h["truth"])
                for res in (False, True):
                    md = _arc_err(p, h["n_true"], A3s[s], Wo, h["truth"], Ops(), res)
                    print(f"truth {h['truth']} strip {s} resolve {res}: exact {ex:.4f} model {md:.4f}")
