"""T280: double-description (DD) enumeration of the activation cells of a bias-free ReLU RNN on R^k (no LP).

A cell is a pointed polyhedral cone with constraint rows G (G x >= 0), unit extreme rays V and the ray/constraint
incidence ACT. A linear function c.x has one sign on the cell iff it has that sign on every ray. Otherwise the cell is
split by c.x = 0: rays on each side are kept, and every ADJACENT (+, -) pair (combinatorial test: no third ray's zero
set contains the pair's common zero set) gives a new ray on the hyperplane, w = s_p v_n - s_n v_p.
usage: python dd.py prefix MODEL K | final H T SEED   -> data/cells_*.pkl (rays + gate word per cell)
"""
import itertools
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-11


def split(G, V, ACT, c):
    """Returns ('on' | 'off', None) if c.x has one sign on the cell, else ('split', (plus_cell, minus_cell))."""
    s = V @ c
    tol = TOL * (np.abs(c).sum() + 1e-300)
    P, N = s > tol, s < -tol
    if not N.any():
        return ("on" if P.any() else "off"), None
    if not P.any():
        return "off", None
    Z = ~(P | N)
    k = V.shape[1]
    newV, newA = [], []
    pi, ni = np.flatnonzero(P), np.flatnonzero(N)
    for p in pi:
        for n in ni:
            common = ACT[p] & ACT[n]
            if common.sum() < k - 2:
                continue
            if (ACT[:, common].all(1)).sum() > 2:          # another ray lies on the common face: not adjacent
                continue
            w = s[p] * V[n] - s[n] * V[p]
            newV.append(w / np.linalg.norm(w)); newA.append(common)
    newV = np.array(newV).reshape(-1, k); newA = np.array(newA, bool).reshape(-1, ACT.shape[1])
    out = []
    for side, keep in ((1.0, P | Z), (-1.0, N | Z)):
        Vs = np.vstack([V[keep], newV])
        As = np.vstack([ACT[keep], newA])
        inc = np.concatenate([Z[keep], np.ones(len(newV), bool)])  # incidence with the new constraint
        out.append((np.vstack([G, side * c]), Vs, np.column_stack([As, inc])))
    return "split", tuple(out)


def orthant_cells(k):
    out = []
    for sg in itertools.product((1.0, -1.0), repeat=k):
        G = np.diag(sg); V = np.diag(sg)
        out.append((G, V, np.ones((k, k), bool) & ~np.eye(k, dtype=bool)))
    return out


def refine(cells, rows):
    """Split every cell by every row (used for the order planes of the final-leaf chamber start)."""
    for c in rows:
        nxt = []
        for G, V, A in cells:
            st, ch = split(G, V, A, c)
            nxt += list(ch) if st == "split" else [(G, V, A)]
        cells = nxt
    return cells


def enumerate_cells(model, k, start):
    Wi, Wh, _ = model; H = len(Wi)
    cells = [(G, V, A, np.zeros((H, k)), b"") for G, V, A in start]
    for t in range(k):
        new = []
        for G, V, A, M, hist in cells:
            Z = Wh @ M; Z[:, t] += Wi
            pieces = [(G, V, A, [])]
            for j in range(H):
                nxt = []
                for Gp, Vp, Ap, g in pieces:
                    st, ch = split(Gp, Vp, Ap, Z[j])
                    if st == "split":
                        nxt.append(ch[0] + (g + [1],)); nxt.append(ch[1] + (g + [0],))
                    else:
                        nxt.append((Gp, Vp, Ap, g + [1 if st == "on" else 0]))
                pieces = nxt
            for Gp, Vp, Ap, g in pieces:
                ga = np.array(g, float)
                new.append((Gp, Vp, Ap, ga[:, None] * Z, hist + np.packbits(ga > 0).tobytes()))
        cells = new
    return cells


def get_model(name):
    import models as Mo
    return Mo.get(name)


def main():
    t0 = time.time()
    if sys.argv[1] == "prefix":
        name, k = sys.argv[2], int(sys.argv[3]); model = get_model(name); start = orthant_cells(k); tag = f"prefix_{name}_k{k}"
    else:
        H, T, sd = (int(v) for v in sys.argv[2:5]); name = f"zoo{H}_{T}_{sd}"; model = get_model(name); k = T
        rows = [np.eye(k)[i] - np.eye(k)[j] for i, j in itertools.combinations(range(k), 2)]
        start = refine(orthant_cells(k), rows); tag = f"final_{name}"
    cells = enumerate_cells(model, k, start)
    regions = len({c[4] for c in cells})
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    pickle.dump([(c[1], c[4]) for c in cells], open(os.path.join(HERE, "data", f"cells_{tag}.pkl"), "wb"))
    print(f'{{"tag": "{tag}", "cells": {len(cells)}, "regions": {regions}, "start_cells": {len(start)}, '
          f'"sec": {time.time() - t0:.1f}}}', flush=True)


if __name__ == "__main__":
    main()
