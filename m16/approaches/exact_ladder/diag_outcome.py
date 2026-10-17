"""Diagnostic (not charged, not part of the estimator): how many 'outcome cells' does the wrong set have?

Wrong set = union of the convex wrong pieces. Merge rule: two E14-style wrong cones (key = full gate pattern,
predicted index, true 2nd index) are merged if they touch across a facet. Probes are deterministic: for each wrong
piece we evaluate the model at points near each ray (0.9 ray + 0.1 centroid, so every key the piece spans is
probably seen) and just outside each facet (facet centroid + eps (facet centroid - piece centroid)). Union-find over
keys. Reports: wrong pieces, distinct wrong keys (cones), connected components of wrong keys (outcome cells).

usage: python diag_outcome.py T name
"""
import json
import sys

import numpy as np

sys.path.append("../..")
import leaderboard as L  # noqa: E402
import cones as E  # noqa: E402


def keys(model, X):
    Wi, Wh, Wo = model
    h = np.zeros((len(X), len(Wi))); gs = []
    for t in range(X.shape[1]):
        z = h @ Wh.T + X[:, t:t + 1] * Wi[None]; gs.append(z > 0); h = np.maximum(z, 0)
    k = (h @ Wo.T).argmax(1); s = np.argsort(X, 1)[:, -2]
    G = np.packbits(np.concatenate(gs, 1), 1)
    return [(G[i].tobytes(), int(k[i]), int(s[i])) for i in range(len(X))], k != s


def facets(R, Z):
    inc = np.unpackbits(Z.view(np.uint8), axis=1, bitorder="little").astype(bool)
    d = R.shape[1]; out = set()
    for c in np.nonzero(inc.any(0))[0]:
        G = tuple(np.nonzero(inc[:, c])[0])
        if len(G) >= d - 1 and G not in out and np.linalg.matrix_rank(R[list(G)], 1e-9) == d - 1:
            out.add(G)
    return out


def main(T, name, eps=1e-6):
    m = L.models(16, T)[name]
    ops = E.Ops(); counts = [0] * T
    par = {}

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a

    def union(a, b):
        par.setdefault(a, a); par.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb

    npieces = 0; nprobe = 0
    for c in E.iter_cells(m, ops, counts):
        _, bad = E.outcome_pieces(c, m, ops, 1 + 16 * (T - 1))
        for R, Z in bad:
            npieces += 1
            cen = R.mean(0); cen /= np.linalg.norm(cen)
            P_in = 0.9 * R + 0.1 * cen
            P_out = []
            for G in facets(R, Z):
                f = R[list(G)].mean(0); P_out.append(f + eps * (f - cen))
            X = np.concatenate([P_in, np.array(P_out).reshape(-1, T)])
            ks, wrong = keys(m, X); nprobe += len(X)
            kin = [k for k, w in zip(ks[:len(P_in)], wrong[:len(P_in)]) if w]
            for k in kin[1:]:
                union(kin[0], k)
            if not kin:
                continue
            for k, w in zip(ks[len(P_in):], wrong[len(P_in):]):
                if w:
                    union(kin[0], k)
    roots = {find(k) for k in par}
    out = dict(T=T, name=name, cells=counts[-1], wrong_pieces=npieces, wrong_keys=len(par),
               outcome_components=len(roots), probes=nprobe)
    print(json.dumps(out), flush=True)
    with open("outcome.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
