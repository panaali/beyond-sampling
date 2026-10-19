"""Part (a), secondary and PROBABILISTIC (sampling-class; never merged with the deterministic bounds).
Plain MC of P(E(t0)) over all 720 triples: x ~ N(0, I_10); E = {argmax of the step-10 logits = position of
x_(3), x_(2) - x_(3) >= t0}. Only draws with g23 >= t0 are run through the network. Prints hit counts; the
Clopper-Pearson upper bound at confidence 1 - alpha is computed by prob_combine.py over shards. Hits are saved
(x, gate pattern) so every event draw can be checked against the verified survivor sets (a soundness test of the
pruning: an event draw inside a pruned cell would refute a certificate).
usage: python prob_bound.py t0 n_draws seed -> data/prob_<t0>_<seed>.npz"""
import sys

import numpy as np

import common

t0 = float(sys.argv[1]); n = int(float(sys.argv[2])); seed = int(sys.argv[3])
rng = np.random.default_rng(10_000 + seed)
B = 4_000_000; done = 0; kept = 0; hitx = []
while done < n:
    m = min(B, n - done)
    X = rng.standard_normal((m, 10))
    top = np.argpartition(-X, 3, axis=1)[:, :3]
    v = np.take_along_axis(X, top, 1)
    o = np.argsort(-v, 1); top = np.take_along_axis(top, o, 1); v = np.take_along_axis(v, o, 1)
    sel = v[:, 1] - v[:, 2] >= t0
    if sel.any():
        Xs = X[sel]; kept += len(Xs)
        hs = common.run(Xs)
        hit = (hs[:, -1] @ common.Wo.T).argmax(1) == top[sel, 2]
        hitx.append(Xs[hit])
    done += m
hx = np.concatenate(hitx) if hitx else np.zeros((0, 10))
np.savez(f"data/prob_{t0}_{seed}.npz", draws=n, kept=kept, hits=len(hx), x=hx)
print("t0", t0, "draws", n, "gap-kept", kept, "event hits", len(hx), flush=True)
