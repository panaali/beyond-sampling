"""DIAGNOSIS: which (time, neuron) gates flip inside the misclassified band [0, delta]? If the set is nearly
fixed per position triple, delta is a min/max of a few linear-fractional forms (a compact 'asymptotic' law)."""
import numpy as np
from collections import Counter
from lp import *
model = get_model("orig"); rng = np.random.default_rng(19)
Wi, Wh, Wo = model
def gates(x):
    h = np.zeros((len(x), 16)); G = np.zeros((len(x), T, 16), bool)
    for t in range(T):
        p = h @ Wh.T + x[:, t:t + 1] * Wi[None]; G[:, t] = p > 0; h = p * G[:, t]
    return G
ent, nfl, top = [], [], []
for j in rng.choice(360, 12, replace=False):
    n = 300
    combo, u, xc, rest = rand23(n, rng, np.repeat(COMBO[j][None], n, 0))
    f = build23(combo, u, xc, rest)
    dl, D0, _ = delta_ab(model, f)
    ok = np.abs(dl) < f["hi"] - 1e-9
    x0 = f["x0"]; d = f["d"]
    sg = np.sign(dl)
    g0 = gates(x0 + (1e-9 * sg)[:, None] * d); g1 = gates(x0 + (dl * (1 - 1e-6))[:, None] * d)
    fl = (g0 != g1)
    sets = [tuple(map(tuple, np.argwhere(fl[k]))) for k in range(n) if ok[k]]
    c = Counter(sets); p = np.array(list(c.values())) / len(sets)
    ent.append(-(p * np.log2(p)).sum()); nfl.append(fl[ok].sum((1, 2)).mean())
    pair = Counter(tuple(x) for k in range(n) if ok[k] for x in np.argwhere(fl[k]))
    print(COMBO[j], "mean #flips in band %.2f  distinct flip-sets %d / %d  entropy %.2f bits  top flips %s"
          % (nfl[-1], len(c), len(sets), ent[-1], [(k, round(v / len(sets), 2)) for k, v in pair.most_common(5)]), flush=True)
