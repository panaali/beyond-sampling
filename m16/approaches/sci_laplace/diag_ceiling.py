"""CEILING TEST (diagnosis, random inputs): if, for each top-3 configuration, the band offset delta were
replaced by its best ADDITIVE approximation in the 7 rest inputs (oracle backfit on 800 samples), how far is
E[W(0,|delta_add|)] from E[W(0,|delta|)] (fresh samples)? The additive law is what a generating-function
(convolution) calculation could give exactly. Also: how well does W(0,|delta|) account for the exact F?"""
import numpy as np
from lp import *
from lp import _trunc_below
model = get_model("orig"); rng = np.random.default_rng(11)
ncfg, per, nb = 50, 800, 10
Wf = lambda x: W(0, x)
rows = []
for j in range(ncfg):
    combo, u, xc, rest = rand23(1, rng)
    C = np.repeat(combo, per, 0); U_ = np.repeat(u, per); X = np.repeat(xc, per)
    def draw():
        Uu = rng.random((per, 7)); r = _trunc_below(U_[:, None], Uu)
        f = build23(C, U_, X, r)
        dl = np.clip(delta_ab(model, f)[0], -f["hi"], f["hi"])
        e1, e2, _ = F_sides(model, f)
        return np.minimum((Uu * nb).astype(int), nb - 1), dl, e1 + e2, f
    B, dl, F, f = draw()
    comp = np.zeros((7, nb)); mu = dl.mean()
    for _ in range(40):
        for t in range(7):
            r = dl - mu - sum(comp[s][B[:, s]] for s in range(7) if s != t)
            comp[t] = [r[B[:, t] == k].mean() for k in range(nb)]
            comp[t] -= comp[t].mean()
    B2, dl2, F2, f2 = draw()
    add = mu + sum(comp[s][B2[:, s]] for s in range(7))
    # exact law of the additive model by brute-force enumeration of the convolution (nb^7 too many): use
    # the fresh sample's empirical law instead (same thing in expectation)
    true_w = Wf(np.abs(dl2)).mean(); add_w = Wf(np.abs(add)).mean()
    rows.append((true_w, add_w, F2.mean()))
r = np.array(rows)
print("sum over configs: E W|delta| %.5f  E W|delta_add| %.5f  E F %.5f" % tuple(r.sum(0) / ncfg))
print("relative error of additive (aggregate): %.3f ; per-config median abs rel err %.3f"
      % (r[:, 1].sum() / r[:, 0].sum() - 1, np.median(np.abs(r[:, 1] / r[:, 0] - 1))))
print("W|delta| vs F aggregate ratio %.3f; per-config corr %.3f" % (r[:, 0].sum() / r[:, 2].sum(), np.corrcoef(r[:, 0], r[:, 2])[0, 1]))
