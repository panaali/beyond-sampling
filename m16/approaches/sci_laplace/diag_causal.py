"""DIAGNOSIS (random inputs). Does the 2-3 band mass F at a tie a<b depend on inputs AFTER the later tie
element b (later corrections) or mostly on inputs before b (the decision is taken when b arrives)?
For each base face point, resample (i) the rest inputs before b, (ii) the rest inputs after b, keeping
everything else fixed; report E[Var(F|fixed)] / Var(F)."""
import sys
import numpy as np
from lp import *
from lp import _trunc_below, _trunc_above, _inv

name = sys.argv[1] if len(sys.argv) > 1 else "orig"
model = get_model(name)
rng = np.random.default_rng(1)
N, R = 1500, 6
combo, u, xc, rest = rand23(N, rng)
oth = np.array([[i for i in range(T) if i not in cb] for cb in combo])     # (N,7) positions of rest
b = combo[:, 1]
F0 = sum(F_sides(model, build23(combo, u, xc, rest))[:2])
res = {}
for lab, mask in (("before_b", oth < b[:, None]), ("after_b", oth > b[:, None])):
    Fs = []
    for r in range(R):
        new = _trunc_below(u[:, None], rng.random((N, 7)))
        rr = np.where(mask, new, rest)
        Fs.append(sum(F_sides(model, build23(combo, u, xc, rr))[:2]))
    Fs = np.array(Fs)
    res[lab] = Fs.var(0, ddof=1).mean()
print(name, "mean F", F0.mean(), "var F", F0.var(), {k: v / F0.var() for k, v in res.items()})
# also by whether c is after b
for cond, lab in ((combo[:, 2] > b, "c after b"), (combo[:, 2] < b, "c before b")):
    print(lab, "share of F", F0[cond].sum() / F0.sum(), "frac", cond.mean())
