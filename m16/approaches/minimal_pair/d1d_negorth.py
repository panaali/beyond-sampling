"""D1d: errors inside the <=1-positive region. Scale-free description: relative gaps, positions, last input."""
import numpy as np
from collections import Counter
import core as C
for T in (4, 5):
  for s in (0, 2, 4):
    m = C.zoo(16, T, s); rng = np.random.default_rng(77 + s + 10 * T)
    x = -np.abs(rng.standard_normal((1_000_000, T)))         # uniform over the negative orthant
    pred = C.forward(m, x).argmax(1)
    order = np.argsort(-x, 1); rk = np.argsort(order, 1)[np.arange(len(x)), pred] + 1
    xs = -np.sort(-x, 1); bad = rk != 2
    rel = np.where(rk == 3, (xs[:, 1] - xs[:, 2]) / np.abs(xs[:, 1]), (xs[:, 0] - xs[:, 1]) / np.abs(xs[:, 1]))
    print(f"T{T} s{s}: P(err | all neg) {bad.mean():.4f}; pred-rank shares {dict(Counter(rk[bad]))}")
    print("   rel-gap quantiles of errors (q50,q90,q99):", np.quantile(rel[bad], [.5, .9, .99]).round(3))
    # conditional error by position of rank-2 element and rank-1 element
    p2 = order[:, 1]; p1 = order[:, 0]
    tab = np.array([[bad[(p1 == i) & (p2 == j)].mean() if i != j else np.nan for j in range(T)] for i in range(T)])
    print("   P(err | pos r1=row, pos r2=col):\n", np.round(tab, 3))
    # error vs the ratio x(1)/x(2) (scale-free): how close the top two are relative to scale
    q = xs[:, 0] / xs[:, 1]    # in (0,1]: near 1 = top two tied, near 0 = top near zero
    bins = np.array([0, .1, .25, .5, .75, .9, 1.0])
    ib = np.digitize(q, bins) - 1
    print("   P(err | x1/x2 bin)", [f"{bins[i]}-{bins[i+1]}:{bad[ib==i].mean():.3f}" for i in range(len(bins) - 1)])
    q3 = xs[:, 2] / xs[:, 1]   # >=1
    bins3 = np.array([1, 1.1, 1.25, 1.5, 2, 4, 1e9]); ib = np.digitize(q3, bins3) - 1
    print("   P(err | x3/x2 bin)", [f"{bins3[i]}-:{bad[ib==i].mean():.3f}" for i in range(len(bins3) - 1)], flush=True)
