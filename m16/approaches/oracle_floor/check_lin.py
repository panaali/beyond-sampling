"""Independent re-implementation of the lin closure (DIAGNOSTIC SAMPLING): exact local logit map of the
conditional-mean point's gate pattern (observer/core.logit_map) instead of diag_cond's finite differences, own
skeleton/low-input sampler (sort of N(0,I), then low inputs by rejection-free inverse CDF), different seed.
Also reports where lin's extra errors come from: fraction of low draws whose true gate pattern differs from the
anchor's.  usage: python check_lin.py name [NS=2000] [M=256]"""
import os, sys
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm
from panel import ROOT, dev2, target, forward
sys.path.insert(0, os.path.join(ROOT, "m16/approaches/observer"))
from core import forward_gates, logit_map  # noqa: E402

name = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000; M = int(sys.argv[3]) if len(sys.argv) > 3 else 256
model = dict([("target", target())] + dev2())[name]
rng = np.random.default_rng(55)
x0 = rng.standard_normal((NS, 10)); o = np.argsort(-x0, 1); top = o[:, :3]; v = np.take_along_axis(x0, top, 1)
rest = np.sort(o[:, 3:], 1); p2 = top[:, 1]
d, dR0, same = [], [], []
for i in range(NS):
    low = ndtri(rng.uniform(size=(M, 7)) * ndtr(v[i, 2]))
    X = np.empty((M, 10)); X[:, top[i]] = v[i]; X[:, rest[i]] = low
    _, G = forward_gates(model, X)
    q = (forward(model, X).argmax(1) == p2[i]).mean()
    xb = np.empty(10); xb[top[i]] = v[i]; xb[rest[i]] = -norm.pdf(v[i, 2]) / norm.cdf(v[i, 2])
    _, Gb = forward_gates(model, xb[None])
    A = logit_map(model, Gb)[0]                        # logits = A x inside the anchor's cone (no biases)
    ql = ((X @ A.T).argmax(1) == p2[i]).mean()
    d.append(ql - q); dR0.append(float((A @ xb).argmax() == p2[i]) - q)
    same.append((G == Gb).all((1, 2)).mean())
d, dR0, same = map(np.array, (d, dR0, same))
nb = v[:, 1] > 0; tail = nb & (np.minimum(v[:, 0] - v[:, 1], v[:, 1] - v[:, 2]) >= .05)
print(name, "lin bias %+.3f +- %.3f pp (tail part %+.3f)  R0 %+.3f pp  | frac low draws in anchor's gate cone: mean %.3f (tail %.3f)" % (
    100 * d.mean(), 100 * d.std() / np.sqrt(NS), 100 * d[tail].sum() / NS, 100 * dR0.mean(), same.mean(), same[tail].mean()))
