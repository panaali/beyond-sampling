"""Diagnostic: is the correctness margin additive in the low inputs (rank>=4), given top-3 values+positions?
Cut-HDMR test with anchor a, target b (both draws of the low inputs, same top-3):
  m_add(b) = m(a) + sum_j [m(a with b_j) - m(a)]."""
import sys; sys.path.insert(0, "/home/user/clean-neurons/m16")
import numpy as np
from arc_eval import forward, Wi0, Wh0, Wo0
from scipy.special import ndtr, ndtri
rng = np.random.default_rng(3); M = (Wi0, Wh0, Wo0)

def margin(x, tgt):
    L = forward(M, x); lt = L[np.arange(len(x)), tgt].copy(); L[np.arange(len(x)), tgt] = -np.inf
    return lt - L.max(1)

N = 40_000
x = rng.standard_normal((N, 10)); o = np.argsort(-x, 1)
g = (np.take_along_axis(x, o[:, 1:2], 1) - np.take_along_axis(x, o[:, 2:3], 1))[:, 0]
sel = g < 0.15                         # near 2-3 ties (where most errors are)
x, o = x[sel], o[sel]; n = len(x); tgt = o[:, 1]
m3 = np.take_along_axis(x, o[:, 2:3], 1)
def redraw(x):
    y = x.copy(); low = o[:, 3:]
    u = rng.random(low.shape) * ndtr(m3); np.put_along_axis(y, low, ndtri(u), 1); return y
a = redraw(x); b = redraw(x)
ma, mb = margin(a, tgt), margin(b, tgt)
madd = ma.copy()
for r in range(3, 10):
    pos = o[:, r]; y = a.copy(); y[np.arange(n), pos] = b[np.arange(n), pos]
    madd += margin(y, tgt) - ma
print("n", n, "err rate a", np.mean(ma < 0), "b", np.mean(mb < 0))
print("sign(a)!=sign(b):", np.mean((ma < 0) != (mb < 0)))
print("sign(add)!=sign(b):", np.mean((madd < 0) != (mb < 0)), " pred err rate", np.mean(madd < 0))
d = mb - ma; e = mb - madd
print("var(mb-ma)", d.var(), "var(mb-madd)", e.var(), "R2 additive of change", 1 - e.var() / d.var())
print("median |m|", np.median(np.abs(mb)))
