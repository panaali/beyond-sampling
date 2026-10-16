import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from bands import line_A, line_B, frac, point_out
M = (Wi, Wh, Wo)
x = np.random.default_rng(3).standard_normal((40000, 10))
out = point_out(M, x); tgt = np.argsort(x, 1)[:, -2]; r1 = np.argmax(x, 1)
w = out != tgt
LA, LB = line_A(x), line_B(x)
fA, inA, pA, _ = frac(M, LA); fB, inB, pB, _ = frac(M, LB)
print("wrong", w.mean())
print("A: fA", fA.mean(), " ind wrong&inA", (w & inA).mean(), " inA&~wrong", (inA & ~w).mean())
print("B: fB", fB.mean(), " ind wrong&inB", (w & inB).mean(), " inB&~wrong", (inB & ~w).mean())
print("overlap", (inA & inB).mean())
print("wrong & out!=r1", (w & (out != r1)).mean(), "wrong & out==r1", (w & (out == r1)).mean())
print("wrong&out!=r1&~inA", (w & (out != r1) & ~inA).mean())
