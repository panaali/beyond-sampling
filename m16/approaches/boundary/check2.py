import sys, time
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from tie23 import segments_23
from local import local_band, remainder_sample
from seg import walk
M = (Wi, Wh, Wo)
rng = np.random.default_rng(1)
x = rng.standard_normal((20000, 10))
x0, d, lo, hi, tn, tp, yb = segments_23(x)
t = time.time()
band, sr, sl, p = local_band(M, x0, d, hi, -lo, tp, tn)
rem, p2 = remainder_sample(M, x0, d, lo, hi, sl, sr, rng)
print("time", time.time() - t)
tot = hi.copy(); from scipy.special import ndtr; tot = ndtr(hi) - ndtr(lo)
c, _, _ = walk(M, x0, d, lo, hi, tn, tp)
exact_wrong = (tot - c) / tot
fb = band / tot; fr = rem / tot
print("exact err", exact_wrong.mean(), "band", fb.mean(), "remainder est", fr.mean(), "exact remainder", (exact_wrong - fb).mean())
f = fb + fr
print("var band-only", fb.var(), "var total", f.var(), "var exact", exact_wrong.var(), "var exact rem", (exact_wrong-fb).var())
print("passes band", p.mean(), "rem", p2.mean())
cost = p.mean() + p2.mean()
print("var*cost", f.var() * cost, "gain vs sampling", 0.0451 / (f.var() * cost))
blind = np.sort(x,1)[:,-2] <= 0
print("blind share of exact err", (exact_wrong*blind).mean(), "band in blind", (fb*blind).mean())
