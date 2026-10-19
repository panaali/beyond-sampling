"""T402 checks of lcmc: (1) e1 reproduces ../cmcgrad.py on orig E1(1.25) (same seed and draws); (2) acc vs plain MC and
its gradient vs a CRN finite difference of plain MC along the T372 top direction of orig."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lcmc  # noqa: E402
import rnn  # noqa: E402

D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "dirs_orig.json")))
sh = [(16,), (16, 16), (10, 16)]; w0 = np.array(D["w"]); m = rnn.unflat(w0, sh)
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cmc_orig.json")))
r = lcmc.e1(m, 1.25, 200_000, 372200)
print("e1 2e5 lines: P %.4e +- %.1e (cmcgrad 5e6: %.4e)" % (r["P"], r["se"], C["P"]), flush=True)
a = lcmc.acc(m, 200_000, 7)
print("acc line-CMC %.5f +- %.5f (truth .95318)  C %.0f FPE" % (a["P"], a["se"], a["C_fpe"]), flush=True)
u = np.array(D["dirs"][0]); h = 0.01; rng = np.random.default_rng(8); N = 4_000_000; s1 = s2 = 0.0
mp, mm = rnn.unflat(w0 + h * u, sh), rnn.unflat(w0 - h * u, sh)
for _ in range(N // 200_000):
    x = rng.standard_normal((200_000, 10)); tgt = np.argsort(-x, 1)[:, 1]
    dd = (rnn.forward(mp, x).argmax(1) == tgt).astype(float) - (rnn.forward(mm, x).argmax(1) == tgt)
    s1 += dd.sum(); s2 += (dd ** 2).sum()
mu = s1 / N; sd = np.sqrt(s2 / N - mu ** 2)
print("d acc/du: line-CMC %.4f +- %.4f   CRN FD %.4f +- %.4f" % (a["grad"] @ u, np.sqrt(((a["grad_se"] * u) ** 2).sum()), mu / (2 * h),
                                                             sd / np.sqrt(N) / (2 * h)))
