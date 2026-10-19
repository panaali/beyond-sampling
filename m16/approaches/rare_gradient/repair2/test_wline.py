"""Check wline against plain conditional MC of W(1.0) on orig, and its gradient against a paired FD along T372's top direction."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rnn, truthc_copy as TCC, wline
D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "dirs_orig.json")))
sh = [(16,), (16, 16), (10, 16)]; w0 = np.array(D["w"]); m = rnn.unflat(w0, sh); t0 = 1.0
u = np.array(D["dirs"][5]); o = wline.w(m, t0, 1_000_000, 15, U=u[None])
draw = TCC.sampler(t0); rng = np.random.default_rng(6); u = np.array(D["dirs"][5]); h = 3e-3
mp, mm = rnn.unflat(w0 + h * u, sh), rnn.unflat(w0 - h * u, sh); n = 0; k = 0; s1 = s2 = 0.0
for _ in range(100):
    x, p3 = draw(rng, 200_000); p2 = np.argsort(-x, 1)[:, 1]
    k += (rnn.forward(m, x).argmax(1) != p2).sum(); dd = (rnn.forward(mp, x).argmax(1) != p2).astype(float) - (rnn.forward(mm, x).argmax(1) != p2)
    s1 += dd.sum(); s2 += (dd ** 2).sum(); n += 200_000
pg = TCC.p_gap(t0); mu = s1 / n
print("P_W line %.4e +- %.1e | plain cond MC %.4e +- %.1e" % (o["P"], o["se"], pg * k / n, pg * np.sqrt(k) / n))
print("dP_W/du line %.4e +- %.1e  FD %.4e +- %.1e   (C line %.3g FPE)" % (o["proj"][0], o["proj_se"][0], pg * mu / (2 * h), pg * np.sqrt(s2 / n - mu ** 2) / np.sqrt(n) / (2 * h), o["C_fpe"]))
