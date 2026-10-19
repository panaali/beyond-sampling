"""Containment and tightness test of zono.rnn_closure's bounds against sampled continuations (T447)."""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mini_lm"))
from load import load
from langs import model_input
import zono, cert

rng = np.random.default_rng(0)
FN = getattr(zono, sys.argv[1] if len(sys.argv) > 1 else "rnn_closure")
for name in ["dyck2_rnn_s0", "dyck2_rnn_s1", "dyck2_rnn_s2"]:
    model, lang = load(name); P = cert.Params(model)
    T = 15
    for k in (6, 10, 12):
        y0 = lang.sample(1, rng, L=T)
        ys = lang.sample(400000, rng, L=T); ys = ys[(ys[:, :k] == y0[0, :k]).all(1)]
        x = model_input(ys, lang.BOS)
        with torch.no_grad():
            lg, hs = model(torch.from_numpy(x), return_hidden=True)
        h0 = hs[:1, k].double()
        A = T - 1 - k
        al = torch.ones(1, A, 4, dtype=torch.bool)
        rec = []
        FN(P, h0, al, lambda a, lo, hi: rec.append((lo, hi)) or torch.ones(1, dtype=torch.bool))
        viol, wid = 0.0, []
        for a, (lo, hi) in enumerate(rec):
            i = k + 1 + a
            own = torch.from_numpy(x[:, i])
            l, h = lo[0, own], hi[0, own]
            viol = max(viol, ((lg[:, i].double() - h).clamp(min=0) + (l - lg[:, i].double()).clamp(min=0)).max().item())
            wid.append(round(((hi - lo) / 2).mean().item(), 3))
        print(name, "k", k, "n", len(ys), "violation %.2e" % viol, "mean logit radius by step", wid)
