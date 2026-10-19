"""T411 diagnostic: P_W(2.0) ratio along the pure-W direction by paired plain conditional MC (1e7 draws), eps grid."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import design2 as D2  # noqa: E402
import truthc_copy as TCC  # noqa: E402
S = np.array(D2.J("dir_r1.json")["S"]); d = np.array(D2.J("wonly.json")["d"]); E = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.01)
ms = [D2.rnn.unflat(D2.W_ORIG, D2.SH)] + [D2.rnn.unflat((lambda v: (v.__setitem__(S, v[S] + e * d), v)[1])(D2.W_ORIG.copy()), D2.SH) for e in E]
draw = TCC.sampler(2.0); rng = np.random.default_rng(411901); k = np.zeros(len(ms)); N = 10_000_000; spl = np.zeros((len(ms), 10))
for _ in range(N // 250_000):
    x, p3 = draw(rng, 250_000); o = np.argsort(-x, 1); p2 = o[:, 1]
    for i, m in enumerate(ms):
        pk = D2.rnn.forward(m, x).argmax(1); k[i] += (pk != p2).sum(); np.add.at(spl[i], (o == pk[:, None]).argmax(1), 1)
out = [dict(eps=0.0, ratio=1.0, split=spl[0].tolist())] + [dict(eps=e, ratio=k[i + 1] / k[0], split=spl[i + 1].tolist()) for i, e in enumerate(E)]
for r in out: print(r["eps"], "%.3f" % r["ratio"], [int(v) for v in r["split"][:5]])
D2.J("path_W.json", out)
