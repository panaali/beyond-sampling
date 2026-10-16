# DIAGNOSIS ONLY: compare propagated per-(p1,p2) masses / means with Monte Carlo
import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from common import Wi, Wh, Wo, run
import gsf as G
x = np.random.default_rng(1).standard_normal((400_000, 10)); hs = run(x)
C = G.Counter(); M = G.preact_rows(Wi, Wh)
c = G.Comp(np.zeros((1, 18)), np.zeros((1, 18, 18)), np.ones(1), np.array([-1]), np.array([-1]))
for t in range(4):
    B = len(c)
    mu = np.concatenate([c.mu, np.zeros((B, 1))], 1)
    S = np.zeros((B, 19, 19)); S[:, :18, :18] = c.S; S[:, 18, 18] = 1
    pieces = G.rank_split(G.Comp(mu, S, c.w, c.p1, c.p2), t, C)
    new = [G.relu_step(p, M, s, C) for p, s in pieces]
    c = G.merge(G.Comp.cat(new), 1, C)
    o = np.argsort(-x[:, :t + 1], 1)
    for k in range(min(len(c), 4)):
        p1, p2 = c.p1[k], c.p2[k]
        sel = (o[:, 0] == p1) & ((o[:, 1] == p2) if t > 0 else True)
        print(t, p1, p2, 'w', round(c.w[k], 4), round(sel.mean(), 4))
        print('  mu h', np.round(c.mu[k, :16], 2)); print('  MC h', np.round(hs[sel, t + 1].mean(0), 2))
        print('  m1m2', np.round(c.mu[k, 16:], 2), np.round([x[sel, p1].mean(), x[sel, p2].mean() if t else 0], 2))
