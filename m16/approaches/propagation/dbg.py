import sys; sys.path.insert(0, '/home/user/clean-neurons/m16')
import numpy as np
from common import Wi, Wh, Wo
import gsf2 as G
C = G.Counter(); M = np.zeros((16, 19)); M[:, :16] = Wh; M[:, 18] = Wi
c = G.Comp(mu=np.zeros((1, 18)), S=np.zeros((1, 18, 18)), w=np.ones(1), p1=np.array([-1]), p2=np.array([-1]), g=np.zeros((1, 16), bool))
for t in range(3):
    B = len(c)
    mu = np.concatenate([c.mu, np.zeros((B, 1))], 1)
    S = np.zeros((B, 19, 19)); S[:, :18, :18] = c.S; S[:, 18, 18] = 1
    aug = G.Comp(mu=mu, S=S, w=c.w, p1=c.p1, p2=c.p2, g=c.g)
    new = []
    for comp, sel in G.rank_split(aug, t, C):
        d = [0.0]
        gc = G.enumerate_gates(comp, M, 1e-5, C, d)
        print(t, sel, 'in mass', comp.w.sum().round(4), 'out', gc.w.sum().round(4), 'n', len(gc), 'drop', round(d[0], 4))
        print('   ', [(''.join('1' if b else '0' for b in g), round(w, 4)) for g, w in zip(gc.g[:6], gc.w[:6])])
        new.append(G.linear_step(gc, M, sel, C))
    c = G.merge(G.Comp.cat(new), 1, C)
