"""DIAGNOSTIC: per relative-gap band, v3 prediction vs sampling, for several settings."""
import sys, numpy as np
sys.path.insert(0, '/home/user/clean-neurons/m16')
from v3 import estimate
from quad import grid
from common import Wi, Wh, Wo
C, U, G, W = grid(16, 16, 16); B = C + U; A = B + G
edges = [0, .01, .03, .08, .2, .5, 100]
samp = {'23': [0.0103, 0.0063, 0.0058, 0.0035, 0.0012, 0.0004], '12': [0.0050, 0.0016, 0.0010, 0.0001, 0.0000, 0.0001]}
for args in sys.argv[1:]:
    nv, ng, q, noise = args.split(',')
    o, ev = estimate((Wi, Wh, Wo), int(nv), int(ng), int(q), noise=float(noise))
    print(args, 'evals', ev, 'err23 %.4f err12 %.4f' % (o['23'].sum(), o['12'].sum()))
    for kind, gp in (('23', U / B), ('12', G / A)):
        print('  ', kind, ' '.join('%.4f' % o[kind + '_node'][(gp >= lo) & (gp < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])))
