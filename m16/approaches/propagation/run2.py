import sys, time
sys.path.insert(0, '/home/user/clean-neurons/m16')
from common import Wi, Wh, Wo
from gsf2 import gsf2
F1 = 10 * (2 * 16 * 16 + 3 * 16) + 2 * 10 * 16
for K, eps in [(1, 1e-3), (1, 1e-4), (4, 1e-4), (16, 1e-5)]:
    for rn in (False, True):
        t0 = time.time(); r = gsf2((Wi, Wh, Wo), K=K, eps=eps, renorm=rn)
        print(K, eps, rn, round(r['acc'], 5), 'mass', round(r['mass'], 4), 'passes', int(r['flops'] / F1),
              'sec', round(time.time() - t0, 1), r['sizes'], flush=True)
