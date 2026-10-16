import sys, time
sys.path.insert(0, '/home/user/clean-neurons/m16')
from common import Wi, Wh, Wo
from gsf import gsf
F1 = 10 * (2 * 16 * 16 + 3 * 16) + 2 * 10 * 16
for K, ns, nf in [(1, 0, 0), (1, 0, 4), (1, 0, 8), (1, 2, 4), (2, 2, 4), (4, 2, 4), (4, 3, 6)]:
    t0 = time.time(); r = gsf((Wi, Wh, Wo), K=K, n_split=ns, n_final=nf)
    print(K, ns, nf, round(r['acc'], 5), 'mass', round(r['mass'], 5), 'passes', int(r['flops'] / F1),
          'sec', round(time.time() - t0, 1), r['sizes'], flush=True)
