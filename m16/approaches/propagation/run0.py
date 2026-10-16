import sys, time
sys.path.insert(0, '/home/user/clean-neurons/m16')
from common import Wi, Wh, Wo
from prop import propagate
F1 = 10 * (2 * 16 * 16 + 3 * 16) + 2 * 10 * 16
for eps in [1e-3, 1e-4, 1e-5]:
    for merge in [False, True]:
        t0 = time.time(); r = propagate((Wi, Wh, Wo), eps=eps, merge=merge)
        print(eps, merge, round(r['acc'], 5), round(r['acc_lower'], 5), 'drop', round(r['dropped'], 5), 'nfinal', r['n_final'],
              'passes', int(r['flops'] / F1), 'sec', round(time.time() - t0, 1), r['branches'], flush=True)
