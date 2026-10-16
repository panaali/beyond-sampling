"""Evaluate configurations on the original and the 12 noisy copies (truth: results/leaderboard_truths.json, n=4M)."""
import sys, json, time, numpy as np
sys.path.insert(0, '/home/user/clean-neurons/m16')
from arc_eval import noisy_models, Wi0, Wh0, Wo0
import estimator as E
truth = json.load(open('/home/user/clean-neurons/results/leaderboard_truths.json'))
models = {'orig': (Wi0, Wh0, Wo0)}
for s in (0.0025, 0.005):
    for k, m in enumerate(noisy_models(6, s, seed=0)):
        models[f's{s}_{k}'] = m
which = sys.argv[1]
cfgs = {'full': {}, 'L0': E.LADDER[0], 'L1': E.LADDER[1], 'L2': E.LADDER[2]}
names = [n for n in models if n == 'orig' or sys.argv[2] == 'all']
out = {}
for name in names:
    t = time.time()
    acc, cost, info = E.estimate(models[name], cfgs[which])
    out[name] = dict(est=acc, truth=truth[name]['acc'], cost=cost, **info)
    print(which, name, f'est {acc:.5f} truth {truth[name]["acc"]:.5f} err {100*(acc-truth[name]["acc"]):+.3f}pp cost {cost:.0f} '
          f'fpos {info["f_pos"]:.5f} fneg {info["f_neg"]:.4f} {time.time()-t:.0f}s', flush=True)
json.dump(out, open(f'sweep_{which}_{sys.argv[2]}.json', 'w'), indent=1)
for s in ('0.0025', '0.005'):
    e = [out[n]['est'] - out[n]['truth'] for n in out if n.startswith('s' + s + '_')]
    if e:
        print(which, 'sigma', s, 'RMSE', f'{100*np.sqrt(np.mean(np.square(e))):.3f}pp', 'mean err', f'{100*np.mean(e):+.3f}pp')
