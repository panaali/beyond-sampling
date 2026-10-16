import sys, json, numpy as np
sys.path.insert(0, '/home/user/clean-neurons/m16')
from arc_eval import noisy_models, Wi0, Wh0, Wo0, est_sampling
from hybrid import est_hybrid
truth = json.load(open('/home/user/clean-neurons/results/leaderboard_truths.json'))
models = {'orig': (Wi0, Wh0, Wo0)}
for s in (0.0025, 0.005):
    for k, m in enumerate(noisy_models(6, s, seed=0)):
        models[f's{s}_{k}'] = m
B = int(sys.argv[1]); reps = int(sys.argv[2])
ests = {'sampling': est_sampling, 'region-strat only': lambda m, b, r: est_hybrid(m, b, r, w_dp=(0, 0)),
        'hybrid DP-IS': est_hybrid, 'hybrid DP-IS w.4': lambda m, b, r: est_hybrid(m, b, r, w_dp=(0.3, 0.4))}
for grp in ('orig', 's0.0025', 's0.005'):
    names = [n for n in models if n.startswith(grp)]
    line = []
    for en, e in ests.items():
        se = []
        for n in names:
            for r in range(reps):
                v, _ = e(models[n], B, np.random.default_rng(100 * r + 7))
                se.append((v - truth[n]['acc']) ** 2)
        line.append(f'{en}: {100*np.sqrt(np.mean(se)):.3f}pp')
    print(grp, 'budget', B, ' | '.join(line), flush=True)
