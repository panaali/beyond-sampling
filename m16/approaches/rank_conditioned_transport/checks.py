import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from engine import estimate, decision_intervals, normal_mass

p = argparse.ArgumentParser()
p.add_argument('--target-rule', choices=('jacobi', 'split_sign'), default='jacobi')
args = p.parse_args()
start = time.process_time()
result = dict(constants=[], scalar_partitions=[], sign_toys=[])
out = Path(__file__).with_name('checks_results.json' if args.target_rule == 'jacobi' else 'checks_split_results.json')
def save():
    result['total_cpu'] = time.process_time()-start
    out.write_text(json.dumps(result, indent=2)+'\n')

for T in (3, 4, 7):
    wi = np.array([.7, -.4])
    wh = np.array([[.8, -.3], [.2, .9]])
    for identical in (False, True):
        wo = np.tile([.4, -.7], (T, 1)) if identical else np.zeros((T, 2))
        p, stats = estimate((wi, wh, wo), hidden_order=5, target_order=8, input_order=8,
                            target_rule=args.target_rule)
        result['constants'].append(dict(T=T, identical=identical, estimate=p, exact=1/T,
                                        error=p-1/T, stats=stats))
        assert abs(p-1/T) < 2e-12
        save()

wi = np.array([.8, -.6])
wh = np.array([[.4, -.7], [.3, .9]])
wo = np.array([[.4, -.2], [-.1, .5], [.3, .3], [.4, -.2]])
for hidden in (np.zeros(2), np.array([.2, .7]), np.array([1.2, 0.])):
    for suffix in (False, True):
        stats = defaultdict(float)
        probabilities = []
        for label in range(4):
            intervals = decision_intervals((wi, wh, wo), hidden, .35, label, 6., stats,
                                           target_last=suffix)
            probabilities.append(sum(normal_mass(lo, hi) for lo, hi in intervals)/normal_mass(-6., 6.))
        row = dict(hidden=hidden.tolist(), target_suffix=suffix, probabilities=probabilities,
                   partition_error=sum(probabilities)-1, work=dict(stats))
        result['scalar_partitions'].append(row)
        assert abs(row['partition_error']) < 2e-12
        assert probabilities[-1] == 0  # duplicate row loses its positive-mass ties
        save()

for T in (3, 4, 7):
    wi = np.array([1., 0.])
    wh = np.zeros((2, 2))
    wo = np.zeros((T, 2))
    wo[-1, 0] = 1.
    truth = 1/T+1/(2*(T-1))-(T+1)/((T-1)*2**T)
    for n in (8, 16, 32):
        p, stats = estimate((wi, wh, wo), hidden_order=3, target_order=n, input_order=8,
                            target_rule=args.target_rule)
        result['sign_toys'].append(dict(T=T, target_order=n, estimate=p, exact=truth,
                                        error=p-truth, stats=stats))
        save()
print(json.dumps(dict(total_cpu=result['total_cpu'],
                      constant_max_error=max(abs(x['error']) for x in result['constants']),
                      partition_max_error=max(abs(x['partition_error']) for x in result['scalar_partitions']),
                      toy_errors=[(x['T'], x['target_order'], x['error']) for x in result['sign_toys']])) )
