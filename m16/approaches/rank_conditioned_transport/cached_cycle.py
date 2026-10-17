import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from engine import estimate
from polynomial import conditional_expectations, conditional_matrices, StagePieces

p = argparse.ArgumentParser()
p.add_argument('--reuse-pieces', action='store_true')
args = p.parse_args()
transition = 'polynomial_piece_cached' if args.reuse_pieces else 'polynomial_cached'
HERE = Path(__file__).resolve().parent
start = time.process_time()
stats = defaultdict(float)
model = (np.array([.8, -.6]), np.array([[.4, -.7], [.3, .9]]), np.zeros((3, 2)))
states = np.array([[0., 0.], [.2, .7], [.5, .1]])
axes = (np.array([0., .1, .7, 2., 6.]), np.array([0., .2, .9, 3., 6.]))
x, y = np.meshgrid(*axes, indexing='ij')
values = np.stack((.2+.02*x+.03*y+.002*x*y, np.ones_like(x)), axis=-1)
u = .35
q = (ndtr(u)-ndtr(-6))/(1-2*ndtr(-6))
low, high, _ = conditional_expectations(model, states, axes, values, u, q, 6., stats, start+18)
if args.reuse_pieces:
    matrices, bound = StagePieces(model, states, axes, 6., stats, start+18).matrices(u, q, stats, start+18)
else:
    matrices, bound = conditional_matrices(model, states, axes, u, q, 6., stats, start+18)
delta = max(float(np.max(abs(matrix@values.reshape(-1, 2)-reference)))
            for matrix, reference in zip(matrices, (low, high)))
assert delta < 1e-12
constant = (model[0], model[1], np.zeros((7, 2)))
value, work = estimate(constant, hidden_order=5, target_order=8, mesh_power=2.,
                        target_rule='split_sign', transition=transition, seconds=18-(time.process_time()-start))
assert abs(value-1/7) < 1e-12
path = Path('/workspace/work/recover-small_h2_78/m16/approaches/small_h2_78/LOCALPILOT_H2_T7_seed0.npz')
arrays = np.load(path)
model = tuple(arrays[k].astype(float) for k in ('Wi', 'Wh', 'Wo'))
result = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              sources={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                       for name in ('engine.py', 'polynomial.py')},
              direct_kernel_max_difference=delta, kernel_control_work=dict(stats),
              constant_control=dict(estimate=value, error=value-1/7, stats=work), rows=[])
for n in (17, 33):
    try:
        value, stats = estimate(model, hidden_order=n, target_order=8, mesh_power=2.,
                                target_rule='split_sign', transition=transition,
                                seconds=18-(time.process_time()-start))
        row = dict(estimate=value, stats=stats)
    except (ValueError, ArithmeticError, TimeoutError) as error:
        row = dict(hidden_order=n, failure=str(error))
    result['rows'].append(row)
    result['total_cpu'] = time.process_time()-start
    (HERE/('piece_cached_results.json' if args.reuse_pieces else 'cached_results.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(row), flush=True)
    if 'failure' in row:
        break
