import argparse
import hashlib
import json
import time
import resource
from pathlib import Path
import numpy as np
from engine import estimate

p = argparse.ArgumentParser()
p.add_argument('--analytic', action='store_true')
args = p.parse_args()
HERE = Path(__file__).resolve().parent
path = Path('/workspace/work/recover-small_h2_78/m16/approaches/small_h2_78/LOCALPILOT_H2_T7_seed0.npz')
weights = np.load(path)
model = tuple(weights[k].astype(float) for k in ('Wi', 'Wh', 'Wo'))
start = time.process_time()
result = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              engine_sha256=hashlib.sha256((HERE/'engine.py').read_bytes()).hexdigest())
try:
    value, stats = estimate(model, hidden_order=65, target_order=8, input_order=8,
                            mesh_power=2., target_rule='split_sign', seconds=18.,
                            transition=('polynomial_piece_cached' if args.analytic else 'quantile'),
                            terminal_reuse=args.analytic)
    result.update(estimate=value, stats=stats)
except (ValueError, ArithmeticError, TimeoutError) as error:
    result['failure'] = str(error)
result['total_cpu'] = time.process_time()-start
result['peak_rss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
(HERE/('fine_analytic_results.json' if args.analytic else 'fine_mesh_results.json')).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
