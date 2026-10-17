"""Separate target/input refinements and exact-reference validation."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from engine import estimate

p = argparse.ArgumentParser()
p.add_argument('case', choices=('t7', 't4'))
args = p.parse_args()
HERE = Path(__file__).resolve().parent
start = time.process_time()
if args.case == 't7':
    path = Path('/workspace/work/recover-small_h2_78/m16/approaches/small_h2_78/LOCALPILOT_H2_T7_seed0.npz')
    arrays = np.load(path)
    model = tuple(arrays[k].astype(float) for k in ('Wi', 'Wh', 'Wo'))
    choices = ((33, 16, 8), (33, 8, 16))
    reference = None
else:
    import torch
    path = Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
    weights = torch.load(path, weights_only=True, map_location='cpu')
    wi, wh, wo = (weights[k].numpy().astype(float) for k in
                  ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
    model = (wi.ravel(), wh, wo)
    choices = ((17, 8, 8), (33, 8, 8))
    reference = .30773074302392744
result = dict(case=args.case, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              engine_sha256=hashlib.sha256((HERE/'engine.py').read_bytes()).hexdigest(),
              cone_reference=reference, rows=[])
for nh, nq, nx in choices:
    try:
        value, stats = estimate(model, hidden_order=nh, target_order=nq, input_order=nx,
                                mesh_power=2., target_rule='split_sign',
                                seconds=18-(time.process_time()-start))
        row = dict(estimate=value, stats=stats)
        if reference is not None:
            row['difference_from_cone'] = value-reference
    except (ValueError, ArithmeticError, TimeoutError) as error:
        row = dict(hidden=nh, target=nq, inputs=nx, failure=str(error))
    result['rows'].append(row)
    result['total_cpu'] = time.process_time()-start
    (HERE/(args.case+'_refine_results.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(row), flush=True)
    if 'failure' in row:
        break
