"""Frozen same-node comparisons against pre-existing exact H2T4 reference."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from engine import estimate
from spatial import terminal_mesh

p = argparse.ArgumentParser()
p.add_argument('--refine-q', action='store_true')
p.add_argument('--q', type=int)
p.add_argument('--gate-cycle', action='store_true')
args = p.parse_args()
HERE = Path(__file__).resolve().parent
path = Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w = torch.load(path, map_location='cpu', weights_only=True)
model = tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
model = (model[0].ravel(), *model[1:])
reference = .30773074302392744
start = time.process_time()
result = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                             for name in ('engine.py', 'spatial.py')},
              reference=reference, weights=[a.tolist() for a in model], rows=[])
target_order = args.q or (16 if args.refine_q else 8)
for h in ((33,) if (args.refine_q or args.q) and not args.gate_cycle else (17, 33)):
    for mode in (('baseline', 'curvature', 'gate') if args.gate_cycle else ('baseline', 'curvature', 'reversed')):
        try:
            value, stats = estimate(model, hidden_order=h, target_order=target_order,
                target_rule='split_sign', mesh_power=2., transition='polynomial_piece_cached',
                terminal_reuse=True, mesh_builder=None if mode=='baseline' else terminal_mesh(mode),
                seconds=18-(time.process_time()-start))
            row = dict(h=h, mode=mode, estimate=value, error=value-reference, stats=stats)
        except (ValueError, ArithmeticError, TimeoutError) as error:
            row = dict(h=h, mode=mode, failure=str(error))
        result['rows'].append(row)
        result['total_cpu_seconds'] = time.process_time()-start
        name = f'spatial_q{target_order}_results.json' if args.refine_q or args.q else 'spatial_results.json'
        if args.gate_cycle:
            name = f'gate_q{target_order}_results.json'
        (HERE/name).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='stats'}), flush=True)
        if 'failure' in row:
            raise SystemExit(1)
