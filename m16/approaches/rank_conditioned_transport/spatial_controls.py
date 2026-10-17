"""Constant class and hidden-unit rescaling for the adaptive mesh."""
import json
import argparse
from pathlib import Path
import time
import numpy as np
from engine import estimate
from spatial import terminal_mesh
HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--gate', action='store_true')
args = p.parse_args()
source = json.loads((HERE/('gate_q32_results.json' if args.gate else 'spatial_results.json')).read_text())
model = tuple(np.array(a) for a in source['weights'])
start = time.process_time()
rows = []
for mode in (('gate',) if args.gate else ('curvature', 'reversed')):
    constant = (model[0], model[1], np.zeros_like(model[2]))
    val, stats = estimate(constant, hidden_order=9, target_order=8, target_rule='split_sign',
        mesh_power=2., transition='polynomial_piece_cached', terminal_reuse=True, mesh_builder=terminal_mesh(mode))
    assert abs(val-.25)<1e-13
    rows.append(dict(control='constant', mode=mode, error=val-.25, cpu_seconds=stats['cpu_seconds']))
    scale = np.array([.125, 7.])
    transformed = model[0]*scale, model[1]*scale[:,None]/scale[None,:], model[2]/scale
    val, stats = estimate(transformed, hidden_order=17, target_order=32 if args.gate else 8, target_rule='split_sign',
        mesh_power=2., transition='polynomial_piece_cached', terminal_reuse=True, mesh_builder=terminal_mesh(mode))
    ref = next(r['estimate'] for r in source['rows'] if r['h']==17 and r['mode']==mode)
    assert abs(val-ref)<1e-12
    rows.append(dict(control='positive_hidden_rescaling', mode=mode, error=val-ref, cpu_seconds=stats['cpu_seconds']))
(HERE/('gate_controls_results.json' if args.gate else 'spatial_controls_results.json')).write_text(json.dumps(dict(rows=rows,total_cpu_seconds=time.process_time()-start),indent=2)+'\n')
print(rows)
