"""Independent exact sign-toy and constant controls for the u-variable law."""
import json
import argparse
import time
from pathlib import Path
import numpy as np
from engine import estimate
start = time.process_time()
rows = []
parser = argparse.ArgumentParser()
parser.add_argument('--target-limit', type=float)
args = parser.parse_args()
p = Path(__file__).resolve().parent/('value_bounded_controls_results.json' if args.target_limit else 'value_controls_results.json')
for T in (3, 4, 7):
    wi, wh = np.array([1., 0.]), np.zeros((2, 2))
    for kind in ('constant', 'sign'):
        wo = np.zeros((T, 2))
        truth = 1/T
        if kind == 'sign':
            wo[-1,0] = 1
            truth += 1/(2*(T-1))-(T+1)/((T-1)*2**T)
        for n in (32,64):
            try:
                value, stats = estimate((wi, wh, wo), hidden_order=5, target_order=n,
                    target_rule='split_value', mesh_power=2., transition='polynomial_piece_cached',
                    terminal_reuse=True, seconds=18-(time.process_time()-start), target_value_limit=args.target_limit)
            except (ArithmeticError, ValueError, TimeoutError) as error:
                rows.append(dict(T=T,kind=kind,order=n,failure=str(error)))
                p.write_text(json.dumps(dict(rows=rows,total_cpu_seconds=time.process_time()-start),indent=2)+'\n')
                raise
            omitted = stats['omitted_target_mass_bound']
            error = value-(truth*(1-omitted) if kind=='constant' else truth)
            assert (-omitted-1e-11 <= error <= 1e-11) if kind=='sign' else abs(error)<1e-11, (T,kind,n,error)
            rows.append(dict(T=T, kind=kind, order=n, error=error,
                omitted_target_mass=omitted,
                mass_defect=stats['work']['target_density_raw_mass_defect'], cpu_seconds=stats['cpu_seconds']))
p.write_text(json.dumps(dict(rows=rows,total_cpu_seconds=time.process_time()-start),indent=2)+'\n')
print(rows)
