"""Independent exact sign-toy and constant controls for the u-variable law."""
import json
import time
from pathlib import Path
import numpy as np
from engine import estimate
start = time.process_time()
rows = []
p = Path(__file__).resolve().parent/'value_controls_results.json'
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
                    terminal_reuse=True, seconds=18-(time.process_time()-start))
            except (ArithmeticError, ValueError, TimeoutError) as error:
                rows.append(dict(T=T,kind=kind,order=n,failure=str(error)))
                p.write_text(json.dumps(dict(rows=rows,total_cpu_seconds=time.process_time()-start),indent=2)+'\n')
                raise
            error = value-truth
            assert abs(error)<1e-11, (T,kind,n,error)
            rows.append(dict(T=T, kind=kind, order=n, error=error,
                mass_defect=stats['work']['target_density_raw_mass_defect'], cpu_seconds=stats['cpu_seconds']))
p.write_text(json.dumps(dict(rows=rows,total_cpu_seconds=time.process_time()-start),indent=2)+'\n')
print(rows)
