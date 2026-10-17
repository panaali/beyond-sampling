"""Prespecified quadratic-grid convergence plus uniform33 control."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from engine import estimate

HERE = Path(__file__).resolve().parent
checkpoint = Path('/workspace/work/recover-small_h2_78/m16/approaches/small_h2_78/LOCALPILOT_H2_T7_seed0.npz')
arrays = np.load(checkpoint)
model = tuple(arrays[k].astype(float) for k in ('Wi', 'Wh', 'Wo'))
start = time.process_time()
result = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              engine_sha256=hashlib.sha256((HERE/'engine.py').read_bytes()).hexdigest(), rows=[])
for power, n in ((2., 9), (2., 17), (2., 33), (1., 33)):
    try:
        value, stats = estimate(model, hidden_order=n, mesh_power=power,
                                target_order=8, input_order=8, target_rule='split_sign',
                                seconds=18-(time.process_time()-start))
        row = dict(estimate=value, stats=stats)
    except (ValueError, ArithmeticError, TimeoutError) as error:
        row = dict(mesh_power=power, hidden_order=n, failure=str(error))
    result['rows'].append(row)
    result['total_cpu'] = time.process_time()-start
    (HERE/'mesh_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(row), flush=True)
    if 'failure' in row:
        break
