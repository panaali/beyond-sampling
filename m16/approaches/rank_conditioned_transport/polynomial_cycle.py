import hashlib
import json
import time
from pathlib import Path
import numpy as np
from engine import estimate

HERE = Path(__file__).resolve().parent
path = Path('/workspace/work/recover-small_h2_78/m16/approaches/small_h2_78/LOCALPILOT_H2_T7_seed0.npz')
arrays = np.load(path)
model = tuple(arrays[k].astype(float) for k in ('Wi', 'Wh', 'Wo'))
start = time.process_time()
result = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              sources={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                       for name in ('engine.py', 'polynomial.py')}, rows=[])
for n in (17, 33):
    try:
        value, stats = estimate(model, hidden_order=n, target_order=8, mesh_power=2.,
                                target_rule='split_sign', transition='polynomial',
                                seconds=18-(time.process_time()-start))
        row = dict(estimate=value, stats=stats)
    except (ValueError, ArithmeticError, TimeoutError) as error:
        row = dict(hidden_order=n, failure=str(error))
    result['rows'].append(row)
    result['total_cpu'] = time.process_time()-start
    (HERE/'polynomial_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(row), flush=True)
    if 'failure' in row:
        break
