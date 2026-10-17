import hashlib
import json
import time
from pathlib import Path
import numpy as np
from engine import estimate

HERE = Path(__file__).resolve().parent
path = Path('/workspace/work/recover-small_h2_78/m16/approaches/small_h2_78/LOCALPILOT_H2_T7_seed0.npz')
weights = np.load(path)
model = tuple(weights[k].astype(float) for k in ('Wi', 'Wh', 'Wo'))
start = time.process_time()
result = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              engine_sha256=hashlib.sha256((HERE/'engine.py').read_bytes()).hexdigest())
try:
    value, stats = estimate(model, hidden_order=65, target_order=8, input_order=8,
                            mesh_power=2., target_rule='split_sign', seconds=18.)
    result.update(estimate=value, stats=stats)
except (ValueError, ArithmeticError, TimeoutError) as error:
    result['failure'] = str(error)
result['total_cpu'] = time.process_time()-start
(HERE/'fine_mesh_results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
