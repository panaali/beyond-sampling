"""Frozen first H2T7 LOCALPILOT experiment: baseline and one-axis refinements."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from engine import estimate

p = argparse.ArgumentParser()
p.add_argument('--seconds', type=float, default=110.)
args = p.parse_args()
HERE = Path(__file__).resolve().parent
source = Path('/workspace/work/recover-small_h2_78/m16/approaches/small_h2_78')
checkpoint = source/'LOCALPILOT_H2_T7_seed0.npz'
arrays = np.load(checkpoint)
model = tuple(arrays[k].astype(float) for k in ('Wi', 'Wh', 'Wo'))
metadata = json.loads(checkpoint.with_suffix('.json').read_text())
start = time.process_time()
result = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              engine_sha256=hashlib.sha256((HERE/'engine.py').read_bytes()).hexdigest(),
              model_provenance='LOCALPILOT, not AlgZoo pretrained model',
              existing_heldout_accuracy=metadata['heldout_accuracy'],
              existing_heldout_standard_error=metadata['heldout_standard_error'],
              rows=[])
for name, nh, nq, nx in (('baseline', 9, 8, 8), ('hidden_refinement', 17, 8, 8),
                         ('target_refinement', 9, 16, 8), ('input_refinement', 9, 8, 16)):
    try:
        value, stats = estimate(model, hidden_order=nh, target_order=nq, input_order=nx,
                                target_rule='split_sign', seconds=args.seconds-(time.process_time()-start))
        row = dict(name=name, estimate=value, stats=stats,
                   difference_from_existing_heldout=value-metadata['heldout_accuracy'])
    except (ArithmeticError, ValueError, TimeoutError) as error:
        row = dict(name=name, failure=str(error))
    result['rows'].append(row)
    result['total_cpu'] = time.process_time()-start
    (HERE/'pilot_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(row), flush=True)
    if 'failure' in row:
        break
