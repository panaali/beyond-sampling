"""Frozen four-case nonzero-history test; references enter validation only."""
import hashlib
import json
from pathlib import Path
import sys
import time
import argparse
import numpy as np
from affine import calculate

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--normal-landmarks',action='store_true')
parser.add_argument('--output',default=None)
args=parser.parse_args()
ROOT=Path('/workspace/clean-neurons/m16/approaches')
sys.path.insert(0,str(ROOT/'small_h2_56'))
from models import load
sys.path.pop(0)
start=time.process_time()
base4,sha4=load(4);base5,sha5=load(5);seed2,sha52=load(5,2)
rng=np.random.default_rng(47021)
noise4=tuple(w+.02*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in base4)
old={r['name']:r for name in ('benchmark.json','transfer.json')
     for r in json.loads((ROOT/'h2_boundary_mechanism'/name).read_text())['cases']}
cases=[('T4_copy0',base4),('T4_rms0.02_copy0',noise4),('T5_sigma0.0',base5),('T5_seed2',seed2)]
out=dict(rows=[],checkpoint_sha256=[sha4,sha5,sha52],
    source_sha256=hashlib.sha256((HERE/'affine.py').read_bytes()).hexdigest(),
    preparation_cpu=time.process_time()-start)
for name,model in cases:
    ref=old[name]
    for mode in ('zero','plugin','exact2'):
        row=dict(name=name,mode=mode,reference=ref['reference'],reference_bound=ref['reference_bound'],
                 previous_zero_estimate=ref['estimate'])
        try:
            value,stats=calculate(model,mode,16,normal_landmarks=args.normal_landmarks)
            fine,fstats=calculate(model,mode,32,normal_landmarks=args.normal_landmarks)
            row.update(estimate=value,reference_error=value-ref['reference'],
                refinement=fine,refinement_change=abs(fine-value),stats=stats,refinement_stats=fstats,
                zero_reproduction_error=abs(fine-ref['estimate']) if mode=='zero' else None)
        except Exception as exc:row['failure']=repr(exc)
        out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
        filename=args.output or ('landmark_pilot.json' if args.normal_landmarks else 'pilot.json')
        (HERE/filename).write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('stats','refinement_stats')}),flush=True)
        if 'failure' in row:raise SystemExit(1)
