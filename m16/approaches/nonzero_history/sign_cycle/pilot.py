"""Frozen four-case correct/shuffled association test; truth only for scoring."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from engine import calculate

HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/clean-neurons/m16/approaches')
sys.path.insert(0,str(ROOT/'small_h2_56'))
from models import load
sys.path.pop(0)
start=time.process_time()
base4,sha4=load(4);base5,sha5=load(5);seed2,sha52=load(5,2)
rng=np.random.default_rng(47021)
noise4=tuple(w+.02*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in base4)
old={r['name']:r for filename in ('benchmark.json','transfer.json')
     for r in json.loads((ROOT/'h2_boundary_mechanism'/filename).read_text())['cases']}
previous={(r['name'],r['mode']):r for r in json.loads(
    (HERE.parent/'accounted_landmark_pilot.json').read_text())['rows']}
cases=[('T4_copy0',base4),('T4_rms0.02_copy0',noise4),
       ('T5_sigma0.0',base5),('T5_seed2',seed2)]
out=dict(rows=[],checkpoint_sha256=[sha4,sha5,sha52],
         source_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest()
                        for f in ('engine.py','ranks.py')},
         preparation_cpu=time.process_time()-start)
for name,model in cases:
    for shuffled in (False,True):
        row=dict(name=name,shuffled=shuffled,reference=old[name]['reference'],
                 reference_bound=old[name]['reference_bound'],
                 previous={mode:previous[name,mode]['estimate']
                           for mode in ('zero','plugin','exact2')})
        try:
            value,stats=calculate(model,shuffled,16)
            fine,fstats=calculate(model,shuffled,32)
            row.update(estimate=value,reference_error=value-row['reference'],
                       refinement=fine,refinement_change=abs(fine-value),
                       stats=stats,refinement_stats=fstats)
        except Exception as exc:
            row['failure']=repr(exc)
        out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
        (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items()
                          if k not in ('stats','refinement_stats')}),flush=True)
        if 'failure' in row:raise SystemExit(1)
        if time.process_time()-start>100:raise SystemExit('frozen CPU limit')
