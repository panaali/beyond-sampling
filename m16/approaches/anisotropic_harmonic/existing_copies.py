"""Unretuned N8/M4 validation on T47's existing parameter-noise ensemble."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from engine import estimate
HERE=Path(__file__).resolve().parent
REVIEW=Path('/workspace/work/recover-small_h2_910/m16/approaches/harmonic_negative_review')
SOURCE=Path('/workspace/clean-neurons/m16/approaches/small_h2_34')
sys.path.insert(0,str(REVIEW))
from diagnose import evaluate
weights=json.loads((SOURCE/'weights_seed0.json').read_text())['T3']
base=tuple(np.array(weights[k]) for k in ('Wi','Wh','Wo'))
truth=json.loads((SOURCE/'results_t3.json').read_text())
noise=np.random.default_rng(47001)
models=[base]+[tuple(w+.05*noise.normal(size=w.shape) for w in base) for _ in range(8)]
start=time.process_time();deadline=start+18
output=dict(source_truth_sha256=hashlib.sha256((SOURCE/'results_t3.json').read_bytes()).hexdigest(),
            frozen_rule='N8 M4',parameter_noise_seed=47001,absolute_sigma=.05,
            random_model_inputs=0,normal_resolution_controls=[],rows=[])
for N in (4,16):
    p,s=estimate(base,N,4,seconds=deadline-time.process_time())
    output['normal_resolution_controls'].append(dict(N=N,M=4,estimate=p,error=p-truth['rows'][0]['accuracy'],stats=s))
for i,model in enumerate(models):
    reference=truth['rows'][i]['accuracy']
    value,stats=estimate(model,8,4,seconds=deadline-time.process_time())
    isotropic=evaluate(model,8,True,True,deadline)
    row=dict(copy=i,reference=reference,model_sha256=hashlib.sha256(b''.join(a.astype('<f8').tobytes() for a in model)).hexdigest(),
             supported=dict(estimate=value,error=value-reference,stats=stats),
             isotropic_corrected=dict(estimate=isotropic['estimate'],error=isotropic['estimate']-reference,stats=isotropic))
    for method in ('supported','isotropic_corrected'):
        row[method]['squared_error_times_cpu']=row[method]['error']**2*row[method]['stats']['cpu_seconds']
    output['rows'].append(row)
    output['total_cpu_seconds']=time.process_time()-start
    (HERE/'existing_copies_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(copy=i,supported_error=value-reference,isotropic_error=isotropic['estimate']-reference)),flush=True)
output['aggregate']={method:dict(rmse=float(np.sqrt(np.mean([r[method]['error']**2 for r in output['rows']]))),
     mean_cpu=float(np.mean([r[method]['stats']['cpu_seconds'] for r in output['rows']])),
     mean_squared_error_times_cpu=float(np.mean([r[method]['squared_error_times_cpu'] for r in output['rows']])))
     for method in ('supported','isotropic_corrected')}
output['cpu_adjusted_squared_error_ratio']=output['aggregate']['supported']['mean_squared_error_times_cpu']/output['aggregate']['isotropic_corrected']['mean_squared_error_times_cpu']
output['cost_limit']='CPU-adjusted squared-error product, not a same-budget or portable-FLOP claim; deterministic runs cannot be averaged to remove bias.'
(HERE/'existing_copies_results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output['aggregate']))
