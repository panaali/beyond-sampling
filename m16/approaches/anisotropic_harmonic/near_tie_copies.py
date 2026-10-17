"""Fixed H4T3 anisotropic rescue on pre-existing noisy copies and symmetries."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from engine import estimate
HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/clean-neurons/m16/approaches/arc_m43_repro')
REVIEW=Path('/workspace/work/recover-small_h2_910/m16/approaches/harmonic_negative_review')
sys.path.insert(0,str(REVIEW))
from diagnose import evaluate
path=SOURCE/'m43_weights.npz'
a=np.load(path);base=tuple(a[k].astype(float) for k in ('Wi','Wh','Wo'))
truth=json.loads((SOURCE/'results.json').read_text())
noise=np.random.default_rng(0)
models=[tuple(w+.05*noise.standard_normal(w.shape) for w in base) for _ in range(8)]
start=time.process_time();deadline=start+18
result=dict(source_truth_sha256=hashlib.sha256((SOURCE/'results.json').read_bytes()).hexdigest(),
            parameter_noise_seed=0,absolute_sigma=.05,random_model_inputs=0,controls=[],rows=[])
reference=json.loads((HERE/'near_tie_results.json').read_text())['rows'][0]['estimate']
scale=np.array([.125,7.,.3,2.])
for kind,model,Q in (('hidden_scaling',(base[0]*scale,base[1]*scale[:,None]/scale[None,:],base[2]/scale),None),
                     ('tiny_positive_scaling',(base[0]*1e-16,base[1],base[2]*1e-16),None),
                     ('double_cut_quadrature',base,128)):
    p,s=estimate(model,8,4,Q,seconds=deadline-time.process_time())
    result['controls'].append(dict(kind=kind,difference=p-reference,stats=s))
    assert abs(p-reference)<1e-8,(kind,p-reference)
for i,model in enumerate(models):
    exact=truth['seeds']['0']['rows'][i]['truth']
    p,s=estimate(model,8,4,seconds=deadline-time.process_time())
    iso=evaluate(model,8,True,True,deadline)
    row=dict(copy=i,reference=exact,model_sha256=hashlib.sha256(b''.join(w.astype('<f8').tobytes() for w in model)).hexdigest(),
             supported=dict(estimate=p,error=p-exact,stats=s),
             isotropic_corrected=dict(estimate=iso['estimate'],error=iso['estimate']-exact,stats=iso))
    result['rows'].append(row)
    result['total_cpu_seconds']=time.process_time()-start
    (HERE/'near_tie_copies_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(copy=i,supported_error=p-exact,isotropic_error=iso['estimate']-exact)),flush=True)
result['aggregate']={method:dict(rmse=float(np.sqrt(np.mean([r[method]['error']**2 for r in result['rows']]))),
     mean_cpu=float(np.mean([r[method]['stats']['cpu_seconds'] for r in result['rows']])),
     mean_squared_error_times_cpu=float(np.mean([r[method]['error']**2*r[method]['stats']['cpu_seconds'] for r in result['rows']])))
     for method in ('supported','isotropic_corrected')}
result['cpu_adjusted_squared_error_ratio']=result['aggregate']['supported']['mean_squared_error_times_cpu']/result['aggregate']['isotropic_corrected']['mean_squared_error_times_cpu']
(HERE/'near_tie_copies_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['aggregate']))
