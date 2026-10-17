"""Validation only: saved exact references never enter the estimator."""
import json,time,hashlib,resource
from pathlib import Path
import numpy as np
from inventory import load,noisy
from suffix import calculate
BASE=Path(__file__).parent;ROOT=Path('/workspace/clean-neurons/m16/approaches')
start=time.process_time();cases=[];base4,sha4=load(4);base5,sha5=load(5)
refs4=json.loads((ROOT/'small_h2_34/cone_ensemble_t4.json').read_text())['rows']
rng=np.random.default_rng(47001)
models4=[base4]+[tuple(w+.05*rng.normal(size=w.shape) for w in base4) for _ in range(8)]
for i,(model,ref) in enumerate(zip(models4,refs4)):
 cases.append((f'T4_copy{i}',model,ref['reference'],ref['reference_integration_only_bound']))
refs5=json.loads((ROOT/'small_h2_56/odd_tetra/counted_full.json').read_text())['models']
for sigma,ref in zip((0.,.0025,.02),refs5):
 cases.append((f'T5_sigma{sigma}',base5 if sigma==0 else noisy(base5,sigma),ref['estimate'],float(ref['integration']['conditional_outer_bound'])+ref['integration']['positive_inner_bound']))
out={'cases':[],'checkpoint_sha256':{'T4':sha4,'T5':sha5},'weight_preparation_cpu':time.process_time()-start}
for name,model,reference,refbound in cases:
 row=dict(name=name,reference=reference,reference_bound=refbound)
 try:
  value,stats=calculate(model,16);fine,fstats=calculate(model,32)
  row.update(estimate=value,stats=stats,refinement_estimate=fine,refinement_difference=abs(value-fine),
             reference_difference=value-reference,reference_error_ceiling=abs(value-reference)+refbound)
  variance=reference*(1-reference)/stats['forward_equivalents']
  row['mse_ratio']=(value-reference)**2/variance
  row['reference_ceiling_mse_ratio']=row['reference_error_ceiling']**2/variance
 except Exception as exc:row['failure']=repr(exc)
 out['cases'].append(row);out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'benchmark.json').write_text(json.dumps(out,indent=2)+'\n')
 print(name,row.get('estimate'),row.get('reference_difference'),row.get('mse_ratio'),row.get('refinement_difference'),row.get('failure'),flush=True)
